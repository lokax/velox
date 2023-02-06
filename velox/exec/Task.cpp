/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <string>

#include "velox/codegen/Codegen.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/time/Timer.h"
#include "velox/exec/CrossJoinBuild.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/LocalPlanner.h"
#include "velox/exec/Merge.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/exec/Task.h"
#if CODEGEN_ENABLED == 1
#include "velox/experimental/codegen/CodegenLogger.h"
#endif

namespace facebook::velox::exec {

namespace {

folly::Synchronized<std::vector<std::shared_ptr<TaskListener>>>& listeners() {
  static folly::Synchronized<std::vector<std::shared_ptr<TaskListener>>>
      kListeners;
  return kListeners;
}

std::string errorMessageImpl(const std::exception_ptr& exception) {
  if (!exception) {
    return "";
  }
  std::string message;
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception& e) {
    message = e.what();
  } catch (...) {
    message = "<Unknown exception type>";
  }
  return message;
}

} // namespace

std::atomic<uint64_t> Task::numCreatedTasks_ = 0;
std::atomic<uint64_t> Task::numDeletedTasks_ = 0;

bool registerTaskListener(std::shared_ptr<TaskListener> listener) {
  return listeners().withWLock([&](auto& listeners) {
    for (const auto& existingListener : listeners) {
      if (existingListener == listener) {
        // Listener already registered. Do not register again.
        return false;
      }
    }
    listeners.push_back(std::move(listener));
    return true;
  });
}

bool unregisterTaskListener(const std::shared_ptr<TaskListener>& listener) {
  return listeners().withWLock([&](auto& listeners) {
    for (auto it = listeners.begin(); it != listeners.end(); ++it) {
      if ((*it) == listener) {
        listeners.erase(it);
        return true;
      }
    }

    // Listener not found.
    return false;
  });
}

namespace {
void collectSplitPlanNodeIds(
    const core::PlanNode* planNode,
    std::unordered_set<core::PlanNodeId>& allIds,
    std::unordered_set<core::PlanNodeId>& sourceIds) {
  bool ok = allIds.insert(planNode->id()).second;
  VELOX_USER_CHECK(
      ok,
      "Plan node IDs must be unique. Found duplicate ID: {}.",
      planNode->id());

  // Check if planNode is a leaf node in the plan tree. If so, it is a source
  // node and may use splits for processing.
  if (planNode->sources().empty()) {
    // Not all leaf nodes require splits. ValuesNode doesn't. Check if this plan
    // node requires splits.
    if (planNode->requiresSplits()) {
      sourceIds.insert(planNode->id());
    }
    return;
  }

  for (const auto& child : planNode->sources()) {
    collectSplitPlanNodeIds(child.get(), allIds, sourceIds);
  }
}

/// Returns a set IDs of source (leaf) plan nodes that require splits. Also,
/// checks that plan node IDs are unique and throws if encounters duplicates.
std::unordered_set<core::PlanNodeId> collectSplitPlanNodeIds(
    const std::shared_ptr<const core::PlanNode>& planNode) {
  std::unordered_set<core::PlanNodeId> allIds;
  std::unordered_set<core::PlanNodeId> sourceIds;
  collectSplitPlanNodeIds(planNode.get(), allIds, sourceIds);
  return sourceIds;
}

} // namespace

Task::Task(
    const std::string& taskId,
    core::PlanFragment planFragment,
    int destination,
    std::shared_ptr<core::QueryCtx> queryCtx,
    Consumer consumer,
    std::function<void(std::exception_ptr)> onError)
    : Task{
          taskId,
          std::move(planFragment),
          destination,
          std::move(queryCtx),
          (consumer ? [c = std::move(consumer)]() { return c; }
                    : ConsumerSupplier{}),
          std::move(onError)} {}

namespace {
std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}
} // namespace

Task::Task(
    const std::string& taskId,
    core::PlanFragment planFragment,
    int destination,
    std::shared_ptr<core::QueryCtx> queryCtx,
    ConsumerSupplier consumerSupplier,
    std::function<void(std::exception_ptr)> onError)
    : uuid_{makeUuid()},
      taskId_(taskId),
      planFragment_(std::move(planFragment)),
      destination_(destination),
      queryCtx_(std::move(queryCtx)),
      pool_(
          queryCtx_->pool()->addChild(fmt::format("task.{}", taskId_.c_str()))),
      splitPlanNodeIds_(collectSplitPlanNodeIds(planFragment_.planNode)),
      consumerSupplier_(std::move(consumerSupplier)),
      onError_(onError),
      bufferManager_(PartitionedOutputBufferManager::getInstance()) {
  auto memoryUsageTracker = pool_->getMemoryUsageTracker();
  if (memoryUsageTracker) {
    memoryUsageTracker->setMakeMemoryCapExceededMessage(
        [&](memory::MemoryUsageTracker& tracker) {
          VELOX_DCHECK(pool()->getMemoryUsageTracker().get() == &tracker);
          return this->getErrorMsgOnMemCapExceeded(tracker);
        });
  }
}

Task::~Task() {
  try {
    if (hasPartitionedOutput_) {
      if (auto bufferManager = bufferManager_.lock()) {
        bufferManager->removeTask(taskId_);
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught exception in ~Task(): " << e.what();
  }
}

velox::memory::MemoryPool* FOLLY_NONNULL
Task::getOrAddNodePool(const core::PlanNodeId& planNodeId) {
    // 如果这个PlanNode存再内存池则直接返回
  if (nodePools_.count(planNodeId) == 1) {
    return nodePools_[planNodeId];
  }
    // 给当前planNode添加一个内存池
  childPools_.push_back(pool_->addChild(fmt::format("node.{}", planNodeId)));
  auto* nodePool = childPools_.back().get();
  auto parentTracker = pool_->getMemoryUsageTracker();
  if (parentTracker != nullptr) {
    nodePool->setMemoryUsageTracker(parentTracker->addChild());
  }
  return nodePool;
}
// 为算子添加一个内存池？
velox::memory::MemoryPool* FOLLY_NONNULL Task::addOperatorPool(
    const core::PlanNodeId& planNodeId,
    int pipelineId,
    const std::string& operatorType) {
  auto* nodePool = getOrAddNodePool(planNodeId);
  childPools_.push_back(nodePool->addChild(
      fmt::format("op.{}.{}.{}", planNodeId, pipelineId, operatorType)));
  return childPools_.back().get();
}

memory::MemoryAllocator* FOLLY_NONNULL Task::addOperatorMemory(
    const std::shared_ptr<memory::MemoryUsageTracker>& tracker) {
  auto allocator = queryCtx_->allocator()->addChild(tracker);
  childAllocators_.emplace_back(allocator);
  return allocator.get();
}

bool Task::supportsSingleThreadedExecution() const {
    // driver工厂？
  std::vector<std::unique_ptr<DriverFactory>> driverFactories;
    // 如果有消费者提供商，则返回false
  if (consumerSupplier_) {
    return false;
  }

  LocalPlanner::plan(planFragment_, nullptr, &driverFactories, 1);

  for (const auto& factory : driverFactories) {
    if (!factory->supportsSingleThreadedExecution()) {
      return false;
    }
  }

  return true;
}

// 单线程api？
RowVectorPtr Task::next() {
  VELOX_CHECK_EQ(
      core::ExecutionStrategy::kUngrouped,
      planFragment_.executionStrategy,
      "Single-threaded execution supports only ungrouped execution");

  if (!splitPlanNodeIds_.empty()) {
    for (const auto& id : splitPlanNodeIds_) {
      VELOX_CHECK(
          splitsStates_[id].noMoreSplits,
          "Single-threaded execution requires all splits to be added before calling Task::next().");
    }
  }

  VELOX_CHECK_EQ(state_, kRunning, "Task has already finished processing.");

  // On first call, create the drivers.
  if (driverFactories_.empty()) {
    VELOX_CHECK_NULL(
        consumerSupplier_,
        "Single-threaded execution doesn't support delivering results to a callback");

    LocalPlanner::plan(planFragment_, nullptr, &driverFactories_, 1);

    exchangeClients_.resize(driverFactories_.size());

    for (const auto& factory : driverFactories_) {
      VELOX_CHECK(factory->supportsSingleThreadedExecution());
      numDriversPerSplitGroup_ += factory->numDrivers;
      numTotalDrivers_ += factory->numTotalDrivers;
      taskStats_.pipelineStats.emplace_back(
          factory->inputDriver, factory->outputDriver);
    }

    // Create drivers.
    auto self = shared_from_this();
    std::vector<std::shared_ptr<Driver>> drivers;
    drivers.reserve(numDriversPerSplitGroup_);
    createSplitGroupStateLocked(self, 0);
    createDriversLocked(self, 0, drivers);

    drivers_ = std::move(drivers);
  }

  // Run drivers one at a time. If a driver blocks, continue running the other
  // drivers. Running other drivers is expected to unblock some or all blocked
  // drivers.
  const auto numDrivers = drivers_.size();

  std::vector<ContinueFuture> futures;
  futures.resize(numDrivers);

  for (;;) {
    int runnableDrivers = 0;
    int blockedDrivers = 0;
    for (auto i = 0; i < numDrivers; ++i) {
      if (drivers_[i] == nullptr) {
        // This driver has finished processing.
        continue;
      }

      if (!futures[i].isReady()) {
        // This driver is still blocked.
        ++blockedDrivers;
        continue;
      }

      ++runnableDrivers;

      std::shared_ptr<BlockingState> blockingState;
      auto result = drivers_[i]->next(blockingState);
      if (result) {
        return result;
      }

      if (blockingState) {
        futures[i] = blockingState->future();
      }

      if (error()) {
        std::rethrow_exception(error());
      }
    }

    if (runnableDrivers == 0) {
      VELOX_CHECK_EQ(
          0,
          blockedDrivers,
          "Cannot make progress as all remaining drivers are blocked");
      return nullptr;
    }
  }
}

void Task::start(
    std::shared_ptr<Task> self,
    uint32_t maxDrivers,
    uint32_t concurrentSplitGroups) {
  VELOX_CHECK_GE(
      maxDrivers, 1, "maxDrivers parameter must be greater then or equal to 1");
  VELOX_CHECK_GE(
      concurrentSplitGroups,
      1,
      "concurrentSplitGroups parameter must be greater then or equal to 1");
  VELOX_CHECK(self->drivers_.empty());
  self->concurrentSplitGroups_ = concurrentSplitGroups;
  {
    std::lock_guard<std::mutex> l(self->mutex_);
    self->taskStats_.executionStartTimeMs = getCurrentTimeMs();
  }

#if CODEGEN_ENABLED == 1
  const auto& config = self->queryCtx()->config();
  if (config.codegenEnabled() &&
      config.codegenConfigurationFilePath().length() != 0) {
    auto codegenLogger =
        std::make_shared<codegen::DefaultLogger>(self->taskId_);
    auto codegen = codegen::Codegen(codegenLogger);
    auto lazyLoading = config.codegenLazyLoading();
    codegen.initializeFromFile(
        config.codegenConfigurationFilePath(), lazyLoading);
    if (auto newPlanNode = codegen.compile(*(self->planFragment_.planNode))) {
      self->planFragment_.planNode = newPlanNode;
    }
  }
#endif

  // Here we create driver factories.
  LocalPlanner::plan(
      self->planFragment_,
      self->consumerSupplier(),
      &self->driverFactories_,
      maxDrivers);

  // Keep one exchange client per pipeline (NULL if not used).
  const auto numPipelines = self->driverFactories_.size();
  self->exchangeClients_.resize(numPipelines);

  // For ungrouped execution we reuse some structures used for grouped
  // execution and assume we have "1 split".
  const uint32_t numSplitGroups =
      std::max(1, self->planFragment_.numSplitGroups);

  // For each pipeline we have a corresponding driver factory.
  // Here we count how many drivers in total we need and create
  // pipeline stats.
  for (auto& factory : self->driverFactories_) {
    self->numDriversPerSplitGroup_ += factory->numDrivers;
    self->numTotalDrivers_ += factory->numTotalDrivers;
    self->taskStats_.pipelineStats.emplace_back(
        factory->inputDriver, factory->outputDriver);
  }

  // Register self for possible memory recovery callback. Do this
  // after sizing 'drivers_' but before starting the
  // Drivers. 'drivers_' can be read by memory recovery or
  // cancellation while Drivers are being made, so the array should
  // have final size from the start.

  auto bufferManager = self->bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager,
      "Unable to initialize task. "
      "PartitionedOutputBufferManager was already destructed");

  // In this loop we prepare the global state of pipelines: partitioned output
  // buffer and exchange client(s).
  for (auto pipeline = 0; pipeline < numPipelines; ++pipeline) {
    auto& factory = self->driverFactories_[pipeline];

    if (auto partitionedOutputNode = factory->needsPartitionedOutput()) {
      self->numDriversInPartitionedOutput_ = factory->numDrivers;
      VELOX_CHECK(
          !self->hasPartitionedOutput_,
          "Only one output pipeline per task is supported");
      self->hasPartitionedOutput_ = true;
      bufferManager->initializeTask(
          self,
          partitionedOutputNode->isBroadcast(),
          partitionedOutputNode->numPartitions(),
          self->numDriversInPartitionedOutput_ * numSplitGroups);
    }

    if (auto exchangeNodeId = factory->needsExchangeClient()) {
      // Low-water mark for filling the exchange queue is 1/2 of the per worker
      // buffer size of the producers.
      self->exchangeClients_[pipeline] = std::make_shared<ExchangeClient>(
          self->destination_,
          self->queryCtx()->config().maxPartitionedOutputBufferSize() / 2);

      self->exchangeClientByPlanNode_.emplace(
          exchangeNodeId.value(), self->exchangeClients_[pipeline]);
    }
  }

  std::unique_lock<std::mutex> l(self->mutex_);

  // For grouped execution we postpone driver creation up until the splits start
  // arriving, as we don't know what split groups we are going to get.
  // Here we create Drivers only for ungrouped (normal) execution.
  if (self->isUngroupedExecution()) {
    // Create the drivers we are going to run for this task.
    std::vector<std::shared_ptr<Driver>> drivers;
    drivers.reserve(self->numDriversPerSplitGroup_);
    self->createSplitGroupStateLocked(self, 0);
    self->createDriversLocked(self, 0, drivers);

    // Set and start all Drivers together inside 'mutex_' so that cancellations
    // and pauses have well defined timing. For example, do not pause and
    // restart a task while it is still adding Drivers.
    // If the given executor is folly::InlineLikeExecutor (or it's child), since
    // the drivers will be executed synchronously on the same thread as the
    // current task, so we need release the lock to avoid the deadlock.
    self->drivers_ = std::move(drivers);
    if (dynamic_cast<const folly::InlineLikeExecutor*>(
            self->queryCtx()->executor())) {
      l.unlock();
    }
    for (auto& driver : self->drivers_) {
      if (driver) {
        ++self->numRunningDrivers_;
        // 把driver放进任务队列中，等待线程池中的工作线程进行处理
        Driver::enqueue(driver);
      }
    }
  } else {
    // Preallocate a bunch of slots for max concurrent drivers during grouped
    // execution.
    self->drivers_.resize(
        self->numDriversPerSplitGroup_ * self->concurrentSplitGroups_);

    // As some splits could have been added before the task start, ensure we
    // start running drivers for them.
    self->ensureSplitGroupsAreBeingProcessedLocked(self);
  }
}

// static
void Task::resume(std::shared_ptr<Task> self) {
  VELOX_CHECK(!self->exception_, "Cannot resume failed task");
  // 获取任务的互斥锁
  std::lock_guard<std::mutex> l(self->mutex_);
  // Setting pause requested must be atomic with the resuming so that
  // suspended sections do not go back on thread during resume.
  // 设置暂停请求为false
  self->requestPauseLocked(false);
  // 遍历每一个driver
  for (auto& driver : self->drivers_) {
    if (driver) {
        // 这个暂时没懂
      if (driver->state().isSuspended) {
        // The Driver will come on thread in its own time as long as
        // the cancel flag is reset. This check needs to be inside 'mutex_'.
        continue;
      }
      // 在被暂停的时候，该driver在队列中，这时候不会添加两次进去
      if (driver->state().isEnqueued) {
        // A Driver can wait for a thread and there can be a
        // pause/resume during the wait. The Driver should not be
        // enqueued twice.
        continue;
      }
      VELOX_CHECK(!driver->isOnThread() && !driver->isTerminated());
      // 如果这里有阻塞发future的话，说明该driver在等待一些事件发生
      // 等到这些事件发生后，又futture设置的回调去放到队列中
      // 而不是由这个函数来放进队列
      // 注意hasBlockingFuture是在任务的互斥锁下进行保护的
      if (!driver->state().hasBlockingFuture) {
        // Do not continue a Driver that is blocked on external
        // event. The Driver gets enqueued by the promise realization.
        Driver::enqueue(driver);
      }
    }
  }
}
// 创建split分组状态
void Task::createSplitGroupStateLocked(
    std::shared_ptr<Task>& self,
    uint32_t splitGroupId) {
  // In this loop we prepare per split group pipelines structures:
  // local exchanges and join bridges.
  const auto numPipelines = self->driverFactories_.size();
  for (auto pipeline = 0; pipeline < numPipelines; ++pipeline) {
    auto& factory = self->driverFactories_[pipeline];

    auto exchangeId = factory->needsLocalExchange();
    if (exchangeId.has_value()) {
        // 创建队列
      self->createLocalExchangeQueuesLocked(
          splitGroupId, exchangeId.value(), factory->numDrivers);
    }

    self->addHashJoinBridgesLocked(
        splitGroupId, factory->needsHashJoinBridges());
    self->addCrossJoinBridgesLocked(
        splitGroupId, factory->needsCrossJoinBridges());
    self->addCustomJoinBridgesLocked(splitGroupId, factory->planNodes);
  }
}

void Task::createDriversLocked(
    std::shared_ptr<Task>& self,
    uint32_t splitGroupId,
    std::vector<std::shared_ptr<Driver>>& out) {
  auto& splitGroupState = self->splitGroupStates_[splitGroupId];
  // 有几条pipeline
  const auto numPipelines = driverFactories_.size();
  for (auto pipeline = 0; pipeline < numPipelines; ++pipeline) {
    auto& factory = driverFactories_[pipeline];
    const uint32_t driverIdOffset = factory->numDrivers * splitGroupId;
    for (uint32_t partitionId = 0; partitionId < factory->numDrivers;
         ++partitionId) {
      out.emplace_back(factory->createDriver(
          std::make_unique<DriverCtx>(
              self,
              driverIdOffset + partitionId,
              pipeline,
              splitGroupId,
              partitionId),
          self->exchangeClients_[pipeline],
          [self](size_t i) {
            return i < self->driverFactories_.size()
                ? self->driverFactories_[i]->numTotalDrivers
                : 0;
          }));
      ++splitGroupState.numRunningDrivers;
    }
  }
  noMoreLocalExchangeProducers(splitGroupId);
  // 正在运行的split分组
  ++numRunningSplitGroups_;

  // Initialize operator stats using the 1st driver of each operator.
  if (not initializedOpStats_) {
    initializedOpStats_ = true;
    size_t driverIndex{0};
    for (auto pipeline = 0; pipeline < numPipelines; ++pipeline) {
      auto& factory = self->driverFactories_[pipeline];
      out[driverIndex]->initializeOperatorStats(
          self->taskStats_.pipelineStats[pipeline].operatorStats);
      driverIndex += factory->numDrivers;
    }
  }

  // Start all the join bridges before we start driver execution.
  for (auto& bridgeEntry : splitGroupState.bridges) {
    bridgeEntry.second->start();
  }

  // Start all the spill groups before we start the driver execution.
  for (auto& coordinatorEntry : splitGroupState.spillOperatorGroups) {
    coordinatorEntry.second->start();
  }
}

// static
void Task::removeDriver(std::shared_ptr<Task> self, Driver* driver) {
  bool foundDriver = false;
  bool allFinished = true;
  {
    // 加锁
    std::lock_guard<std::mutex> taskLock(self->mutex_);
    // 遍历该任务的每一个driver
    for (auto& driverPtr : self->drivers_) {
      if (driverPtr.get() != driver) {
        continue;
      }
        // 找到相等的driver
      // Mark the closure of another driver for its split group (even in
      // ungrouped execution mode).
      const auto splitGroupId = driver->driverCtx()->splitGroupId;
      auto& splitGroupState = self->splitGroupStates_[splitGroupId];
      // 减少运行的driver
      --splitGroupState.numRunningDrivers;
        // pipeline的id号
      auto pipelineId = driver->driverCtx()->pipelineId;
        // output pipeline是什么
      if (self->isOutputPipeline(pipelineId)) {
        ++splitGroupState.numFinishedOutputDrivers;
      }

      // Release the driver, note that after this 'driver' is invalid.
      driverPtr = nullptr;
      self->driverClosedLocked();

      allFinished = self->checkIfFinishedLocked();

      // Check if a split group is finished.
      if (splitGroupState.numRunningDrivers == 0) {
        if (self->isGroupedExecution()) {
          --self->numRunningSplitGroups_;
          self->taskStats_.completedSplitGroups.emplace(splitGroupId);
          splitGroupState.clear();
          self->ensureSplitGroupsAreBeingProcessedLocked(self);
        } else {
          splitGroupState.clear();
        }
      }
      foundDriver = true;
      break;
    }
  }

  if (!foundDriver) {
    LOG(WARNING) << "Trying to remove a Driver twice from its Task";
  }

  if (allFinished) {
    self->terminate(TaskState::kFinished);
  }
}

void Task::ensureSplitGroupsAreBeingProcessedLocked(
    std::shared_ptr<Task>& self) {
  // Only try creating more drivers if we are running.
  if (not isRunningLocked() or (numDriversPerSplitGroup_ == 0)) {
    return;
  }

  while (numRunningSplitGroups_ < concurrentSplitGroups_ and
         not queuedSplitGroups_.empty()) {
    const uint32_t splitGroupId = queuedSplitGroups_.front();
    queuedSplitGroups_.pop();

    std::vector<std::shared_ptr<Driver>> drivers;
    drivers.reserve(numDriversPerSplitGroup_);
    createSplitGroupStateLocked(self, splitGroupId);
    createDriversLocked(self, splitGroupId, drivers);
    // Move created drivers into the vacant spots in 'drivers_' and enqueue
    // them. We have vacant spots, because we initially allocate enough items in
    // the vector and keep null pointers for completed drivers.
    size_t i = 0;
    for (auto& newDriverPtr : drivers) {
      while (drivers_[i] != nullptr) {
        VELOX_CHECK_LT(i, drivers_.size());
        ++i;
      }
      auto& targetPtr = drivers_[i];
      targetPtr = std::move(newDriverPtr);
      if (targetPtr) {
        ++numRunningDrivers_;
        Driver::enqueue(targetPtr);
      }
    }
  }
}

void Task::setMaxSplitSequenceId(
    const core::PlanNodeId& planNodeId,
    long maxSequenceId) {
  checkPlanNodeIdForSplit(planNodeId);

  std::lock_guard<std::mutex> l(mutex_);
  if (isRunningLocked()) {
    auto& splitsState = splitsStates_[planNodeId];
    // We could have been sent an old split again, so only change max id, when
    // the new one is greater.
    splitsState.maxSequenceId =
        std::max(splitsState.maxSequenceId, maxSequenceId);
  }
}

bool Task::addSplitWithSequence(
    const core::PlanNodeId& planNodeId,
    exec::Split&& split,
    long sequenceId) {
  checkPlanNodeIdForSplit(planNodeId);
  std::unique_ptr<ContinuePromise> promise;
  bool added = false;
  bool isTaskRunning;
  {
    std::lock_guard<std::mutex> l(mutex_);
    isTaskRunning = isRunningLocked();
    if (isTaskRunning) {
      // The same split can be added again in some systems. The systems that
      // want 'one split processed once only' would use this method and
      // duplicate splits would be ignored.
      auto& splitsState = splitsStates_[planNodeId];
      if (sequenceId > splitsState.maxSequenceId) {
        promise = addSplitLocked(splitsState, std::move(split));
        added = true;
      }
    }
  }

  if (promise) {
    promise->setValue();
  }

  if (!isTaskRunning) {
    addRemoteSplit(planNodeId, split);
  }

  return added;
}

void Task::addSplit(const core::PlanNodeId& planNodeId, exec::Split&& split) {
  checkPlanNodeIdForSplit(planNodeId);
  bool isTaskRunning;
  std::unique_ptr<ContinuePromise> promise;
  {
    // 加锁
    std::lock_guard<std::mutex> l(mutex_);
    // 任务是否在运行
    isTaskRunning = isRunningLocked();
    if (isTaskRunning) {
      promise = addSplitLocked(splitsStates_[planNodeId], std::move(split));
    }
  }

  if (promise) {
    promise->setValue();
  }

  if (!isTaskRunning) {
    addRemoteSplit(planNodeId, split);
  }
}

void Task::addRemoteSplit(
    const core::PlanNodeId& planNodeId,
    const exec::Split& split) {
  if (split.hasConnectorSplit()) {
    if (exchangeClientByPlanNode_.count(planNodeId)) {
      auto remoteSplit =
          std::dynamic_pointer_cast<RemoteConnectorSplit>(split.connectorSplit);
      VELOX_CHECK(remoteSplit, "Wrong type of split");
      exchangeClientByPlanNode_[planNodeId]->addRemoteTaskId(
          remoteSplit->taskId);
    }
  }
}

void Task::checkPlanNodeIdForSplit(const core::PlanNodeId& id) const {
  VELOX_USER_CHECK(
      splitPlanNodeIds_.find(id) != splitPlanNodeIds_.end(),
      "Splits can be associated only with leaf plan nodes which require splits. Plan node ID {} doesn't refer to such plan node.",
      id);
}

std::unique_ptr<ContinuePromise> Task::addSplitLocked(
    SplitsState& splitsState,
    exec::Split&& split) {
  ++taskStats_.numTotalSplits;
  ++taskStats_.numQueuedSplits;

  if (isUngroupedExecution()) {
    VELOX_DCHECK(
        not split.hasGroup(), "Got split group for ungrouped execution!");
    return addSplitToStoreLocked(
        splitsState.groupSplitsStores[0], std::move(split));
  } else {
    VELOX_CHECK(split.hasGroup(), "Missing split group for grouped execution!");
    const auto splitGroupId = split.groupId; // Avoid eval order c++ warning.
    // If this is the 1st split from this group, add the split group to queue.
    // Also add that split group to the set of 'seen' split groups.
    if (seenSplitGroups_.find(splitGroupId) == seenSplitGroups_.end()) {
      seenSplitGroups_.emplace(splitGroupId);
      queuedSplitGroups_.push(splitGroupId);
      auto self = shared_from_this();
      // We might have some free driver slots to process this split group.
      ensureSplitGroupsAreBeingProcessedLocked(self);
    }
    return addSplitToStoreLocked(
        splitsState.groupSplitsStores[splitGroupId], std::move(split));
  }
}

std::unique_ptr<ContinuePromise> Task::addSplitToStoreLocked(
    SplitsStore& splitsStore,
    exec::Split&& split) {
  splitsStore.splits.push_back(split);
  if (not splitsStore.splitPromises.empty()) {
    auto promise = std::make_unique<ContinuePromise>(
        std::move(splitsStore.splitPromises.back()));
    splitsStore.splitPromises.pop_back();
    return promise;
  }
  return nullptr;
}

void Task::noMoreSplitsForGroup(
    const core::PlanNodeId& planNodeId,
    int32_t splitGroupId) {
  checkPlanNodeIdForSplit(planNodeId);
  std::vector<ContinuePromise> promises;
  {
    // 加锁
    std::lock_guard<std::mutex> l(mutex_);
    // split状态？
    auto& splitsState = splitsStates_[planNodeId];
    auto& splitsStore = splitsState.groupSplitsStores[splitGroupId];
    // 设置没有更多的split
    splitsStore.noMoreSplits = true;
    promises = std::move(splitsStore.splitPromises);

    // There were no splits in this group, hence, no active drivers. Mark the
    // group complete.
    if (seenSplitGroups_.count(splitGroupId) == 0) {
      taskStats_.completedSplitGroups.insert(splitGroupId);
    }
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void Task::noMoreSplits(const core::PlanNodeId& planNodeId) {
  checkPlanNodeIdForSplit(planNodeId);
  std::vector<ContinuePromise> splitPromises;
  bool allFinished;
  std::shared_ptr<ExchangeClient> exchangeClient;
  {
    std::lock_guard<std::mutex> l(mutex_);

    // Global 'no more splits' for a plan node comes in case of ungrouped
    // execution when no more splits will arrive. For grouped execution it
    // comes when no more split groups will arrive for that plan node.
    auto& splitsState = splitsStates_[planNodeId];
    splitsState.noMoreSplits = true;
    if (not splitsState.groupSplitsStores.empty()) {
      // Mark all split stores as 'no more splits'.
      for (auto& it : splitsState.groupSplitsStores) {
        it.second.noMoreSplits = true;
        // 这里感觉不对
        splitPromises = std::move(it.second.splitPromises);
      }
    } else if (isUngroupedExecution()) {
      // During ungrouped execution, in the unlikely case there are no split
      // stores (this means there were no splits at all), we create one.
      splitsState.groupSplitsStores.emplace(0, SplitsStore{{}, true, {}});
    }

    allFinished = checkNoMoreSplitGroupsLocked();

    if (!isRunningLocked() && exchangeClientByPlanNode_.count(planNodeId)) {
      exchangeClient = exchangeClientByPlanNode_[planNodeId];
      exchangeClientByPlanNode_.erase(planNodeId);
    }
  }

  for (auto& promise : splitPromises) {
    promise.setValue();
  }

  if (exchangeClient) {
    exchangeClient->noMoreRemoteTasks();
  }

  if (allFinished) {
    terminate(kFinished);
  }
}

bool Task::checkNoMoreSplitGroupsLocked() {
  if (isUngroupedExecution()) {
    return false;
  }

  // For grouped execution, when all plan nodes have 'no more splits' coming,
  // we should review the total number of drivers, which initially is set to
  // process all split groups, but in reality workers share split groups and
  // each worker processes only a part of them, meaning much less than all.
  bool noMoreSplitGroups = true;
  for (auto& it : splitsStates_) {
    if (not it.second.noMoreSplits) {
      noMoreSplitGroups = false;
      break;
    }
  }
  if (noMoreSplitGroups) {
    numTotalDrivers_ = seenSplitGroups_.size() * numDriversPerSplitGroup_;
    if (hasPartitionedOutput_) {
      auto bufferManager = bufferManager_.lock();
      bufferManager->updateNumDrivers(
          taskId(), numDriversInPartitionedOutput_ * seenSplitGroups_.size());
    }

    return checkIfFinishedLocked();
  }

  return false;
}

bool Task::isAllSplitsFinishedLocked() {
  if (taskStats_.numFinishedSplits == taskStats_.numTotalSplits) {
    for (const auto& it : splitsStates_) {
      if (not it.second.noMoreSplits) {
        return false;
      }
    }
    return true;
  }
  return false;
}

BlockingReason Task::getSplitOrFuture(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId,
    exec::Split& split,
    ContinueFuture& future) {
  std::lock_guard<std::mutex> l(mutex_);

  auto& splitsState = splitsStates_[planNodeId];

  if (isUngroupedExecution()) {
    return getSplitOrFutureLocked(
        splitsState.groupSplitsStores[0], split, future);
  } else {
    return getSplitOrFutureLocked(
        splitsState.groupSplitsStores[splitGroupId], split, future);
  }
}
// 看过
BlockingReason Task::getSplitOrFutureLocked(
    SplitsStore& splitsStore,
    exec::Split& split,
    ContinueFuture& future) {
        // 如果没有split
  if (splitsStore.splits.empty()) {
    if (splitsStore.noMoreSplits) {
        // 不再需要split，返回不阻塞
      return BlockingReason::kNotBlocked;
    }
    // 创建future和promise
    auto [splitPromise, splitFuture] = makeVeloxContinuePromiseContract(
        fmt::format("Task::getSplitOrFuture {}", taskId_));
    future = std::move(splitFuture);
    // 保存promise以便将来有split的时候进行通知
    splitsStore.splitPromises.push_back(std::move(splitPromise));
    return BlockingReason::kWaitForSplit;
  }

  split = getSplitLocked(splitsStore);
  return BlockingReason::kNotBlocked;
}
// 看过
exec::Split Task::getSplitLocked(SplitsStore& splitsStore) {
    // 从队列中弹出第一个split
  auto split = std::move(splitsStore.splits.front());
  splitsStore.splits.pop_front();

  --taskStats_.numQueuedSplits;
  ++taskStats_.numRunningSplits;
  taskStats_.lastSplitStartTimeMs = getCurrentTimeMs();
  if (taskStats_.firstSplitStartTimeMs == 0) {
    taskStats_.firstSplitStartTimeMs = taskStats_.lastSplitStartTimeMs;
  }

  return split;
}

void Task::splitFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  ++taskStats_.numFinishedSplits;
  --taskStats_.numRunningSplits;
  if (isAllSplitsFinishedLocked()) {
    taskStats_.executionEndTimeMs = getCurrentTimeMs();
  }
}

void Task::multipleSplitsFinished(int32_t numSplits) {
  std::lock_guard<std::mutex> l(mutex_);
  taskStats_.numFinishedSplits += numSplits;
  taskStats_.numRunningSplits -= numSplits;
  if (isAllSplitsFinishedLocked()) {
    taskStats_.executionEndTimeMs = getCurrentTimeMs();
  }
}

bool Task::isGroupedExecution() const {
  return planFragment_.isGroupedExecution();
}

bool Task::isUngroupedExecution() const {
  return not isGroupedExecution();
}
// 任务正在运行
bool Task::isRunning() const {
  std::lock_guard<std::mutex> l(mutex_);
  return (state_ == TaskState::kRunning);
}
// 任务完成，但是这里为什么要加锁
bool Task::isFinished() const {
  std::lock_guard<std::mutex> l(mutex_);
  return (state_ == TaskState::kFinished);
}
// 任务正在运行
bool Task::isRunningLocked() const {
  return (state_ == TaskState::kRunning);
}
// 任务已经完成
bool Task::isFinishedLocked() const {
  return (state_ == TaskState::kFinished);
}

void Task::updateBroadcastOutputBuffers(int numBuffers, bool noMoreBuffers) {
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager,
      "Unable to initialize task. "
      "PartitionedOutputBufferManager was already destructed");

  {
    std::lock_guard<std::mutex> l(mutex_);
    if (noMoreBroadcastBuffers_) {
      // Ignore messages received after no-more-buffers message.
      return;
    }

    if (noMoreBuffers) {
      noMoreBroadcastBuffers_ = true;
    }
  }

  bufferManager->updateBroadcastOutputBuffers(
      taskId_, numBuffers, noMoreBuffers);
}

int Task::getOutputPipelineId() const {
  for (auto i = 0; i < driverFactories_.size(); ++i) {
    if (driverFactories_[i]->outputDriver) {
      return i;
    }
  }

  VELOX_FAIL("Output pipeline not found");
}

void Task::setAllOutputConsumed() {
  bool allFinished;
  {
    std::lock_guard<std::mutex> l(mutex_);
    partitionedOutputConsumed_ = true;
    allFinished = checkIfFinishedLocked();
  }

  if (allFinished) {
    terminate(TaskState::kFinished);
  }
}

void Task::driverClosedLocked() {
  if (isRunningLocked()) {
    --numRunningDrivers_;
  }
  // 增加完成的driver
  ++numFinishedDrivers_;
}

bool Task::checkIfFinishedLocked() {
    // 如果没有处于Running状态，则返回false，这种情况可能被取消，出错之类的
    // 自然不能表示Finished状态
  if (!isRunningLocked()) {
    return false;
  }

  // TODO Add support for terminating processing early in grouped execution.
  bool allFinished = numFinishedDrivers_ == numTotalDrivers_;
  if (!allFinished && isUngroupedExecution()) {
    auto outputPipelineId = getOutputPipelineId();
    if (splitGroupStates_[0].numFinishedOutputDrivers ==
        numDrivers(outputPipelineId)) {
      allFinished = true;

      if (taskStats_.executionEndTimeMs == 0) {
        // In case we haven't set executionEndTimeMs due to all splits
        // depleted, we set it here. This can happen due to task error or task
        // being cancelled.
        taskStats_.executionEndTimeMs = getCurrentTimeMs();
      }
    }
  }

  if (allFinished) {
    // 如果没有分区输出或者分区输出已经被消费了？
    if ((not hasPartitionedOutput_) || partitionedOutputConsumed_) {
      taskStats_.endTimeMs = getCurrentTimeMs();
      return true;
    }
  }
    // 如果有分区没有被消费，是不能进入Finished状态的
  return false;
}

bool Task::allPeersFinished(
    const core::PlanNodeId& planNodeId,
    Driver* caller,
    ContinueFuture* future,
    std::vector<ContinuePromise>& promises,
    std::vector<std::shared_ptr<Driver>>& peers) {
        // 加锁
  std::lock_guard<std::mutex> l(mutex_);
  if (exception_) {
    // 抛异常
    VELOX_FAIL(
        "Task is terminating because of error: {}",
        errorMessageImpl(exception_));
  }
  const auto splitGroupId = caller->driverCtx()->splitGroupId;
  auto& barriers = splitGroupStates_[splitGroupId].barriers;
  auto& state = barriers[planNodeId];

  const auto numPeers = numDrivers(caller->driverCtx()->pipelineId);
  if (++state.numRequested == numPeers) {
    peers = std::move(state.drivers);
    promises = std::move(state.promises);
    barriers.erase(planNodeId);
    return true;
  }
  std::shared_ptr<Driver> callerShared;
  // 遍历每一个driver
  for (auto& driver : drivers_) {
    if (driver.get() == caller) {
      callerShared = driver;
      break;
    }
  }
  VELOX_CHECK(
      callerShared, "Caller of Task::allPeersFinished is not a valid Driver");
      // 把driver推进去？
  state.drivers.push_back(callerShared);
  // 创建promises和future
  state.promises.emplace_back(
      fmt::format("Task::allPeersFinished {}", taskId_));
  *future = state.promises.back().getSemiFuture();

  return false;
}

void Task::addHashJoinBridgesLocked(
    uint32_t splitGroupId,
    const std::vector<core::PlanNodeId>& planNodeIds) {
  auto& splitGroupState = splitGroupStates_[splitGroupId];
  for (const auto& planNodeId : planNodeIds) {
    splitGroupState.bridges.emplace(
        planNodeId, std::make_shared<HashJoinBridge>());
    splitGroupState.spillOperatorGroups.emplace(
        planNodeId,
        std::make_unique<SpillOperatorGroup>(
            taskId_, splitGroupId, planNodeId));
  }
}

void Task::addCustomJoinBridgesLocked(
    uint32_t splitGroupId,
    const std::vector<core::PlanNodePtr>& planNodes) {
  auto& splitGroupState = splitGroupStates_[splitGroupId];
  for (const auto& planNode : planNodes) {
    if (auto joinBridge = Operator::joinBridgeFromPlanNode(planNode)) {
      splitGroupState.bridges.emplace(planNode->id(), std::move(joinBridge));
      return;
    }
  }
}

std::shared_ptr<JoinBridge> Task::getCustomJoinBridge(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  return getJoinBridgeInternal<JoinBridge>(splitGroupId, planNodeId);
}

void Task::addCrossJoinBridgesLocked(
    uint32_t splitGroupId,
    const std::vector<core::PlanNodeId>& planNodeIds) {
  auto& splitGroupState = splitGroupStates_[splitGroupId];
  for (const auto& planNodeId : planNodeIds) {
    splitGroupState.bridges.emplace(
        planNodeId, std::make_shared<CrossJoinBridge>());
  }
}

std::shared_ptr<HashJoinBridge> Task::getHashJoinBridge(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  return getJoinBridgeInternal<HashJoinBridge>(splitGroupId, planNodeId);
}

std::shared_ptr<HashJoinBridge> Task::getHashJoinBridgeLocked(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  return getJoinBridgeInternalLocked<HashJoinBridge>(splitGroupId, planNodeId);
}

std::shared_ptr<CrossJoinBridge> Task::getCrossJoinBridge(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  return getJoinBridgeInternal<CrossJoinBridge>(splitGroupId, planNodeId);
}

template <class TBridgeType>
std::shared_ptr<TBridgeType> Task::getJoinBridgeInternal(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  std::lock_guard<std::mutex> l(mutex_);
  return getJoinBridgeInternalLocked<TBridgeType>(splitGroupId, planNodeId);
}

template <class TBridgeType>
std::shared_ptr<TBridgeType> Task::getJoinBridgeInternalLocked(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  const auto& splitGroupState = splitGroupStates_[splitGroupId];

  auto it = splitGroupState.bridges.find(planNodeId);
  VELOX_CHECK(
      it != splitGroupState.bridges.end(),
      "Join bridge for plan node ID not found: {}",
      planNodeId);
  auto bridge = std::dynamic_pointer_cast<TBridgeType>(it->second);
  VELOX_CHECK_NOT_NULL(
      bridge,
      "Join bridge for plan node ID is of the wrong type: {}",
      planNodeId);
  return bridge;
}

//  static
std::string Task::shortId(const std::string& id) {
  if (id.size() < 12) {
    return id;
  }
  const char* str = id.c_str();
  const char* dot = strchr(str, '.');
  if (!dot) {
    return id;
  }
  auto hash = std::hash<std::string_view>()(std::string_view(str, dot - str));
  return fmt::format("tk:{}", hash & 0xffff);
}

/// Moves split promises from one vector to another.
static void movePromisesOut(
    std::vector<ContinuePromise>& from,
    std::vector<ContinuePromise>& to) {
  for (auto& promise : from) {
    to.push_back(std::move(promise));
  }
  from.clear();
}

ContinueFuture Task::terminate(TaskState terminalState) {
  std::vector<std::shared_ptr<Driver>> offThreadDrivers;
  TaskCompletionNotifier completionNotifier;
  {
    // 加锁
    std::lock_guard<std::mutex> l(mutex_);
    if (taskStats_.executionEndTimeMs == 0) {
      taskStats_.executionEndTimeMs = getCurrentTimeMs();
    }
    // 如果任务不处于以运行状态, 比如同时调用了terminate
    // 但是这里会获取任务的互斥锁的，所以有一个修改了任务的状态
    if (not isRunningLocked()) {
        // 所以这里也需要future
      return makeFinishFutureLocked("Task::terminate");
    }
    // 修改任务状态，这里是任务的状态
    state_ = terminalState;
    if (state_ == TaskState::kCanceled || state_ == TaskState::kAborted) {
      try {
        VELOX_FAIL(
            state_ == TaskState::kCanceled ? "Cancelled"
                                           : "Aborted for external error");
                                           // 这里是主动抛异常
      } catch (const std::exception& e) {
        // 创建一个异常，然后保存下来
        exception_ = std::current_exception();
      }
    }

    activateTaskCompletionNotifier(completionNotifier);

    // Drivers that are on thread will see this at latest when they go off
    // thread.
    // 设置这个flag后，其他在线程上的driver会退出线程
    // 如果有driver被阻塞的话，在放回队列并且被调度执行后，同样会退出，不过我觉得不用放回队列了
    terminateRequested_ = true;
    // The drivers that are on thread will go off thread in time and
    // 'numRunningDrivers_' is cleared here so that this is 0 right
    // after terminate as tests expect.
    // 这里直接清0
    numRunningDrivers_ = 0;
    for (auto& driver : drivers_) {
      if (driver) {
        // 这种情况说明driver不在线程上,调用者负责清理资源？ 
        // 这里面会设置线程的状态为terminate
        if (enterForTerminateLocked(driver->state()) ==
            StopReason::kTerminate) {
                // 保存离线线程driver
          offThreadDrivers.push_back(std::move(driver));
          driverClosedLocked();
        }
      }
    }
  }
    // 通知
  completionNotifier.notify();

  // Get the stats and free the resources of Drivers that were not on
  // thread.
  for (auto& driver : offThreadDrivers) {
    driver->closeByTask();
  }

  // We continue all Drivers waiting for promises known to the
  // Task. The Drivers are now detached from Task and therefore will
  // not go on thread. The reference in the future callback is
  // typically the last one.
  if (hasPartitionedOutput_) {
    if (auto bufferManager = bufferManager_.lock()) {
      bufferManager->removeTask(taskId_);
    }
  }

  for (auto& exchangeClient : exchangeClients_) {
    if (exchangeClient) {
      exchangeClient->close();
    }
  }

  // Release reference to exchange client, so that it will close exchange
  // sources and prevent resending requests for data.
  exchangeClients_.clear();

  std::vector<ContinuePromise> splitPromises;
  std::vector<std::shared_ptr<JoinBridge>> oldBridges;
  std::vector<SplitGroupState> splitGroupStates;
  std::
      unordered_map<core::PlanNodeId, std::pair<std::vector<exec::Split>, bool>>
          remainingRemoteSplits;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // Collect all the join bridges to clear them.
    for (auto& splitGroupState : splitGroupStates_) {
      for (auto& pair : splitGroupState.second.bridges) {
        oldBridges.emplace_back(std::move(pair.second));
      }
      splitGroupStates.push_back(std::move(splitGroupState.second));
    }

    // Collect all outstanding split promises from all splits state structures.
    for (auto& pair : splitsStates_) {
      auto& splitState = pair.second;
      for (auto& it : pair.second.groupSplitsStores) {
        movePromisesOut(it.second.splitPromises, splitPromises);
      }

      // Process remaining remote splits.
      auto exchangeClientIt = exchangeClientByPlanNode_.find(pair.first);
      if (exchangeClientIt != exchangeClientByPlanNode_.end()) {
        auto exchangeClient = exchangeClientIt->second;
        std::vector<exec::Split> splits;
        for (auto& [groupId, store] : splitState.groupSplitsStores) {
          while (!store.splits.empty()) {
            splits.emplace_back(getSplitLocked(store));
          }
        }
        if (!splits.empty()) {
          remainingRemoteSplits.emplace(
              pair.first,
              std::make_pair(std::move(splits), splitState.noMoreSplits));
        }
      }
    }
  }

  for (auto& [planNodeId, splits] : remainingRemoteSplits) {
    for (auto& split : splits.first) {
      if (!exchangeClientByPlanNode_[planNodeId]->pool()) {
        // If we terminate even before the client's initialization, we
        // initialize the client with Task's memory pool.
        exchangeClientByPlanNode_[planNodeId]->initialize(pool_.get());
      }
      addRemoteSplit(planNodeId, split);
    }
    if (splits.second) {
      exchangeClientByPlanNode_[planNodeId]->noMoreRemoteTasks();
    }
  }

  for (auto& splitGroupState : splitGroupStates) {
    splitGroupState.clear();
  }

  for (auto& promise : splitPromises) {
    promise.setValue();
  }

  for (auto& bridge : oldBridges) {
    bridge->cancel();
  }

    // 创建finished future
  std::lock_guard<std::mutex> l(mutex_);
  return makeFinishFutureLocked("Task::terminate");
}
// 等待driver全部退出线程的future
ContinueFuture Task::makeFinishFutureLocked(const char* FOLLY_NONNULL comment) {
  auto [promise, future] = makeVeloxContinuePromiseContract(comment);
    // 0个线程的时候通知promise
    // 线程数是代表当前的任务中有几个driver在线程上运行
    // 0则代表没有，比如可能进入阻塞态了
  if (numThreads_ == 0) {
    promise.setValue();\
    return std::move(future);
  }
  // 完成时通知
  threadFinishPromises_.push_back(std::move(promise));
  return std::move(future);
}

void Task::addOperatorStats(OperatorStats& stats) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(
      stats.pipelineId >= 0 &&
      stats.pipelineId < taskStats_.pipelineStats.size());
  VELOX_CHECK(
      stats.operatorId >= 0 &&
      stats.operatorId <
          taskStats_.pipelineStats[stats.pipelineId].operatorStats.size());
  taskStats_.pipelineStats[stats.pipelineId]
      .operatorStats[stats.operatorId]
      .add(stats);
}

TaskStats Task::taskStats() const {
  std::lock_guard<std::mutex> l(mutex_);

  // 'taskStats_' contains task stats plus stats for the completed drivers
  // (their operators).
  TaskStats taskStats = taskStats_;

  taskStats.numTotalDrivers = drivers_.size();

  // Add stats of the drivers (their operators) that are still running.
  for (const auto& driver : drivers_) {
    // Driver can be null.
    if (driver == nullptr) {
      ++taskStats.numCompletedDrivers;
      continue;
    }

    for (auto& op : driver->operators()) {
      auto statsCopy = op->stats(false);
      taskStats.pipelineStats[statsCopy.pipelineId]
          .operatorStats[statsCopy.operatorId]
          .add(statsCopy);
    }
    if (driver->isOnThread()) {
      ++taskStats.numRunningDrivers;
    } else if (driver->isTerminated()) {
      ++taskStats.numTerminatedDrivers;
    } else {
      ++taskStats.numBlockedDrivers[driver->blockingReason()];
    }
  }

  return taskStats;
}

uint64_t Task::timeSinceStartMs() const {
  std::lock_guard<std::mutex> l(mutex_);
  if (taskStats_.executionStartTimeMs == 0UL) {
    return 0UL;
  }
  return getCurrentTimeMs() - taskStats_.executionStartTimeMs;
}

uint64_t Task::timeSinceEndMs() const {
  std::lock_guard<std::mutex> l(mutex_);
  if (taskStats_.executionEndTimeMs == 0UL) {
    return 0UL;
  }
  return getCurrentTimeMs() - taskStats_.executionEndTimeMs;
}

void Task::onTaskCompletion() {
  listeners().withRLock([&](auto& listeners) {
    // 没有监听者
    if (listeners.empty()) {
      return;
    }

    TaskStats stats;
    TaskState state;
    std::exception_ptr exception;
    {
      std::lock_guard<std::mutex> l(mutex_);
      stats = taskStats_;
      state = state_;
      exception = exception_;
    }
    // 调用函数
    for (auto& listener : listeners) {
      listener->onTaskCompletion(uuid_, taskId_, state, exception, stats);
    }
  });
}
// 用来等待状态发生改变的
ContinueFuture Task::stateChangeFuture(uint64_t maxWaitMicros) {
  // 获取任务互斥锁
  std::lock_guard<std::mutex> l(mutex_);
  // If 'this' is running, the future is realized on timeout or when
  // this no longer is running.
  if (not isRunningLocked()) {
    // 直接返回一个默认构造的？
    return ContinueFuture();
  }
  auto [promise, future] = makeVeloxContinuePromiseContract(
      fmt::format("Task::stateChangeFuture {}", taskId_));
  stateChangePromises_.emplace_back(std::move(promise));
  if (maxWaitMicros) {
    return std::move(future).within(std::chrono::microseconds(maxWaitMicros));
  }
  return std::move(future);
}

std::string Task::toString() const {
  std::stringstream out;
  out << "{Task " << shortId(taskId_) << " (" << taskId_ << ")";

  if (exception_) {
    out << "Error: " << errorMessage() << std::endl;
  }

  if (planFragment_.planNode) {
    out << "Plan: " << planFragment_.planNode->toString() << std::endl;
  }

  out << " drivers:\n";
  for (auto& driver : drivers_) {
    if (driver) {
      out << driver->toString() << std::endl;
    }
  }

  return out.str();
}

std::shared_ptr<MergeSource> Task::addLocalMergeSource(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId,
    const RowTypePtr& rowType) {
  auto source = MergeSource::createLocalMergeSource();
  splitGroupStates_[splitGroupId].localMergeSources[planNodeId].push_back(
      source);
  return source;
}

const std::vector<std::shared_ptr<MergeSource>>& Task::getLocalMergeSources(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  return splitGroupStates_[splitGroupId].localMergeSources[planNodeId];
}

void Task::createMergeJoinSource(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  auto& splitGroupState = splitGroupStates_[splitGroupId];

  VELOX_CHECK(
      splitGroupState.mergeJoinSources.find(planNodeId) ==
          splitGroupState.mergeJoinSources.end(),
      "Merge join sources already exist: {}",
      planNodeId);

  splitGroupState.mergeJoinSources.insert(
      {planNodeId, std::make_shared<MergeJoinSource>()});
}

std::shared_ptr<MergeJoinSource> Task::getMergeJoinSource(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  auto& splitGroupState = splitGroupStates_[splitGroupId];

  auto it = splitGroupState.mergeJoinSources.find(planNodeId);
  VELOX_CHECK(
      it != splitGroupState.mergeJoinSources.end(),
      "Merge join source for specified plan node doesn't exist: {}",
      planNodeId);
  return it->second;
}

void Task::createLocalExchangeQueuesLocked(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId,
    int numPartitions) {
        // 拿出split分组状态
  auto& splitGroupState = splitGroupStates_[splitGroupId];
  VELOX_CHECK(
      splitGroupState.localExchanges.find(planNodeId) ==
          splitGroupState.localExchanges.end(),
      "Local exchange already exists: {}",
      planNodeId);

  // TODO(spershin): Should we have one memory manager for all local exchanges
  //  in all split groups?
  LocalExchangeState exchange;
  exchange.memoryManager = std::make_shared<LocalExchangeMemoryManager>(
      queryCtx_->config().maxLocalExchangeBufferSize());
    // 几个分区几个队列
  exchange.queues.reserve(numPartitions);
  for (auto i = 0; i < numPartitions; ++i) {
    exchange.queues.emplace_back(
        std::make_shared<LocalExchangeQueue>(exchange.memoryManager, i));
  }

  splitGroupState.localExchanges.insert({planNodeId, std::move(exchange)});
}

void Task::noMoreLocalExchangeProducers(uint32_t splitGroupId) {
    // 获取split分组状态
  auto& splitGroupState = splitGroupStates_[splitGroupId];

  for (auto& exchange : splitGroupState.localExchanges) {
    for (auto& queue : exchange.second.queues) {
        // 调用这些队列，不再有生产者
      queue->noMoreProducers();
    }
  }
}

std::shared_ptr<LocalExchangeQueue> Task::getLocalExchangeQueue(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId,
    int partition) {
  const auto& queues = getLocalExchangeQueues(splitGroupId, planNodeId);
  VELOX_CHECK_LT(
      partition,
      queues.size(),
      "Incorrect partition for local exchange {}",
      planNodeId);
  return queues[partition];
}

const std::vector<std::shared_ptr<LocalExchangeQueue>>&
Task::getLocalExchangeQueues(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  auto& splitGroupState = splitGroupStates_[splitGroupId];

  auto it = splitGroupState.localExchanges.find(planNodeId);
  VELOX_CHECK(
      it != splitGroupState.localExchanges.end(),
      "Incorrect local exchange ID: {}",
      planNodeId);
  return it->second.queues;
}

void Task::setError(const std::exception_ptr& exception) {
  bool isFirstError = false;
  {
    // 加锁
    std::lock_guard<std::mutex> l(mutex_);
    // 说明任务状态已经被设置过了
    if (not isRunningLocked()) {
      return;
    }
    if (!exception_) {
      exception_ = exception;
      isFirstError = true;
    }
  }
  // 如果是第一个异常的话，调用terminate
  if (isFirstError) {
    terminate(TaskState::kFailed);
  }
  // 调用回调函数
  if (isFirstError && onError_) {
    onError_(exception_);
  }
}

void Task::setError(const std::string& message) {
  // The only way to acquire an std::exception_ptr is via throw and
  // std::current_exception().
  try {
    throw std::runtime_error(message);
  } catch (const std::runtime_error& e) {
    setError(std::current_exception());
  }
}

std::string Task::errorMessage() const {
  std::lock_guard<std::mutex> l(mutex_);
  return errorMessageImpl(exception_);
}

StopReason Task::enter(ThreadState& state) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(state.isEnqueued);
  // 表明弹出队列？
  state.isEnqueued = false;
  // 其他调用了terminate，自然会影响到当前的driver
  if (state.isTerminated) {
    return StopReason::kAlreadyTerminated;
  }
  // 已经在线程上了，直接返回
  if (state.isOnThread()) {
    return StopReason::kAlreadyOnThread;
  }
  auto reason = shouldStopLocked();
  if (reason == StopReason::kTerminate) {
    // 设置状态为终结
    state.isTerminated = true;
  }
  if (reason == StopReason::kNone) {
    // 增加线程数
    ++numThreads_;
    // 设置线程状态里的线程为当前线程
    state.setThread();
    state.hasBlockingFuture = false;
  }
  return reason;
}

StopReason Task::enterForTerminateLocked(ThreadState& state) {
    // 如果driver在线程上的话，只是设置它的isTerminate
  if (state.isOnThread() || state.isTerminated) {
    state.isTerminated = true;
    return StopReason::kAlreadyOnThread;
  }
  // 由当前线程负责清理资源
  state.isTerminated = true;
  state.setThread();
  return StopReason::kTerminate;
}

StopReason Task::leave(ThreadState& state) {
  std::lock_guard<std::mutex> l(mutex_);
  // 减少线程数
  if (--numThreads_ == 0) {
    // 在没有driver在线程上运行的时候，通知promise
    finishedLocked();
  }
  // 重置线程状态
  state.clearThread();
  // 如果被终结了，就返回终结停止原因
  if (state.isTerminated) {
    // 这里为什么不返回AlreadyTerminate
    return StopReason::kTerminate;
  }
  auto reason = shouldStopLocked();
  if (reason == StopReason::kTerminate) {
    state.isTerminated = true;
  }
  return reason;
}

StopReason Task::enterSuspended(ThreadState& state) {
  VELOX_CHECK(!state.hasBlockingFuture);
  // 这里已经断言state在线程上了，下面加锁后又做了判断?
  VELOX_CHECK(state.isOnThread());
  std::lock_guard<std::mutex> l(mutex_);
  if (state.isTerminated) {
    // 返回已经终结
    return StopReason::kAlreadyTerminated;
  }
  // 感觉这个逻辑不会进入
  if (!state.isOnThread()) {
    // 不在线程上也返回已经终结？什么几把玩意
    return StopReason::kAlreadyTerminated;
  }

  // 检查是否已经stop
  auto reason = shouldStopLocked();
  // 如果是因为有终结请求
  if (reason == StopReason::kTerminate) {
    // 为什么这种情况下不返回终结
    state.isTerminated = true;
  }
  // A pause will not stop entering the suspended section. It will
  // just ack that the thread is no longer in inside the
  // CancelPool. The pause can wait at the exit of the suspended
  // section.
  if (reason == StopReason::kNone || reason == StopReason::kPause) {
    // 挂起
    state.isSuspended = true;
    // 减少线程数
    if (--numThreads_ == 0) {
        // 通知promise
      finishedLocked();
    }
  }
  // 这里总是返回None
  return StopReason::kNone;
}

StopReason Task::leaveSuspended(ThreadState& state) {
    // 无限循环
  for (;;) {
    {
      std::lock_guard<std::mutex> l(mutex_);
      ++numThreads_;
      // 设置线程状态为不挂起
      state.isSuspended = false;
      if (state.isTerminated) {
        // 如果终结了，就返回已经终结
        return StopReason::kAlreadyTerminated;
      }
      // 如果有终结请求
      if (terminateRequested_) {
        state.isTerminated = true;
        return StopReason::kTerminate;
      }
      // 如果没有暂停请求，就直接返回None
      if (!pauseRequested_) {
        // For yield or anything but pause  we return here.
        return StopReason::kNone;
      }
      // 如果有暂停请求，就重新挂起
      --numThreads_;
      state.isSuspended = true;
    }
    // If the pause flag is on when trying to reenter, sleep a while
    // outside of the mutex and recheck. This is rare and not time
    // critical. Can happen if memory interrupt sets pause while
    // already inside a suspended section for other reason, like
    // IO.
    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
  }
}

StopReason Task::shouldStop() {
    // 终结请求
  if (terminateRequested_) {
    return StopReason::kTerminate;
  }
  // 停止请求
  if (pauseRequested_) {
    return StopReason::kPause;
  }
  if (toYield_) {
    // 这里要加锁,是因为里面要修改toYield_的值的原因吧
    std::lock_guard<std::mutex> l(mutex_);
    return shouldStopLocked();
  }
  return StopReason::kNone;
}
// 所有driver都离开线程了，比如进入阻塞了，或者暂停之类的
void Task::finishedLocked() {
    // 遍历每一个promise，通知完成
  for (auto& promise : threadFinishPromises_) {
    promise.setValue();
  }
  threadFinishPromises_.clear();
}

StopReason Task::shouldStopLocked() {
    // 如果有终结请求的话
  if (terminateRequested_) {
    return StopReason::kTerminate;
  }
  // 如果有暂停请求的话
  if (pauseRequested_) {
    return StopReason::kPause;
  }
  if (toYield_) {
    // 减少1
    --toYield_;
    return StopReason::kYield;
  }
  // 否则返回None
  return StopReason::kNone;
}
// 在任务互斥锁的保护下
ContinueFuture Task::requestPauseLocked(bool pause) {
    // 设置暂停请求
  pauseRequested_ = pause;
  // 返回future
  return makeFinishFutureLocked("Task::requestPause");
}

Task::TaskCompletionNotifier::~TaskCompletionNotifier() {
  notify();
}

void Task::TaskCompletionNotifier::activate(
    std::function<void()> callback,
    std::vector<ContinuePromise> promises) {
  active_ = true;
  callback_ = callback;
  promises_ = std::move(promises);
}

void Task::TaskCompletionNotifier::notify() {
  if (active_) {
    for (auto& promise : promises_) {
      promise.setValue();
    }
    promises_.clear();

    callback_();

    active_ = false;
  }
}

namespace {
// Describes memory usage stats of a Memory Pool (optionally aggregated).
struct MemoryUsage {
  int64_t totalBytes{0};
  int64_t minBytes{std::numeric_limits<int64_t>::max()};
  int64_t maxBytes{0};
  size_t numEntries{0};

  void update(int64_t bytes) {
    maxBytes = std::max(maxBytes, bytes);
    minBytes = std::min(minBytes, bytes);
    totalBytes += bytes;
    ++numEntries;
  }

  void toString(std::stringstream& out, const char* entriesName = "entries")
      const {
    out << succinctBytes(totalBytes) << " in " << numEntries << " "
        << entriesName << ", min " << succinctBytes(minBytes) << ", max "
        << succinctBytes(maxBytes);
  }
};

// Aggregated memory usage stats of a single task plan node memory pool.
struct NodeMemoryUsage {
  MemoryUsage total;
  std::unordered_map<std::string, MemoryUsage> operators;
};

// Aggregated memory usage stats of a single Task Pipeline Memory Pool.
struct TaskMemoryUsage {
  std::string taskId;
  MemoryUsage total;
  // The map from node id to its collected memory usage.
  std::map<std::string, NodeMemoryUsage> nodes;

  void toString(std::stringstream& out) const {
    // Using 4 spaces for indent in the output.
    out << "\n    ";
    out << taskId;
    out << ": ";
    total.toString(out, "nodes");
    for (const auto& [nodeId, nodeMemoryUsage] : nodes) {
      out << "\n        ";
      out << nodeId;
      out << ": ";
      nodeMemoryUsage.total.toString(out, "operators");
      for (const auto& it : nodeMemoryUsage.operators) {
        out << "\n            ";
        out << it.first;
        out << ": ";
        it.second.toString(out, "instances");
      }
    }
  }
};

void collectOperatorMemoryUsage(
    NodeMemoryUsage& nodeMemoryUsage,
    memory::MemoryPool* operatorPool) {
  const auto numBytes =
      operatorPool->getMemoryUsageTracker()->getCurrentTotalBytes();
  nodeMemoryUsage.total.update(numBytes);
  auto& operatorMemoryUsage = nodeMemoryUsage.operators[operatorPool->name()];
  operatorMemoryUsage.update(numBytes);
}

void collectNodeMemoryUsage(
    TaskMemoryUsage& taskMemoryUsage,
    memory::MemoryPool* nodePool) {
  // Update task's stats from each node.
  taskMemoryUsage.total.update(
      nodePool->getMemoryUsageTracker()->getCurrentTotalBytes());

  // NOTE: we use a plan node id as the node memory pool's name.
  const auto& poolName = nodePool->name();
  auto& nodeMemoryUsage = taskMemoryUsage.nodes[poolName];

  // Run through the node's child operator pools and update the memory usage.
  nodePool->visitChildren([&nodeMemoryUsage](memory::MemoryPool* operatorPool) {
    collectOperatorMemoryUsage(nodeMemoryUsage, operatorPool);
  });
}

void collectTaskMemoryUsage(
    TaskMemoryUsage& taskMemoryUsage,
    memory::MemoryPool* taskPool) {
  taskMemoryUsage.taskId = taskPool->name();
  taskPool->visitChildren([&taskMemoryUsage](memory::MemoryPool* nodePool) {
    collectNodeMemoryUsage(taskMemoryUsage, nodePool);
  });
}

std::string getQueryMemoryUsageString(memory::MemoryPool* queryPool) {
  // Collect the memory usage numbers from query's tasks, nodes and operators.
  std::vector<TaskMemoryUsage> taskMemoryUsages;
  taskMemoryUsages.reserve(queryPool->getChildCount());
  queryPool->visitChildren([&taskMemoryUsages](memory::MemoryPool* taskPool) {
    taskMemoryUsages.emplace_back(TaskMemoryUsage{});
    collectTaskMemoryUsage(taskMemoryUsages.back(), taskPool);
  });

  // We will collect each operator's aggregated memory usage to later show the
  // largest memory consumers.
  struct TopMemoryUsage {
    int64_t totalBytes;
    std::string description;
  };
  std::vector<TopMemoryUsage> topOperatorMemoryUsages;

  // Build the query memory use tree (task->node->operator).
  std::stringstream out;
  out << "\n";
  out << queryPool->name();
  out << ": total: ";
  out << succinctBytes(
      queryPool->getMemoryUsageTracker()->getCurrentTotalBytes());
  for (const auto& taskMemoryUsage : taskMemoryUsages) {
    taskMemoryUsage.toString(out);
    // Collect each operator's memory usage into the vector.
    for (const auto& [nodeId, nodeMemUsage] : taskMemoryUsage.nodes) {
      for (const auto& it : nodeMemUsage.operators) {
        const MemoryUsage& operatorMemoryUsage = it.second;
        // Ignore operators with zero memory for top memory users.
        if (operatorMemoryUsage.totalBytes > 0) {
          topOperatorMemoryUsages.emplace_back(TopMemoryUsage{
              operatorMemoryUsage.totalBytes,
              fmt::format(
                  "{}.{}.{}", taskMemoryUsage.taskId, nodeId, it.first)});
        }
      }
    }
  }

  // Sort and show top memory users.
  out << "\nTop operator memory usages:";
  std::sort(
      topOperatorMemoryUsages.begin(),
      topOperatorMemoryUsages.end(),
      [](const TopMemoryUsage& left, const TopMemoryUsage& right) {
        return left.totalBytes > right.totalBytes;
      });
  for (const auto& top : topOperatorMemoryUsages) {
    out << "\n    " << top.description << ": " << succinctBytes(top.totalBytes);
  }

  return out.str();
}
} // namespace

std::shared_ptr<SpillOperatorGroup> Task::getSpillOperatorGroupLocked(
    uint32_t splitGroupId,
    const core::PlanNodeId& planNodeId) {
  auto& groups = splitGroupStates_[splitGroupId].spillOperatorGroups;
  auto it = groups.find(planNodeId);
  VELOX_CHECK(it != groups.end(), "Split group is not set {}", splitGroupId);
  auto group = it->second;
  VELOX_CHECK_NOT_NULL(
      group,
      "Spill group for plan node ID {} is not set in split group {}",
      planNodeId,
      splitGroupId);
  return group;
}

std::string Task::getErrorMsgOnMemCapExceeded(
    memory::MemoryUsageTracker& /*tracker*/) {
  return getQueryMemoryUsageString(queryCtx()->pool());
}

// static
void Task::testingWaitForAllTasksToBeDeleted(uint64_t maxWaitUs) {
  const uint64_t numCreatedTasks = Task::numCreatedTasks();
  uint64_t numDeletedTasks = Task::numDeletedTasks();
  uint64_t waitUs = 0;
  while (numCreatedTasks > numDeletedTasks) {
    constexpr uint64_t kWaitInternalUs = 1'000;
    std::this_thread::sleep_for(std::chrono::microseconds(kWaitInternalUs));
    waitUs += kWaitInternalUs;
    numDeletedTasks = Task::numDeletedTasks();
    if (waitUs >= maxWaitUs) {
      break;
    }
  }
  VELOX_CHECK_EQ(
      numDeletedTasks,
      numCreatedTasks,
      "{} tasks have been created while only {} have been deleted after waiting for {} us",
      numCreatedTasks,
      numDeletedTasks,
      waitUs);
}

} // namespace facebook::velox::exec
