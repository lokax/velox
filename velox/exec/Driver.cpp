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

#include <folly/ScopeGuard.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/executors/task_queue/UnboundedBlockingQueue.h>
#include <folly/executors/thread_factory/InitThreadFactory.h>
#include <gflags/gflags.h>
#include "velox/common/time/Timer.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

DriverCtx::DriverCtx(
    std::shared_ptr<Task> _task,
    int _driverId,
    int _pipelineId,
    uint32_t _splitGroupId,
    uint32_t _partitionId)
    : driverId(_driverId),
      pipelineId(_pipelineId),
      splitGroupId(_splitGroupId),
      partitionId(_partitionId),
      task(_task),
      pool(task->addDriverPool(pipelineId, driverId)) {}

const core::QueryConfig& DriverCtx::queryConfig() const {
  return task->queryCtx()->config();
}

velox::memory::MemoryPool* FOLLY_NONNULL
DriverCtx::addOperatorPool(const std::string& operatorType) {
  return task->addOperatorPool(pool, operatorType);
}

std::atomic_uint64_t BlockingState::numBlockedDrivers_{0};

BlockingState::BlockingState(
    std::shared_ptr<Driver> driver,
    ContinueFuture&& future,
    Operator* op,
    BlockingReason reason)
    : driver_(std::move(driver)),
      future_(std::move(future)),
      operator_(op),
      reason_(reason),
      sinceMicros_(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count()) {
  // Set before leaving the thread.
  driver_->state().hasBlockingFuture = true;
  numBlockedDrivers_++;
}

// static
void BlockingState::setResume(std::shared_ptr<BlockingState> state) {
  VELOX_CHECK(!state->driver_->isOnThread());
  // 拿出执行器
  auto& exec = folly::QueuedImmediateExecutor::instance();
  // 把future交给执行器去处理
  std::move(state->future_)
      .via(&exec)
      .thenValue([state](auto&& /* unused */) {
        auto driver = state->driver_;
        auto task = driver->task();
        // 获取任务的互斥锁
        std::lock_guard<std::mutex> l(task->mutex());
        if (!driver->state().isTerminated) {
          state->operator_->recordBlockingTime(state->sinceMicros_);
        }
        // 不能是挂起的状态
        VELOX_CHECK(!driver->state().isSuspended);
        // 这个时候有阻塞的future
        // 创建BlockState的时候，在构造函数里面就会设置这个flag
        VELOX_CHECK(driver->state().hasBlockingFuture);
        driver->state().hasBlockingFuture = false;
        // 哪怕有终结请求，或者任务已经被终止了，在有暂停请求时这里也会等待唤醒？
        // 如果该任务被暂停的话，不由我们来放进队列，而是由resume函数来执行
        if (task->pauseRequested()) {
          // The thread will be enqueued at resume.
          return;
        }
        // 重新放回队列中，等待被执行
        Driver::enqueue(state->driver_);
      })
      .thenError(
          folly::tag_t<std::exception>{}, [state](std::exception const& e) {
            LOG(ERROR)
                << "A ContinueFuture should not be realized with an error"
                << e.what();
            state->driver_->setError(std::make_exception_ptr(e));
          });
}

namespace {

// Ensures that the thread is removed from its Task's thread count on exit.
class CancelGuard {
 public:
  CancelGuard(
      Task* task,
      ThreadState* state,
      std::function<void(StopReason)> onTerminate)
      : task_(task), state_(state), onTerminate_(onTerminate) {}

  void notThrown() {
    isThrow_ = false;
  }

  ~CancelGuard() {
    bool onTerminateCalled = false;
    if (isThrow_) {
      // Runtime error. Driver is on thread, hence safe.
      state_->isTerminated = true;
      onTerminate_(StopReason::kNone);
      onTerminateCalled = true;
    }
    auto reason = task_->leave(*state_);
    if (reason == StopReason::kTerminate) {
      // Terminate requested via Task. The Driver is not on
      // thread but 'terminated_' is set, hence no other threads will
      // enter. onTerminateCalled will be true if both runtime error and
      // terminate requested via Task.
      if (!onTerminateCalled) {
        onTerminate_(reason);
      }
    }
  }

 private:
  Task* task_;
  ThreadState* state_;
  std::function<void(StopReason reason)> onTerminate_;
  bool isThrow_ = true;
};
} // namespace

// static
void Driver::enqueue(std::shared_ptr<Driver> driver) {
  // This is expected to be called inside the Driver's Tasks's mutex.
  driver->enqueueInternal();
  if (driver->closed_) {
    return;
  }
  driver->task()->queryCtx()->executor()->add(
      [driver]() { Driver::run(driver); });
}

Driver::Driver(
    std::unique_ptr<DriverCtx> ctx,
    std::vector<std::unique_ptr<Operator>> operators)
    : ctx_(std::move(ctx)), operators_(std::move(operators)) {
  curOpIndex_ = operators_.size() - 1;
  // Operators need access to their Driver for adaptation.
  ctx_->driver = this;
  trackOperatorCpuUsage_ = ctx_->queryConfig().operatorTrackCpuUsage();
}

namespace {
/// Checks if output channel is produced using identity projection and returns
/// input channel if so.
std::optional<column_index_t> getIdentityProjection(
    const std::vector<IdentityProjection>& projections,
    column_index_t outputChannel) {
  for (const auto& projection : projections) {
    if (projection.outputChannel == outputChannel) {
      return projection.inputChannel;
    }
  }
  return std::nullopt;
}
} // namespace

void Driver::pushdownFilters(int operatorIndex) {
    // 算子
  auto op = operators_[operatorIndex].get();
  // 获取动态过滤器
  const auto& filters = op->getDynamicFilters();
  // 没有动态过滤器，则直接返回
  if (filters.empty()) {
    return;
  }
    // 统计数据？
  op->stats().addRuntimeStat(
      "dynamicFiltersProduced", RuntimeCounter(filters.size()));

  // Walk operator list upstream and find a place to install the filters.
  for (const auto& entry : filters) {
    // column index?
    auto channel = entry.first;
    // 遍历当前算子等待每一个之前的算子
    for (auto i = operatorIndex - 1; i >= 0; --i) {
        // 之前的每一个算子
      auto prevOp = operators_[i].get();

      if (i == 0) {
        // Source operator.
        VELOX_CHECK(
            prevOp->canAddDynamicFilter(),
            "Cannot push down dynamic filters produced by {}",
            op->toString());
        prevOp->addDynamicFilter(channel, entry.second);
        prevOp->stats().addRuntimeStat(
            "dynamicFiltersAccepted", RuntimeCounter(1));
        break;
      }

      const auto& identityProjections = prevOp->identityProjections();
      auto inputChannel = getIdentityProjection(identityProjections, channel);
      if (!inputChannel.has_value()) {
        // Filter channel is not an identity projection.
        VELOX_CHECK(
            prevOp->canAddDynamicFilter(),
            "Cannot push down dynamic filters produced by {}",
            op->toString());
        prevOp->addDynamicFilter(channel, entry.second);
        prevOp->stats().addRuntimeStat(
            "dynamicFiltersAccepted", RuntimeCounter(1));
        break;
      }

      // Continue walking upstream.
      channel = inputChannel.value();
    }
  }
    // 清除
  op->clearDynamicFilters();
}

RowVectorPtr Driver::next(std::shared_ptr<BlockingState>& blockingState) {
  enqueueInternal();

  auto self = shared_from_this();
  RowVectorPtr result;
  auto stop = runInternal(self, blockingState, result);

  // We get kBlock if 'result' was produced; kAtEnd if pipeline has finished
  // processing and no more results will be produced; kAlreadyTerminated on
  // error.
  VELOX_CHECK(
      stop == StopReason::kBlock || stop == StopReason::kAtEnd ||
      stop == StopReason::kAlreadyTerminated);

  return result;
}

void Driver::enqueueInternal() {
  VELOX_CHECK(!state_.isEnqueued);
  // 设置为进入队列
  state_.isEnqueued = true;
  // When enqueuing, starting timing the queue time.
  queueTimeStartMicros_ = getCurrentTimeMicro();
}

StopReason Driver::runInternal(
    std::shared_ptr<Driver>& self,
    std::shared_ptr<BlockingState>& blockingState,
    RowVectorPtr& result) {
  auto queuedTime = (getCurrentTimeMicro() - queueTimeStartMicros_) * 1'000;
  // Update the next operator's queueTime.
  // 如果driver已经关闭了，这里就返回终结状态
  auto stop = closed_ ? StopReason::kTerminate : task()->enter(state_);
  if (stop != StopReason::kNone) {
    if (stop == StopReason::kTerminate) {
      // ctx_ still has a reference to the Task. 'this' is not on
      // thread from the Task's viewpoint, hence no need to call
      // close().
      ctx_->task->setError(std::make_exception_ptr(VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "Cancelled",
          error_source::kErrorSourceRuntime,
          error_code::kInvalidState,
          false)));
    }
    // 返回停止状态
    return stop;
  }

  // Update the queued time after entering the Task to ensure the stats have not
  // been deleted.
  if (curOpIndex_ < operators_.size()) {
    // 当前CurOpIndex算子是阻塞的？
    operators_[curOpIndex_]->stats().addRuntimeStat(
        "queuedWallNanos",
        RuntimeCounter(queuedTime, RuntimeCounter::Unit::kNanos));
  }

  CancelGuard guard(task().get(), &state_, [&](StopReason reason) {
    // This is run on error or cancel exit.
    if (reason == StopReason::kTerminate) {
      task()->setError(std::make_exception_ptr(VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "Cancelled",
          error_source::kErrorSourceRuntime,
          error_code::kInvalidState,
          false)));
    }
    close();
  });

  // Ensure we remove the writer we might have set when we exit this function in
  // any way.
  const auto statWriterGuard =
      folly::makeGuard([]() { setRunTimeStatWriter(nullptr); });

  try {
    // 算子的数量
    int32_t numOperators = operators_.size();
    ContinueFuture future;

    for (;;) {
        // 逆序遍历,从sink算子开始
      for (int32_t i = numOperators - 1; i >= 0; --i) {
        // 任务是否应该停止
        // 这个shouldStop一开始是没有加锁的
        // 可能是因为每次循环都会调用，所以这里干脆不加锁了?但是是通过原子变量去访问的
        stop = task()->shouldStop();
        if (stop != StopReason::kNone) {
          guard.notThrown();
          return stop;
        }
        // 当前算子
        auto op = operators_[i].get();
        // In case we are blocked, this index will point to the operator, whose
        // queuedTime we should update.
        curOpIndex_ = i;
        // Set up the runtime stats writer with the current operator, whose
        // runtime stats would be updated (for instance time taken to load lazy
        // vectors).
        setRunTimeStatWriter(std::make_unique<OperatorRuntimeStatWriter>(op));
        // 检查当前算子阻塞原因
        blockingReason_ = op->isBlocked(&future);
        // 如果算子发生了阻塞
        if (blockingReason_ != BlockingReason::kNotBlocked) {
            // 创建blockingState
          blockingState = std::make_shared<BlockingState>(
              self, std::move(future), op, blockingReason_);
          guard.notThrown();
          return StopReason::kBlock;
        }
        // 下一个算子
        Operator* nextOp = nullptr;
        // i不是最后一个算子
        if (i < operators_.size() - 1) {
            // 获取下一个算子
          nextOp = operators_[i + 1].get();
          // 检查下一个算子的阻塞原因
          blockingReason_ = nextOp->isBlocked(&future);
          if (blockingReason_ != BlockingReason::kNotBlocked) {
            blockingState = std::make_shared<BlockingState>(
                self, std::move(future), nextOp, blockingReason_);
            guard.notThrown();
            // 下一个算子阻塞的话，这里就也不处理了？
            return StopReason::kBlock;
          }
          // 如果下一个算子需要输入的话
          if (nextOp->needsInput()) {
            uint64_t resultBytes = 0;
            RowVectorPtr result;
            {
              auto timer = cpuWallTimer(op->stats().getOutputTiming);
              result = op->getOutput();
              if (result) {
                VELOX_CHECK(
                    result->size() > 0,
                    "Operator::getOutput() must return nullptr or a non-empty vector: {}",
                    op->stats().operatorType);
                op->stats().outputVectors += 1;
                op->stats().outputPositions += result->size();
                resultBytes = result->estimateFlatSize();
                op->stats().outputBytes += resultBytes;
              }
            }
            pushdownFilters(i);
            // 如果当前算子有输出
            if (result) {
              auto timer = cpuWallTimer(op->stats().addInputTiming);
              nextOp->stats().inputVectors += 1;
              nextOp->stats().inputPositions += result->size();
              nextOp->stats().inputBytes += resultBytes;
              // 把输出结果添加到下一个算子上的输入上
              nextOp->addInput(result);
              // The next iteration will see if operators_[i + 1] has
              // output now that it got input.
              // 继续处理下一个算子
              i += 2;
              continue;
            } else {
                // 如果当前算子没有输出数据
              stop = task()->shouldStop();
              if (stop != StopReason::kNone) {
                guard.notThrown();
                return stop;
              }
              // The op is at end. If this is finishing, propagate the
              // finish to the next op. The op could have run out
              // because it is blocked. If the op is the source and it
              // is not blocked and empty, this is finished. If this is
              // not the source, just try to get output from the one
              // before.
              // 之前不是不阻塞吗，为什么这里又检查是否发生了阻塞
              blockingReason_ = op->isBlocked(&future);
              if (blockingReason_ != BlockingReason::kNotBlocked) {
                blockingState = std::make_shared<BlockingState>(
                    self, std::move(future), op, blockingReason_);
                guard.notThrown();
                return StopReason::kBlock;
              }
              if (op->isFinished()) {
                auto timer = cpuWallTimer(op->stats().finishTiming);
                nextOp->noMoreInput();
                break;
              }
            }
          }
        } else {
          // A sink (last) operator, after getting unblocked, gets
          // control here, so it can advance. If it is again blocked,
          // this will be detected when trying to add input, and we
          // will come back here after this is again on thread.
          {
            auto timer = cpuWallTimer(op->stats().getOutputTiming);
            result = op->getOutput();
            if (result) {
              VELOX_CHECK(
                  result->size() > 0,
                  "Operator::getOutput() must return nullptr or a non-empty vector: {}",
                  op->stats().operatorType);

              // This code path is used only in single-threaded execution.
              blockingReason_ = BlockingReason::kWaitForConsumer;
              guard.notThrown();
              // 返回阻塞态，等待消费者进行消费
              return StopReason::kBlock;
            }
          }
          // 如果算子完成了，返回结束状态
          if (op->isFinished()) {
            guard.notThrown();
            close();
            return StopReason::kAtEnd;
          }
          // 这是干什么
          pushdownFilters(i);
          continue;
        }
      }
    }
  } catch (velox::VeloxException& e) {
    task()->setError(std::current_exception());
    // The CancelPoolGuard will close 'self' and remove from Task.
    return StopReason::kAlreadyTerminated;
  } catch (std::exception& e) {
    task()->setError(std::current_exception());
    // The CancelGuard will close 'self' and remove from Task.
    return StopReason::kAlreadyTerminated;
  }
}

// static
void Driver::run(std::shared_ptr<Driver> self) {
  std::shared_ptr<BlockingState> blockingState;
  RowVectorPtr nullResult;
  auto reason = self->runInternal(self, blockingState, nullResult);

    // 产生的数据必须被回调或者其他算子消费掉，最后一个算子不能产生输出数据
  // When Driver runs on an executor, the last operator (sink) must not produce
  // any results.
  VELOX_CHECK_NULL(
      nullResult,
      "The last operator (sink) must not produce any results. "
      "Results need to be consumed by either a callback or another operator. ")

  switch (reason) {
    case StopReason::kBlock:
      // Set the resume action outside the Task so that, if the
      // future is already realized we do not have a second thread
      // entering the same Driver.
      BlockingState::setResume(blockingState);
      return;

    case StopReason::kYield:
    // 主动让出CPU，那么
        // 将任务放到队列的末尾
      // Go to the end of the queue.
      enqueue(self);
      return;
    // 暂停的话，什么不处理?
    // 暂停的话，就由别人调用resume后从而重新放进队列中进行计算
    case StopReason::kPause:
    case StopReason::kTerminate:
    case StopReason::kAlreadyTerminated:
    case StopReason::kAtEnd:
      return;
    default:
      VELOX_CHECK(false, "Unhandled stop reason");
  }
}

void Driver::initializeOperatorStats(std::vector<OperatorStats>& stats) {
  stats.resize(operators_.size(), OperatorStats(0, 0, "", ""));
  // initialize the place in stats given by the operatorId. Use the
  // operatorId instead of i as the index to document the usage. The
  // operators are sequentially numbered but they could be reordered
  // in the pipeline later, so the ordinal position of the Operator is
  // not always the index into the stats.
  for (auto& op : operators_) {
    auto id = op->stats().operatorId;
    assert(id < stats.size());
    stats[id] = op->stats();
  }
}

void Driver::addStatsToTask() {
  for (auto& op : operators_) {
    auto& stats = op->stats();
    stats.memoryStats.update(op->pool()->getMemoryUsageTracker());
    stats.numDrivers = 1;
    task()->addOperatorStats(stats);
  }
}

void Driver::close() {
  if (closed_) {
    // Already closed.
    return;
  }
  if (!isOnThread() && !isTerminated()) {
    LOG(FATAL) << "Driver::close is only allowed from the Driver's thread";
  }
  addStatsToTask();
  for (auto& op : operators_) {
    // 关闭每一个算子
    op->close();
  }
  // 设置closed为true
  closed_ = true;
  Task::removeDriver(ctx_->task, this);
}

void Driver::closeByTask() {
  VELOX_CHECK(isTerminated());
  addStatsToTask();
  // 关闭每一个算子，然后设置closed为true
  for (auto& op : operators_) {
    op->close();
  }
  closed_ = true;
}

bool Driver::mayPushdownAggregation(Operator* aggregation) const {
  for (auto i = 1; i < operators_.size(); ++i) {
    auto op = operators_[i].get();
    if (aggregation == op) {
      return true;
    }
    if (!op->isFilter() || !op->preservesOrder()) {
      return false;
    }
  }
  VELOX_FAIL(
      "Aggregation operator not found in its Driver: {}",
      aggregation->toString());
}

std::unordered_set<column_index_t> Driver::canPushdownFilters(
    const Operator* FOLLY_NONNULL filterSource,
    const std::vector<column_index_t>& channels) const {
  int filterSourceIndex = -1;
  // 找到filterSource所在的index
  for (auto i = 0; i < operators_.size(); ++i) {
    auto op = operators_[i].get();
    if (filterSource == op) {
      filterSourceIndex = i;
      break;
    }
  }
  VELOX_CHECK_GE(
      filterSourceIndex,
      0,
      "Operator not found in its Driver: {}",
      filterSource->toString());

  std::unordered_set<column_index_t> supportedChannels;
  for (auto i = 0; i < channels.size(); ++i) {
    auto channel = channels[i];
    for (auto j = filterSourceIndex - 1; j >= 0; --j) {
        // 遍历之前的每一个算子
      auto prevOp = operators_[j].get();

      if (j == 0) {
        // Source operator.
        if (prevOp->canAddDynamicFilter()) {
            // 如果source能支持，则添加channel进去，这个channel是op的input chanal，
            // 应该也是prevop的output chanal
          supportedChannels.emplace(channels[i]);
        }
        break;
      }
        // 拿出相同的projecion?
      const auto& identityProjections = prevOp->identityProjections();
      auto inputChannel = getIdentityProjection(identityProjections, channel);
      if (!inputChannel.has_value()) {
        // 不能再下推的意思吗？
        // Filter channel is not an identity projection.
        if (prevOp->canAddDynamicFilter()) {
          supportedChannels.emplace(channels[i]);
        }
        break;
      }

      // Continue walking upstream.
      channel = inputChannel.value();
    }
  }

  return supportedChannels;
}

Operator* FOLLY_NULLABLE
Driver::findOperator(std::string_view planNodeId) const {
  for (auto& op : operators_) {
    if (op->planNodeId() == planNodeId) {
      return op.get();
    }
  }
  return nullptr;
}

std::vector<Operator*> Driver::operators() const {
  std::vector<Operator*> operators;
  operators.reserve(operators_.size());
  for (auto& op : operators_) {
    operators.push_back(op.get());
  }
  return operators;
}

void Driver::setError(std::exception_ptr exception) {
  task()->setError(exception);
}

std::string Driver::toString() {
  std::stringstream out;
  out << "{Driver: ";
  if (state_.isOnThread()) {
    out << "running ";
  } else {
    out << "blocked " << static_cast<int>(blockingReason_) << " ";
  }
  for (auto& op : operators_) {
    out << op->toString() << " ";
  }
  out << "}";
  return out.str();
}
SuspendedSection::SuspendedSection(Driver* FOLLY_NONNULL driver)
    : driver_(driver) {
  if (driver->task()->enterSuspended(driver->state()) != StopReason::kNone) {
    VELOX_FAIL("Terminate detected when entering suspended section");
  }
}

SuspendedSection::~SuspendedSection() {
  if (driver_->task()->leaveSuspended(driver_->state()) != StopReason::kNone) {
    VELOX_FAIL("Terminate detected when leaving suspended section");
  }
}

std::string Driver::label() const {
  return fmt::format("<Driver {}:{}>", task()->taskId(), ctx_->driverId);
}

std::string blockingReasonToString(BlockingReason reason) {
  switch (reason) {
    case BlockingReason::kNotBlocked:
      return "kNotBlocked";
    case BlockingReason::kWaitForConsumer:
      return "kWaitForConsumer";
    case BlockingReason::kWaitForSplit:
      return "kWaitForSplit";
    case BlockingReason::kWaitForExchange:
      return "kWaitForExchange";
    case BlockingReason::kWaitForJoinBuild:
      return "kWaitForJoinBuild";
    case BlockingReason::kWaitForJoinProbe:
      return "kWaitForJoinProbe";
    case BlockingReason::kWaitForMemory:
      return "kWaitForMemory";
    case BlockingReason::kWaitForConnector:
      return "kWaitForConnector";
    case BlockingReason::kWaitForSpill:
      return "kWaitForSpill";
  }
  VELOX_UNREACHABLE();
  return "";
};

} // namespace facebook::velox::exec
