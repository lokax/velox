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

#include "velox/exec/HashBuild.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/expression/FieldReference.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {
namespace {
// Map HashBuild 'state' to the corresponding driver blocking reason.
BlockingReason fromStateToBlockingReason(HashBuild::State state) {
  switch (state) {
    case HashBuild::State::kRunning:
      FOLLY_FALLTHROUGH;
    case HashBuild::State::kFinish:
      return BlockingReason::kNotBlocked;
    case HashBuild::State::kWaitForSpill:
      return BlockingReason::kWaitForSpill;
    case HashBuild::State::kWaitForBuild:
      return BlockingReason::kWaitForJoinBuild;
    case HashBuild::State::kWaitForProbe:
      return BlockingReason::kWaitForJoinProbe;
    default:
      VELOX_UNREACHABLE(HashBuild::stateName(state));
  }
}
} // namespace

HashBuild::HashBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : Operator(driverCtx, nullptr, operatorId, joinNode->id(), "HashBuild"),
      joinNode_(std::move(joinNode)),
      joinType_{joinNode_->joinType()},
      allocator_(operatorCtx_->allocator()),
      joinBridge_(operatorCtx_->task()->getHashJoinBridgeLocked(
          operatorCtx_->driverCtx()->splitGroupId,
          planNodeId())),
      spillConfig_(
          isSpillAllowed()
              ? operatorCtx_->makeSpillConfig(Spiller::Type::kHashJoinBuild)
              : std::nullopt),
      spillGroup_(
          spillEnabled() ? operatorCtx_->task()->getSpillOperatorGroupLocked(
                               operatorCtx_->driverCtx()->splitGroupId,
                               planNodeId())
                         : nullptr) {
  VELOX_CHECK_NOT_NULL(joinBridge_);
  joinBridge_->addBuilder();

  auto outputType = joinNode_->sources()[1]->outputType();

  auto numKeys = joinNode_->rightKeys().size();
  keyChannels_.reserve(numKeys);
  folly::F14FastMap<column_index_t, column_index_t> keyChannelMap(numKeys);
  std::vector<std::string> names;
  names.reserve(outputType->size());
  std::vector<TypePtr> types;
  types.reserve(outputType->size());

    // 遍历右表的每一个key
  for (int i = 0; i < joinNode_->rightKeys().size(); ++i) {
    // 拿出key，获取对应的channel
    auto& key = joinNode_->rightKeys()[i];
    auto channel = exprToChannel(key.get(), outputType);
    keyChannelMap[channel] = i;
    keyChannels_.emplace_back(channel);
    names.emplace_back(outputType->nameOf(channel));
    types.emplace_back(outputType->childAt(channel));
  }

  // Identify the non-key build side columns and make a decoder for each.
  const auto numDependents = outputType->size() - numKeys;
  dependentChannels_.reserve(numDependents);
  decoders_.reserve(numDependents);
  for (auto i = 0; i < outputType->size(); ++i) {
    if (keyChannelMap.find(i) == keyChannelMap.end()) {
        // 创建对应的解码向量
      dependentChannels_.emplace_back(i);
      decoders_.emplace_back(std::make_unique<DecodedVector>());
      names.emplace_back(outputType->nameOf(i));
      types.emplace_back(outputType->childAt(i));
    }
  }
    // 创建行类型
  tableType_ = ROW(std::move(names), std::move(types));
  setupTable();
  setupSpiller();

  if (isAntiJoins(joinType_) && joinNode_->filter()) {
    setupFilterForAntiJoins(keyChannelMap);
  }
}

void HashBuild::setupTable() {
  VELOX_CHECK_NULL(table_);
    // key的数量
  const auto numKeys = keyChannels_.size();
  std::vector<std::unique_ptr<VectorHasher>> keyHashers;
  keyHashers.reserve(numKeys);
  // 创建hasher用来计算哈希？
  for (vector_size_t i = 0; i < numKeys; ++i) {
    keyHashers.emplace_back(std::make_unique<VectorHasher>(
        tableType_->childAt(i), keyChannels_[i]));
  }

  const auto numDependents = tableType_->size() - numKeys;
  std::vector<TypePtr> dependentTypes;
  dependentTypes.reserve(numDependents);
  for (int i = numKeys; i < tableType_->size(); ++i) {
    dependentTypes.emplace_back(tableType_->childAt(i));
  }
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin() ||
      joinNode_->isRightSemiProjectJoin()) {
    // Do not ignore null keys.
    table_ = HashTable<false>::createForJoin(
        std::move(keyHashers),
        dependentTypes,
        true, // allowDuplicates
        true, // hasProbedFlag
        allocator_);
  } else {
    // (Left) semi and anti join with no extra filter only needs to know whether
    // there is a match. Hence, no need to store entries with duplicate keys.
    const bool dropDuplicates = !joinNode_->filter() &&
        (joinNode_->isLeftSemiFilterJoin() ||
         joinNode_->isLeftSemiProjectJoin() || isAntiJoins(joinType_));
    // Right semi join needs to tag build rows that were probed.
    const bool needProbedFlag = joinNode_->isRightSemiFilterJoin();
    if (isNullAwareAntiJoinWithFilter(joinNode_)) {
      // We need to check null key rows in build side in case of null-aware anti
      // join with filter set.
      table_ = HashTable<false>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          !dropDuplicates, // allowDuplicates
          needProbedFlag, // hasProbedFlag
          allocator_);
    } else {
      // Ignore null keys
      table_ = HashTable<true>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          !dropDuplicates, // allowDuplicates
          needProbedFlag, // hasProbedFlag
          allocator_);
    }
  }
  analyzeKeys_ = table_->hashMode() != BaseHashTable::HashMode::kHash;
}

void HashBuild::setupSpiller(SpillPartition* spillPartition) {
  VELOX_CHECK_NULL(spiller_);
  VELOX_CHECK_NULL(spillInputReader_);
    // 如果没有启动spiller，则直接返回
  if (!spillEnabled()) {
    return;
  }
  // 获取spill配置
  const auto& spillConfig = spillConfig_.value();
  HashBitRange hashBits = spillConfig.hashBitRange;

  if (spillPartition == nullptr) {
    // 添加算子
    spillGroup_->addOperator(
        *this,
        [&](const std::vector<Operator*>& operators) { runSpill(operators); });
  } else {
    // 创建reader
    spillInputReader_ = spillPartition->createReader();
    // 这里先增加是为了在下一轮继续Spill？
    const auto startBit = spillPartition->id().partitionBitOffset() +
        spillConfig.hashBitRange.numBits();
    // Disable spilling if exceeding the max spill level and the query might run
    // out of memory if the restored partition still can't fit in memory.
    if (spillConfig.exceedSpillLevelLimit(startBit)) {
        // 不再进行Spill
      return;
    }
    hashBits =
        HashBitRange(startBit, startBit + spillConfig.hashBitRange.numBits());
  }

    // 创建新的Spiller
  spiller_ = std::make_unique<Spiller>(
      Spiller::Type::kHashJoinBuild,
      table_->rows(),
      [&](folly::Range<char**> rows) { table_->rows()->eraseRows(rows); },
      tableType_,
      std::move(hashBits),
      keyChannels_.size(),
      std::vector<CompareFlags>(),
      spillConfig.filePath,
      spillConfig.maxFileSize,
      spillConfig.minSpillRunSize,
      Spiller::spillPool(),
      spillConfig.executor);

    // 分区数量
  const int32_t numPartitions = spiller_->hashBits().numPartitions();
  spillInputIndicesBuffers_.resize(numPartitions);
  rawSpillInputIndicesBuffers_.resize(numPartitions);
  numSpillInputs_.resize(numPartitions, 0);
  spillChildVectors_.resize(tableType_->size());
}

bool HashBuild::isInputFromSpill() const {
  return spillInputReader_ != nullptr;
}

// 看注释
RowTypePtr HashBuild::inputType() const {
  return isInputFromSpill() ? tableType_
                            : joinNode_->sources()[1]->outputType();
}

void HashBuild::setupFilterForAntiJoins(
    const folly::F14FastMap<column_index_t, column_index_t>& keyChannelMap) {
  VELOX_DCHECK(
      std::is_sorted(dependentChannels_.begin(), dependentChannels_.end()));

  ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
  VELOX_DCHECK_EQ(exprs.exprs().size(), 1);
  const auto& expr = exprs.expr(0);
  filterPropagatesNulls_ = expr->propagatesNulls();
  if (filterPropagatesNulls_) {
    const auto outputType = joinNode_->sources()[1]->outputType();
    for (const auto& field : expr->distinctFields()) {
      const auto index = outputType->getChildIdxIfExists(field->field());
      if (!index.has_value()) {
        continue;
      }
      auto keyIter = keyChannelMap.find(*index);
      if (keyIter != keyChannelMap.end()) {
        keyFilterChannels_.push_back(keyIter->second);
      } else {
        auto dependentIter = std::lower_bound(
            dependentChannels_.begin(), dependentChannels_.end(), *index);
        VELOX_DCHECK(
            dependentIter != dependentChannels_.end() &&
            *dependentIter == *index);
        dependentFilterChannels_.push_back(
            dependentIter - dependentChannels_.begin());
      }
    }
  }
}

void HashBuild::removeInputRowsForAntiJoinFilter() {
  bool changed = false;
  auto* rawActiveRows = activeRows_.asMutableRange().bits();
  auto removeNulls = [&](DecodedVector& decoded) {
    if (decoded.mayHaveNulls()) {
      changed = true;
      // NOTE: the true value of a raw null bit indicates non-null so we AND
      // 'rawActiveRows' with the raw bit.
      bits::andBits(rawActiveRows, decoded.nulls(), 0, activeRows_.end());
    }
  };
  for (auto channel : keyFilterChannels_) {
    removeNulls(table_->hashers()[channel]->decodedVector());
  }
  for (auto channel : dependentFilterChannels_) {
    removeNulls(*decoders_[channel]);
  }
  if (changed) {
    activeRows_.updateBounds();
  }
}

void HashBuild::addInput(RowVectorPtr input) {
  checkRunning();

  if (!ensureInputFits(input)) {
    VELOX_CHECK_NOT_NULL(input_);
    VELOX_CHECK(future_.valid());
    // 直接返回
    return;
  }

  activeRows_.resize(input->size());
  activeRows_.setAll();

  auto& hashers = table_->hashers();

    // 计算哈希
  for (auto i = 0; i < hashers.size(); ++i) {
    auto key = input->childAt(hashers[i]->channel())->loadedVector();
    hashers[i]->decode(*key, activeRows_);
  }

  if (!isRightJoin(joinType_) && !isFullJoin(joinType_) &&
      !isRightSemiProjectJoin(joinType_) &&
      !isNullAwareAntiJoinWithFilter(joinNode_)) {
        // de选择掉NULL值
    deselectRowsWithNulls(hashers, activeRows_);
  }
    // 解码数据
  for (auto i = 0; i < dependentChannels_.size(); ++i) {
    decoders_[i]->decode(
        *input->childAt(dependentChannels_[i])->loadedVector(), activeRows_);
  }

  if (isAntiJoins(joinType_) && joinNode_->filter()) {
    if (filterPropagatesNulls_) {
      removeInputRowsForAntiJoinFilter();
    }
  } else if (
      (isNullAwareAntiJoin(joinType_) || isLeftSemiProjectJoin(joinType_)) &&
      activeRows_.countSelected() < input->size()) {
    joinHasNullKeys_ = true;
    if (isNullAwareAntiJoin(joinType_)) {
      // Null-aware anti join with no extra filter returns no rows if build side
      // has nulls in join keys. Hence, we can stop processing on first null.
      noMoreInput();
      return;
    }
  }

  spillInput(input);
  if (!activeRows_.hasSelections()) {
    // 直接返回
    return;
  }

  if (analyzeKeys_ && hashes_.size() < activeRows_.end()) {
    hashes_.resize(activeRows_.end());
  }

  // As long as analyzeKeys is true, we keep running the keys through
  // the Vectorhashers so that we get a possible mapping of the keys
  // to small ints for array or normalized key. When mayUseValueIds is
  // false for the first time we stop. We do not retain the value ids
  // since the final ones will only be known after all data is
  // received.
  for (auto& hasher : hashers) {
    // TODO: Load only for active rows, except if right/full outer join.
    if (analyzeKeys_) {
      hasher->computeValueIds(activeRows_, hashes_);
      analyzeKeys_ = hasher->mayUseValueIds();
    }
  }
  auto rows = table_->rows();
  auto nextOffset = rows->nextOffset();
  activeRows_.applyToSelected([&](auto rowIndex) {
    // 创建新行
    char* newRow = rows->newRow();
    if (nextOffset) {
        // 设置成NULL
      *reinterpret_cast<char**>(newRow + nextOffset) = nullptr;
    }
    // Store the columns for each row in sequence. At probe time
    // strings of the row will probably be in consecutive places, so
    // reading one will prime the cache for the next.
    for (auto i = 0; i < hashers.size(); ++i) {
      rows->store(hashers[i]->decodedVector(), rowIndex, newRow, i);
    }
    for (auto i = 0; i < dependentChannels_.size(); ++i) {
      rows->store(*decoders_[i], rowIndex, newRow, i + hashers.size());
    }
  });
}

bool HashBuild::ensureInputFits(RowVectorPtr& input) {
  // NOTE: we don't need memory reservation if all the partitions are spilling
  // as we spill all the input rows to disk directly.
  if (!spillEnabled() || spiller_ == nullptr || spiller_->isAllSpilled()) {
    // 直接返回就行
    return true;
  }

  // NOTE: we simply reserve memory all inputs even though some of them are
  // spilling directly. It is okay as we will accumulate the extra reservation
  // in the operator's memory pool, and won't make any new reservation if there
  // is already sufficient reservations.
  if (!reserveMemory(input)) {
    if (!requestSpill(input)) {
      return false;
    }
  } else {
    // Check if any other peer operator has requested group spill.
    if (waitSpill(input)) {
      return false;
    }
  }
  return true;
}

// 预留内存
bool HashBuild::reserveMemory(const RowVectorPtr& input) {
  VELOX_CHECK(spillEnabled());

  numSpillRows_ = 0;
  numSpillBytes_ = 0;

  auto rows = table_->rows();
  auto numRows = rows->numRows();
  // 直接返回即可
  if (numRows == 0) {
    // Skip the memory reservation for the first input as we are lack of memory
    // usage stats for estimation. It is safe to skip as the query should have
    // sufficient memory initially.
    return true;
  }

  auto [freeRows, outOfLineFreeBytes] = rows->freeSpace();
  auto outOfLineBytes =
      rows->stringAllocator().retainedSize() - outOfLineFreeBytes;
  auto outOfLineBytesPerRow = std::max<uint64_t>(1, outOfLineBytes / numRows);
  int64_t flatBytes = input->estimateFlatSize();

  // Test-only spill path.
  if (testingTriggerSpill()) {
    numSpillRows_ = std::max<int64_t>(1, numRows / 10);
    numSpillBytes_ = numSpillRows_ * outOfLineBytesPerRow;
    return false;
  }

  if (freeRows > input->size() &&
      (outOfLineBytes == 0 || outOfLineFreeBytes >= flatBytes)) {
    // Enough free rows for input rows and enough variable length free
    // space for the flat size of the whole vector. If outOfLineBytes
    // is 0 there is no need for variable length space.
    return true;
  }

  // If there is variable length data we take the flat size of the
  // input as a cap on the new variable length data needed.
  const auto increment =
      rows->sizeIncrement(input->size(), outOfLineBytes ? flatBytes : 0);

  auto tracker = CHECK_NOTNULL(allocator_->tracker());
  // There must be at least 2x the increments in reservation.
  if (tracker->getAvailableReservation() > 2 * increment) {
    return true;
  }

  // Check if we can increase reservation. The increment is the larger of
  // twice the maximum increment from this input and
  // 'spillableReservationGrowthPct_' of the current reservation.
  auto targetIncrement = std::max<int64_t>(
      increment * 2,
      tracker->getCurrentUserBytes() *
          spillConfig()->spillableReservationGrowthPct / 100);
  if (tracker->maybeReserve(targetIncrement)) {
    return true;
  }
  numSpillRows_ = std::max<int64_t>(
      1, targetIncrement / (rows->fixedRowSize() + outOfLineBytesPerRow));
  numSpillBytes_ = numSpillRows_ * outOfLineBytesPerRow;
  return false;
}

void HashBuild::spillInput(const RowVectorPtr& input) {
  VELOX_CHECK_EQ(input->size(), activeRows_.size());

  if (!spillEnabled() || spiller_ == nullptr || !spiller_->isAnySpilled() ||
      !activeRows_.hasSelections()) {
    return;
  }
    // 数据行数
  const auto numInput = input->size();
  prepareInputIndicesBuffers(numInput, spiller_->spilledPartitionSet());
  computeSpillPartitions(input);

  vector_size_t numSpillInputs = 0;
  for (auto row = 0; row < numInput; ++row) {
    const auto partition = spillPartitions_[row];
    if (FOLLY_UNLIKELY(!activeRows_.isValid(row))) {
      continue;
    }
    // 如果这个分区没有spill，则无所谓了
    if (!spiller_->isSpilled(partition)) {
      continue;
    }
    // 设置该行无效
    activeRows_.setValid(row, false);
    ++numSpillInputs;
    rawSpillInputIndicesBuffers_[partition][numSpillInputs_[partition]++] = row;
  }
  // 没有直接返回
  if (numSpillInputs == 0) {
    return;
  }

  maybeSetupSpillChildVectors(input);

  for (uint32_t partition = 0; partition < numSpillInputs_.size();
       ++partition) {
    const int numInputs = numSpillInputs_[partition];
    if (numInputs == 0) {
      continue;
    }
    VELOX_CHECK(spiller_->isSpilled(partition));
    spillPartition(
        partition, numInputs, spillInputIndicesBuffers_[partition], input);
  }
  activeRows_.updateBounds();
}

void HashBuild::maybeSetupSpillChildVectors(const RowVectorPtr& input) {
  if (isInputFromSpill()) {
    return;
  }
  int32_t spillChannel = 0;
  for (const auto& channel : keyChannels_) {
    spillChildVectors_[spillChannel++] = input->childAt(channel);
  }
  for (const auto& channel : dependentChannels_) {
    spillChildVectors_[spillChannel++] = input->childAt(channel);
  }
}

void HashBuild::prepareInputIndicesBuffers(
    vector_size_t numInput,
    const SpillPartitionNumSet& spillPartitions) {
  const auto maxIndicesBufferBytes = numInput * sizeof(vector_size_t);
  // 遍历每个分区
  for (const auto& partition : spillPartitions) {
    if (spillInputIndicesBuffers_[partition] == nullptr ||
        (spillInputIndicesBuffers_[partition]->size() <
         maxIndicesBufferBytes)) {
      spillInputIndicesBuffers_[partition] = allocateIndices(numInput, pool());
      rawSpillInputIndicesBuffers_[partition] =
          spillInputIndicesBuffers_[partition]->asMutable<vector_size_t>();
    }
  }
  std::fill(numSpillInputs_.begin(), numSpillInputs_.end(), 0);
}

void HashBuild::computeSpillPartitions(const RowVectorPtr& input) {
  if (hashes_.size() < activeRows_.end()) {
    hashes_.resize(activeRows_.end());
  }
  const auto& hashers = table_->hashers();
  for (auto i = 0; i < hashers.size(); ++i) {
    auto& hasher = hashers[i];
    if (hasher->channel() != kConstantChannel) {
      hashers[i]->hash(activeRows_, i > 0, hashes_);
    } else {
      hashers[i]->hashPrecomputed(activeRows_, i > 0, hashes_);
    }
  }
    // 保存数据分区到哪个分区上
  spillPartitions_.resize(input->size());
  for (auto i = 0; i < spillPartitions_.size(); ++i) {
    spillPartitions_[i] = spiller_->hashBits().partition(hashes_[i]);
  }
}

void HashBuild::spillPartition(
    uint32_t partition,
    vector_size_t size,
    const BufferPtr& indices,
    const RowVectorPtr& input) {
  VELOX_DCHECK(spillEnabled());

  if (isInputFromSpill()) {
    spiller_->spill(partition, wrap(size, indices, input));
  } else {
    spiller_->spill(
        partition,
        wrap(size, indices, tableType_, spillChildVectors_, input->pool()));
  }
}

bool HashBuild::requestSpill(RowVectorPtr& input) {
  VELOX_CHECK_GT(numSpillRows_, 0);
  VELOX_CHECK_GT(numSpillBytes_, 0);

  input_ = std::move(input);
  if (spillGroup_->requestSpill(*this, future_)) {
    VELOX_CHECK(future_.valid());
    // 设置状态
    setState(State::kWaitForSpill);
    return false;
  }
  input = std::move(input_);
  return true;
}

bool HashBuild::waitSpill(RowVectorPtr& input) {
    // 不需要Spill，直接返回false
  if (!spillGroup_->needSpill()) {
    return false;
  }
    // 等待Spill
  if (spillGroup_->waitSpill(*this, future_)) {
    VELOX_CHECK(future_.valid());
    input_ = std::move(input);
    setState(State::kWaitForSpill);
    return true;
  }
  return false;
}

void HashBuild::runSpill(const std::vector<Operator*>& spillOperators) {
  VELOX_CHECK(spillEnabled());
  VELOX_CHECK(!spiller_->state().isAllPartitionSpilled());

  uint64_t targetRows = 0;
  uint64_t targetBytes = 0;
  std::vector<Spiller*> spillers;
  spillers.reserve(spillOperators.size());
  // 遍历每个算子
  for (auto& spillOp : spillOperators) {
    // 转换成HashBuild算子？
    HashBuild* build = dynamic_cast<HashBuild*>(spillOp);
    VELOX_CHECK_NOT_NULL(build);
    // 保存spiller_的裸指针，每个算子都有一个Spiller
    spillers.push_back(build->spiller_.get());
    build->addAndClearSpillTarget(targetRows, targetBytes);
  }
  VELOX_CHECK_GT(targetRows, 0);
  VELOX_CHECK_GT(targetBytes, 0);

  std::vector<Spiller::SpillableStats> spillableStats(
      spiller_->hashBits().numPartitions());
  for (auto* spiller : spillers) {
    spiller->fillSpillRuns(spillableStats);
  }

  // Sort the partitions based on the amount of spillable data.
  SpillPartitionNumSet partitionsToSpill;
  std::vector<int32_t> partitionIndices(spillableStats.size());
  std::iota(partitionIndices.begin(), partitionIndices.end(), 0);
  // 从大到小进行排序
  std::sort(
      partitionIndices.begin(),
      partitionIndices.end(),
      [&](int32_t lhs, int32_t rhs) {
        return spillableStats[lhs].numBytes > spillableStats[rhs].numBytes;
      });
  int64_t numRows = 0;
  int64_t numBytes = 0;
  for (auto partitionNum : partitionIndices) {
    if (spillableStats[partitionNum].numBytes == 0) {
        // 如果已经是0了，直接跳出就行了
      break;
    }
    partitionsToSpill.insert(partitionNum);
    numRows += spillableStats[partitionNum].numRows;
    numBytes += spillableStats[partitionNum].numBytes;
    if (numRows >= targetRows && numBytes >= targetBytes) {
        // 跳出
      break;
    }
  }
  VELOX_CHECK(!partitionsToSpill.empty());

  // TODO: consider to offload the partition spill processing to an executor to
  // run in parallel.
  for (auto* spiller : spillers) {
    spiller->spill(partitionsToSpill);
  }
}

void HashBuild::addAndClearSpillTarget(uint64_t& numRows, uint64_t& numBytes) {
  numRows += numSpillRows_;
  numSpillRows_ = 0;
  numBytes += numSpillBytes_;
  numSpillBytes_ = 0;
}

void HashBuild::noMoreInput() {
  checkRunning();

  if (noMoreInput_) {
    return;
  }
  Operator::noMoreInput();

  noMoreInputInternal();
}

void HashBuild::noMoreInputInternal() {
  if (spillEnabled()) {
    // 停止
    spillGroup_->operatorStopped(*this);
  }

  if (!finishHashBuild()) {
    return;
  }

  postHashBuildProcess();
}

bool HashBuild::finishHashBuild() {
  checkRunning();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Driver>> peers;
  // The last Driver to hit HashBuild::finish gathers the data from
  // all build Drivers and hands it over to the probe side. At this
  // point all build Drivers are continued and will free their
  // state. allPeersFinished is true only for the last Driver of the
  // build pipeline.
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    // 设置当前状态为等待
    setState(State::kWaitForBuild);
    return false;
  }

  std::vector<std::unique_ptr<BaseHashTable>> otherTables;
  otherTables.reserve(peers.size());
  SpillPartitionSet spillPartitions;
  Spiller::Stats spillStats;
  if (joinHasNullKeys_ && isNullAwareAntiJoin(joinType_)) {
    joinBridge_->setAntiJoinHasNullKeys();
  } else {
    // 遍历每一个driver
    for (auto& peer : peers) {
        // 拿出算子
      auto op = peer->findOperator(planNodeId());
      HashBuild* build = dynamic_cast<HashBuild*>(op);
      VELOX_CHECK(build);
      if (build->joinHasNullKeys_) {
        joinHasNullKeys_ = true;
        if (isNullAwareAntiJoin(joinType_)) {
          break;
        }
      }
      // 把哈希表移动进来
      otherTables.push_back(std::move(build->table_));
      if (build->spiller_ != nullptr) {
        // 更新Spiller统计数据？
        spillStats += build->spiller_->stats();
        build->spiller_->finishSpill(spillPartitions);
      }
    }

    if (joinHasNullKeys_ && isNullAwareAntiJoin(joinType_)) {
      joinBridge_->setAntiJoinHasNullKeys();
    } else {
        // 如果spiller不是nullptr
      if (spiller_ != nullptr) {
        spillStats += spiller_->stats();

        {
          auto lockedStats = stats_.wlock();
          lockedStats->spilledBytes += spillStats.spilledBytes;
          lockedStats->spilledRows += spillStats.spilledRows;
          lockedStats->spilledPartitions += spillStats.spilledPartitions;
          lockedStats->spilledFiles += spillStats.spilledFiles;
        }
        
        spiller_->finishSpill(spillPartitions);

        // Verify all the spilled partitions are not empty as we won't spill on
        // an empty one.
        for (const auto& spillPartitionEntry : spillPartitions) {
          VELOX_CHECK_GT(spillPartitionEntry.second->numFiles(), 0);
        }
      }

      const bool hasOthers = !otherTables.empty();
      table_->prepareJoinTable(
          std::move(otherTables),
          hasOthers ? operatorCtx_->task()->queryCtx()->executor() : nullptr);

      addRuntimeStats();
      if (joinBridge_->setHashTable(
              std::move(table_),
              std::move(spillPartitions),
              joinHasNullKeys_)) {
        spillGroup_->restart();
      }
    }
  }

  // Realize the promises so that the other Drivers (which were not
  // the last to finish) can continue from the barrier and finish.
  peers.clear();
  for (auto& promise : promises) {
    // 通知所有的future
    promise.setValue();
  }
  return true;
}

void HashBuild::postHashBuildProcess() {
  checkRunning();

  // Release the unused memory reservation since we have finished the table
  // build.
  operatorCtx_->allocator()->tracker()->release();

    // 如果没有启动Spill，则设置状态为Finished，返回
  if (!spillEnabled()) {
    setState(State::kFinish);
    return;
  }

  auto spillInput = joinBridge_->spillInputOrFuture(&future_);
  if (!spillInput.has_value()) {
    VELOX_CHECK(future_.valid());
    // 等待探测？
    setState(State::kWaitForProbe);
    return;
  }
  setupSpillInput(std::move(spillInput.value()));
}

void HashBuild::setupSpillInput(HashJoinBridge::SpillInput spillInput) {
  checkRunning();

  if (spillInput.spillPartition == nullptr) {
    // 设置状态为完成
    setState(State::kFinish);
    return;
  }

  table_.reset();
  spiller_.reset();
  spillInputReader_.reset();

  // Reset the key and dependent channels as the spilled data columns have
  // already been ordered.
  std::iota(keyChannels_.begin(), keyChannels_.end(), 0);
  std::iota(
      dependentChannels_.begin(),
      dependentChannels_.end(),
      keyChannels_.size());

  setupTable();
  setupSpiller(spillInput.spillPartition.get());

  // Start to process spill input.
  processSpillInput();
}

// 处理输入
void HashBuild::processSpillInput() {
  checkRunning();

  while (spillInputReader_->nextBatch(input_)) {
    addInput(std::move(input_));
    if (!isRunning()) {
      return;
    }
  }
  noMoreInputInternal();
}

void HashBuild::addRuntimeStats() {
  // Report range sizes and number of distinct values for the join keys.
  const auto& hashers = table_->hashers();
  uint64_t asRange;
  uint64_t asDistinct;
  auto lockedStats = stats_.wlock();
  for (auto i = 0; i < hashers.size(); i++) {
    hashers[i]->cardinality(0, asRange, asDistinct);
    if (asRange != VectorHasher::kRangeTooLarge) {
      lockedStats->addRuntimeStat(
          fmt::format("rangeKey{}", i), RuntimeCounter(asRange));
    }
    if (asDistinct != VectorHasher::kRangeTooLarge) {
      lockedStats->addRuntimeStat(
          fmt::format("distinctKey{}", i), RuntimeCounter(asDistinct));
    }
  }
  // Add max spilling level stats if spilling has been triggered.
  if (spiller_ != nullptr && spiller_->isAnySpilled()) {
    lockedStats->addRuntimeStat(
        "maxSpillLevel",
        RuntimeCounter(
            spillConfig()->spillLevel(spiller_->hashBits().begin())));
  }
}

BlockingReason HashBuild::isBlocked(ContinueFuture* future) {
  switch (state_) {
    case State::kRunning:
      if (isInputFromSpill()) {
        processSpillInput();
      }
      break;
    case State::kFinish:
      break;
    case State::kWaitForSpill:
      if (!future_.valid()) {
        setRunning();
        VELOX_CHECK_NOT_NULL(input_);
        addInput(std::move(input_));
      }
      break;
    case State::kWaitForBuild:
      FOLLY_FALLTHROUGH;
    case State::kWaitForProbe:
      if (!future_.valid()) {
        setRunning();
        postHashBuildProcess();
      }
      break;
    default:
      VELOX_UNREACHABLE("Unexpected state: {}", stateName(state_));
      break;
  }
  if (future_.valid()) {
    VELOX_CHECK(!isRunning() && !isFinished());
    *future = std::move(future_);
  }
  return fromStateToBlockingReason(state_);
}

bool HashBuild::isFinished() {
  return state_ == State::kFinish;
}
// 算子是否处于运行态
bool HashBuild::isRunning() const {
  return state_ == State::kRunning;
}

void HashBuild::checkRunning() const {
  VELOX_CHECK(isRunning(), stateName(state_));
}

void HashBuild::setRunning() {
  setState(State::kRunning);
}

void HashBuild::setState(State state) {
  checkStateTransition(state);
  state_ = state;
}

void HashBuild::checkStateTransition(State state) {
  VELOX_CHECK_NE(state_, state);
  switch (state) {
    case State::kRunning:
      if (!spillEnabled()) {
        VELOX_CHECK_EQ(state_, State::kWaitForBuild);
      } else {
        VELOX_CHECK_NE(state_, State::kFinish);
      }
      break;
    case State::kWaitForBuild:
      FOLLY_FALLTHROUGH;
    case State::kWaitForSpill:
      FOLLY_FALLTHROUGH;
    case State::kWaitForProbe:
      FOLLY_FALLTHROUGH;
    case State::kFinish:
      VELOX_CHECK_EQ(state_, State::kRunning);
      break;
    default:
      VELOX_UNREACHABLE(stateName(state_));
      break;
  }
}

std::string HashBuild::stateName(State state) {
  switch (state) {
    case State::kRunning:
      return "RUNNING";
    case State::kWaitForSpill:
      return "WAIT_FOR_SPILL";
    case State::kWaitForBuild:
      return "WAIT_FOR_BUILD";
    case State::kWaitForProbe:
      return "WAIT_FOR_PROBE";
    case State::kFinish:
      return "FINISH";
    default:
      return fmt::format("UNKNOWN: {}", static_cast<int>(state));
  }
}

bool HashBuild::testingTriggerSpill() {
  // Test-only spill path.
  if (spillConfig()->testSpillPct == 0) {
    return false;
  }
  return folly::hasher<uint64_t>()(++spillTestCounter_) % 100 <=
      spillConfig()->testSpillPct;
}

} // namespace facebook::velox::exec
