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

#include "velox/exec/Unnest.h"
#include "velox/common/base/Nulls.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {
Unnest::Unnest(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::UnnestNode>& unnestNode)
    : Operator(
          driverCtx,
          unnestNode->outputType(),
          operatorId,
          unnestNode->id(),
          "Unnest"),
      withOrdinality_(unnestNode->withOrdinality()) {
        // 输入类型
  const auto& inputType = unnestNode->sources()[0]->outputType();
  const auto& unnestVariables = unnestNode->unnestVariables();
  // 遍历每一个要unnest的变量
  for (const auto& variable : unnestVariables) {
    // 如果不是数组或者MAP，则抛出异常
    if (!variable->type()->isArray() && !variable->type()->isMap()) {
      VELOX_UNSUPPORTED("Unnest operator supports only ARRAY and MAP types")
    }
    // 获取输入位置？
    unnestChannels_.push_back(inputType->getChildIdx(variable->name()));
  }

    // 解码向量
  unnestDecoded_.resize(unnestVariables.size());

  if (withOrdinality_) {
    // 存在序数列，则应该是BIGINT类型
    VELOX_CHECK_EQ(
        outputType_->children().back(),
        BIGINT(),
        "Ordinality column should be BIGINT type.")
  }

  column_index_t outputChannel = 0;
  // 进行重复的变量
  for (const auto& variable : unnestNode->replicateVariables()) {
    identityProjections_.emplace_back(
        inputType->getChildIdx(variable->name()), outputChannel++);
  }
}

// 添加输入向量
void Unnest::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr Unnest::getOutput() {
  if (!input_) {
    // 输入向量为空指针，则直接返回空指针
    return nullptr;
  }

    // 输入的行数
  auto size = input_->size();
  // 对选择向量进行扩容
  inputRows_.resize(size);

    // 分配内存
  // The max number of elements at each row across all unnested columns.
  auto maxSizes = AlignedBuffer::allocate<int64_t>(size, pool(), 0);
  auto rawMaxSizes = maxSizes->asMutable<int64_t>();

  std::vector<const vector_size_t*> rawSizes;
  std::vector<const vector_size_t*> rawOffsets;
  std::vector<const vector_size_t*> rawIndices;

  rawSizes.resize(unnestChannels_.size());
  rawOffsets.resize(unnestChannels_.size());
  rawIndices.resize(unnestChannels_.size());

    // 遍历每一个要unnest的channel
  for (auto channel = 0; channel < unnestChannels_.size(); ++channel) {
    // 拿出对应的输入列的向量
    const auto& unnestVector = input_->childAt(unnestChannels_[channel]);
    // 进行解码
    unnestDecoded_[channel].decode(*unnestVector, inputRows_);

    auto& currentDecoded = unnestDecoded_[channel];
    // 保存索引裸指针
    rawIndices[channel] = currentDecoded.indices();

    const ArrayVector* unnestBaseArray;
    const MapVector* unnestBaseMap;
    if (unnestVector->typeKind() == TypeKind::ARRAY) {
      unnestBaseArray = currentDecoded.base()->as<ArrayVector>();
      rawSizes[channel] = unnestBaseArray->rawSizes();
      rawOffsets[channel] = unnestBaseArray->rawOffsets();
    } else {
      VELOX_CHECK(unnestVector->typeKind() == TypeKind::MAP);
      unnestBaseMap = currentDecoded.base()->as<MapVector>();
      rawSizes[channel] = unnestBaseMap->rawSizes();
      rawOffsets[channel] = unnestBaseMap->rawOffsets();
    }

    // Count max number of elements per row.
    auto currentSizes = rawSizes[channel];
    auto currentIndices = rawIndices[channel];
    // 遍历每一行
    for (auto row = 0; row < size; ++row) {
        // 如果不是NULL值
      if (!currentDecoded.isNullAt(row)) {
        auto unnestSize = currentSizes[currentIndices[row]];
        if (rawMaxSizes[row] < unnestSize) {
            // 更新该行的的最大的size?
          rawMaxSizes[row] = unnestSize;
        }
      }
    }
  }

  // Calculate the number of rows in the unnest result.
  int numElements = 0;
  for (auto row = 0; row < size; ++row) {
    numElements += rawMaxSizes[row];
  }

  if (numElements == 0) {
    // All arrays/maps are null or empty.
    input_ = nullptr;
    return nullptr;
  }

  // Create "indices" buffer to repeat rows as many times as there are elements
  // in the array (or map) in unnestDecoded.
  auto repeatedIndices = allocateIndices(numElements, pool());
  auto* rawRepeatedIndices = repeatedIndices->asMutable<vector_size_t>();
  vector_size_t index = 0;
  for (auto row = 0; row < size; ++row) {
    for (auto i = 0; i < rawMaxSizes[row]; i++) {
      rawRepeatedIndices[index++] = row;
    }
  }

  // Wrap "replicated" columns in a dictionary using 'repeatedIndices'.
  std::vector<VectorPtr> outputs(outputType_->size());
  for (const auto& projection : identityProjections_) {
    // TODO(lokax): 包装成字典向量
    outputs[projection.outputChannel] = wrapChild(
        numElements, repeatedIndices, input_->childAt(projection.inputChannel));
  }

  // Create unnest columns.
  vector_size_t outputsIndex = identityProjections_.size();
  for (auto channel = 0; channel < unnestChannels_.size(); ++channel) {
    auto& currentDecoded = unnestDecoded_[channel];
    auto currentSizes = rawSizes[channel];
    auto currentOffsets = rawOffsets[channel];
    auto currentIndices = rawIndices[channel];

    BufferPtr elementIndices = allocateIndices(numElements, pool());
    auto* rawElementIndices = elementIndices->asMutable<vector_size_t>();

    auto nulls =
        AlignedBuffer::allocate<bool>(numElements, pool(), bits::kNotNull);
    auto rawNulls = nulls->asMutable<uint64_t>();

    // Make dictionary index for elements column since they may be out of order.
    index = 0;
    bool identityMapping = true;
    // 遍历单独一列内的每一行
    for (auto row = 0; row < size; ++row) {
        // 最大大小
      auto maxSize = rawMaxSizes[row];
        // 如果当前行不是NULL值
      if (!currentDecoded.isNullAt(row)) {
        auto offset = currentOffsets[currentIndices[row]];
        auto unnestSize = currentSizes[currentIndices[row]];

        if (index != offset || unnestSize < maxSize) {
          identityMapping = false;
        }

        for (auto i = 0; i < unnestSize; i++) {
          rawElementIndices[index++] = offset + i;
        }

        // 不够的部分填充NULL值
        for (auto i = unnestSize; i < maxSize; ++i) {
          bits::setNull(rawNulls, index++, true);
        }
      } else if (maxSize > 0) {
        identityMapping = false;
        // 设置成NULL
        for (auto i = 0; i < maxSize; ++i) {
          bits::setNull(rawNulls, index++, true);
        }
      }
    }

    if (currentDecoded.base()->typeKind() == TypeKind::ARRAY) {
      // Construct unnest column using Array elements wrapped using above
      // created dictionary.
      auto unnestBaseArray = currentDecoded.base()->as<ArrayVector>();
      outputs[outputsIndex++] = identityMapping
          ? unnestBaseArray->elements()
          : wrapChild(
                numElements,
                elementIndices,
                unnestBaseArray->elements(),
                nulls);
    } else {
      // Construct two unnest columns for Map keys and values vectors wrapped
      // using above created dictionary.
      auto unnestBaseMap = currentDecoded.base()->as<MapVector>();
      outputs[outputsIndex++] = identityMapping
          ? unnestBaseMap->mapKeys()
          : wrapChild(
                numElements, elementIndices, unnestBaseMap->mapKeys(), nulls);
      outputs[outputsIndex++] = identityMapping
          ? unnestBaseMap->mapValues()
          : wrapChild(
                numElements, elementIndices, unnestBaseMap->mapValues(), nulls);
    }
  }

  if (withOrdinality_) {
    auto ordinalityVector = std::dynamic_pointer_cast<FlatVector<int64_t>>(
        BaseVector::create(BIGINT(), numElements, pool()));

    // Set the ordinality at each result row to be the index of the element in
    // the original array (or map) plus one.
    auto rawOrdinality = ordinalityVector->mutableRawValues();
    for (auto row = 0; row < size; ++row) {
      auto maxSize = rawMaxSizes[row];
      std::iota(rawOrdinality, rawOrdinality + maxSize, 1);
      rawOrdinality += maxSize;
    }

    // Ordinality column is always at the end.
    outputs.back() = std::move(ordinalityVector);
  }

  input_ = nullptr;
  return std::make_shared<RowVector>(
      pool(), outputType_, BufferPtr(nullptr), numElements, std::move(outputs));
}

bool Unnest::isFinished() {
  return noMoreInput_ && input_ == nullptr;
}
} // namespace facebook::velox::exec
