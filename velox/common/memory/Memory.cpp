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

#include "velox/common/memory/Memory.h"

#include "velox/common/base/BitUtil.h"

DEFINE_bool(
    use_mmap_allocator_for_memory_pool,
    false,
    "If true, use MmapMemoryAllocator to allocate memory for MemoryPool");

namespace facebook {
namespace velox {
namespace memory {

MemoryPool::MemoryPool(
    const std::string& name,
    std::shared_ptr<MemoryPool> parent)
    : name_(name), parent_(std::move(parent)) {}

MemoryPool::~MemoryPool() {
  VELOX_CHECK(children_.empty());
  if (parent_ != nullptr) {
    parent_->dropChild(this);
  }
}

const std::string& MemoryPool::name() const {
  return name_;
}

MemoryPool* MemoryPool::parent() const {
  return parent_.get();
}

uint64_t MemoryPool::getChildCount() const {
  folly::SharedMutex::ReadHolder guard{childrenMutex_};
  return children_.size();
}

void MemoryPool::visitChildren(
    std::function<void(MemoryPool* FOLLY_NONNULL)> visitor) const {
  folly::SharedMutex::ReadHolder guard{childrenMutex_};
  for (const auto& child : children_) {
    visitor(child);
  }
}
// 给当前内存池添加一个孩子
std::shared_ptr<MemoryPool> MemoryPool::addChild(
    const std::string& name,
    int64_t cap) {
        // 加孩子的写锁
  folly::SharedMutex::WriteHolder guard{childrenMutex_};
  // Upon name collision we would throw and not modify the map.
  auto child = genChild(shared_from_this(), name, cap);
  // 如果当前已经cap了，那孩子也要cap
  if (isMemoryCapped()) {
    child->capMemoryAllocation();
  }
  if (auto usageTracker = getMemoryUsageTracker()) {
    // 内存跟踪器也要添加一个孩子
    child->setMemoryUsageTracker(usageTracker->addChild());
  }
  children_.emplace_back(child.get());
  return child;
}
// 删除孩子
void MemoryPool::dropChild(const MemoryPool* FOLLY_NONNULL child) {
  folly::SharedMutex::WriteHolder guard{childrenMutex_};
  // Implicitly synchronized in dtor of child so it's impossible for
  // MemoryManager to access after destruction of child.
  auto iter = std::find_if(
      children_.begin(), children_.end(), [child](const MemoryPool* e) {
        return e == child;
      });
  VELOX_CHECK(iter != children_.end());
  children_.erase(iter);
}
// TODO(lokax): 看一下
size_t MemoryPool::getPreferredSize(size_t size) {
  if (size < 8) {
    return 8;
  }
  int32_t bits = 63 - bits::countLeadingZeros(size);
  size_t lower = 1ULL << bits;
  // Size is a power of 2.
  if (lower == size) {
    return size;
  }
  // If size is below 1.5 * previous power of two, return 1.5 *
  // the previous power of two, else the next power of 2.
  if (lower + (lower / 2) >= size) {
    return lower + (lower / 2);
  }
  return lower * 2;
}
// 构造函数
MemoryPoolImpl::MemoryPoolImpl(
    MemoryManager& memoryManager,
    const std::string& name,
    std::shared_ptr<MemoryPool> parent,
    int64_t cap)
    : MemoryPool{name, parent},
      memoryManager_{memoryManager},
      localMemoryUsage_{},
      cap_{cap},
      allocator_{memoryManager_.getAllocator()} {
  VELOX_USER_CHECK_GT(cap, 0);
}
// 对齐大小的实现
/* static */
int64_t MemoryPoolImpl::sizeAlign(int64_t size) {
  const auto remainder = size % alignment_;
  return (remainder == 0) ? size : (size + alignment_ - remainder);
}

void* MemoryPoolImpl::allocate(int64_t size) {
    // 首先检查是否到达上限？
  if (this->isMemoryCapped()) {
    // 抛异常
    VELOX_MEM_MANUAL_CAP();
  }
  // 进行对齐
  auto alignedSize = sizeAlign(size);
  reserve(alignedSize);
  // 使用allocator分配内存
  return allocator_.allocateBytes(alignedSize, alignment_);
}
// 分配内存并填充0
void* MemoryPoolImpl::allocateZeroFilled(int64_t numEntries, int64_t sizeEach) {
  if (this->isMemoryCapped()) {
    VELOX_MEM_MANUAL_CAP();
  }
  const auto alignedSize = sizeAlign(sizeEach * numEntries);
  reserve(alignedSize);
  return allocator_.allocateZeroFilled(alignedSize, alignment_);
}

void* MemoryPoolImpl::reallocate(
    void* FOLLY_NULLABLE p,
    int64_t size,
    int64_t newSize) {
        // 首先进行大小对齐
  auto alignedSize = sizeAlign(size);
  auto alignedNewSize = sizeAlign(newSize);
  const int64_t difference = alignedNewSize - alignedSize;
  if (FOLLY_UNLIKELY(difference <= 0)) {
    // Track and pretend the shrink took place for accounting purposes.
    release(-difference, true);
    return p;
  }

  reserve(difference);
  // 重新分配
  void* newP =
      allocator_.reallocateBytes(p, alignedSize, alignedNewSize, alignment_);
      // 分配失败抛异常
  if (FOLLY_UNLIKELY(newP == nullptr)) {
    free(p, alignedSize);
    auto errorMessage = fmt::format(
        MEM_CAP_EXCEEDED_ERROR_FORMAT,
        succinctBytes(cap_),
        succinctBytes(difference));
    VELOX_MEM_CAP_EXCEEDED(errorMessage);
  }
  return newP;
}

// 释放内存
void MemoryPoolImpl::free(void* p, int64_t size) {
  const auto alignedSize = sizeAlign(size);
  allocator_.freeBytes(p, alignedSize);
  release(alignedSize);
}

int64_t MemoryPoolImpl::getCurrentBytes() const {
  return getAggregateBytes();
}

// 返回当前pool和子树中最大的bytes
int64_t MemoryPoolImpl::getMaxBytes() const {
  return std::max(getSubtreeMaxBytes(), localMemoryUsage_.getMaxBytes());
}

void MemoryPoolImpl::setMemoryUsageTracker(
    const std::shared_ptr<MemoryUsageTracker>& tracker) {
  const auto currentBytes = getCurrentBytes();
  if (memoryUsageTracker_) {
    memoryUsageTracker_->update(-currentBytes);
  }
  memoryUsageTracker_ = tracker;
  memoryUsageTracker_->update(currentBytes);
}

const std::shared_ptr<MemoryUsageTracker>&
MemoryPoolImpl::getMemoryUsageTracker() const {
  return memoryUsageTracker_;
}

void MemoryPoolImpl::setSubtreeMemoryUsage(int64_t size) {
  updateSubtreeMemoryUsage([size](MemoryUsage& subtreeUsage) {
    subtreeUsage.setCurrentBytes(size);
  });
}

int64_t MemoryPoolImpl::updateSubtreeMemoryUsage(int64_t size) {
  int64_t aggregateBytes;
  updateSubtreeMemoryUsage([&aggregateBytes, size](MemoryUsage& subtreeUsage) {
    aggregateBytes = subtreeUsage.getCurrentBytes() + size;
    subtreeUsage.setCurrentBytes(aggregateBytes);
  });
  return aggregateBytes;
}

// 返回当前pool的容量
int64_t MemoryPoolImpl::cap() const {
  return cap_;
}

// 返回对齐大小
uint16_t MemoryPoolImpl::getAlignment() const {
  return alignment_;
}
// 冻结内存？
void MemoryPoolImpl::capMemoryAllocation() {
  capped_.store(true);
  for (const auto& child : children_) {
    child->capMemoryAllocation();
  }
}

void MemoryPoolImpl::uncapMemoryAllocation() {
  // This means if we try to post-order traverse the tree like we do
  // in MemoryManager, only parent has the right to lift the cap.
  // This suffices because parent will then recursively lift the cap on the
  // entire tree.
  if (getAggregateBytes() > cap()) {
    return;
  }
  if (parent_ != nullptr && parent_->isMemoryCapped()) {
    return;
  }
  capped_.store(false);
  visitChildren([](MemoryPool* child) { child->uncapMemoryAllocation(); });
}
// 返回内存是否达到上限
bool MemoryPoolImpl::isMemoryCapped() const {
  return capped_.load();
}

// 生成一个孩子
std::shared_ptr<MemoryPool> MemoryPoolImpl::genChild(
    std::shared_ptr<MemoryPool> parent,
    const std::string& name,
    int64_t cap) {
  return std::make_shared<MemoryPoolImpl>(memoryManager_, name, parent, cap);
}

const MemoryUsage& MemoryPoolImpl::getLocalMemoryUsage() const {
  return localMemoryUsage_;
}

// 等于当前pool和该pool孩子分配的内存大小
int64_t MemoryPoolImpl::getAggregateBytes() const {
  int64_t aggregateBytes = localMemoryUsage_.getCurrentBytes();
  accessSubtreeMemoryUsage([&aggregateBytes](const MemoryUsage& subtreeUsage) {
    aggregateBytes += subtreeUsage.getCurrentBytes();
  });
  return aggregateBytes;
}

int64_t MemoryPoolImpl::getSubtreeMaxBytes() const {
  int64_t maxBytes;
  accessSubtreeMemoryUsage([&maxBytes](const MemoryUsage& subtreeUsage) {
    // TODO(lokax): 这个地方的实现应该是不对的，应该用max函数
    maxBytes = subtreeUsage.getMaxBytes();
  });
  return maxBytes;
}

void MemoryPoolImpl::accessSubtreeMemoryUsage(
    std::function<void(const MemoryUsage&)> visitor) const {
  folly::SharedMutex::ReadHolder readLock{subtreeUsageMutex_};
  visitor(subtreeMemoryUsage_);
}

void MemoryPoolImpl::updateSubtreeMemoryUsage(
    std::function<void(MemoryUsage&)> visitor) {
  folly::SharedMutex::WriteHolder writeLock{subtreeUsageMutex_};
  visitor(subtreeMemoryUsage_);
}

void MemoryPoolImpl::reserve(int64_t size) {
  if (memoryUsageTracker_) {
    memoryUsageTracker_->update(size);
  }
  localMemoryUsage_.incrementCurrentBytes(size);

  bool success = memoryManager_.reserve(size);
  bool manualCap = isMemoryCapped();
  int64_t aggregateBytes = getAggregateBytes();
  if (UNLIKELY(!success || manualCap || aggregateBytes > cap_)) {
    // NOTE: If we can make the reserve and release a single transaction we
    // would have more accurate aggregates in intermediate states. However, this
    // is low-pri because we can only have inflated aggregates, and be on the
    // more conservative side.
    release(size);
    if (!success) {
        // 超出总内存限制？抛异常
      VELOX_MEM_MANAGER_CAP_EXCEEDED(memoryManager_.getMemoryQuota());
    }
    if (manualCap) {
        // 手动cap了，抛异常
      VELOX_MEM_MANUAL_CAP();
    }
    // 抛异常
    auto errorMessage = fmt::format(
        MEM_CAP_EXCEEDED_ERROR_FORMAT,
        succinctBytes(cap_),
        succinctBytes(size));
    VELOX_MEM_CAP_EXCEEDED(errorMessage);
  }
}

void MemoryPoolImpl::release(int64_t size, bool mock) {
  memoryManager_.release(size);
  localMemoryUsage_.incrementCurrentBytes(-size);
  if (memoryUsageTracker_) {
    memoryUsageTracker_->update(-size, mock);
  }
}

MemoryManager::MemoryManager(
    int64_t memoryQuota,
    MemoryAllocator* FOLLY_NONNULL allocator)
    : allocator_{std::move(allocator)},
      memoryQuota_{memoryQuota},
      root_{std::make_shared<MemoryPoolImpl>(
          *this,
          kRootNodeName.str(),
          nullptr,
          memoryQuota)} {
  VELOX_USER_CHECK_GE(memoryQuota_, 0);
}

MemoryManager::~MemoryManager() {
  auto currentBytes = getTotalBytes();
  if (currentBytes > 0) {
    LOG(WARNING) << "Leaked total memory of " << currentBytes << " bytes.";
  }
}
// 返回内存配额
int64_t MemoryManager::getMemoryQuota() const {
  return memoryQuota_;
}
// 返回根内存池
MemoryPool& MemoryManager::getRoot() const {
  return *root_;
}
// 返回一个孩子内存池
std::shared_ptr<MemoryPool> MemoryManager::getChild(int64_t cap) {
  return root_->addChild(
      fmt::format(
          "default_usage_node_{}",
          folly::to<std::string>(folly::Random::rand64())),
      cap);
}
// 总分配的内存，使用原子变量
int64_t MemoryManager::getTotalBytes() const {
  return totalBytes_.load(std::memory_order_relaxed);
}

bool MemoryManager::reserve(int64_t size) {
  return totalBytes_.fetch_add(size, std::memory_order_relaxed) + size <=
      memoryQuota_;
}

void MemoryManager::release(int64_t size) {
  totalBytes_.fetch_sub(size, std::memory_order_relaxed);
}

// 返回allocattor
MemoryAllocator& MemoryManager::getAllocator() {
  return *allocator_;
}

IMemoryManager& getProcessDefaultMemoryManager() {
  return MemoryManager::getInstance();
}

std::shared_ptr<MemoryPool> getDefaultMemoryPool(int64_t cap) {
  auto& memoryManager = getProcessDefaultMemoryManager();
  return memoryManager.getChild(cap);
}

} // namespace memory
} // namespace velox
} // namespace facebook
