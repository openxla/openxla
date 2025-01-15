/* Copyright 2018 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_STREAM_EXECUTOR_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_STREAM_EXECUTOR_ALLOCATOR_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "tsl/profiler/lib/traceme.h"

namespace stream_executor {

// Implements a tsl::SubAllocator interface for StreamExecutor-based devices.
class StreamExecutorAllocator : public tsl::SubAllocator {
 public:
  explicit StreamExecutorAllocator(
      std::unique_ptr<MemoryAllocator> memory_allocator, MemoryType memory_type,
      int index, const std::vector<Visitor>& alloc_visitors,
      const std::vector<Visitor>& free_visitors)
      : tsl::SubAllocator(alloc_visitors, free_visitors),
        memory_allocator_(std::move(memory_allocator)),
        memory_type_(memory_type),
        index_(index) {}

  ~StreamExecutorAllocator() override = default;

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    tsl::profiler::TraceMe traceme("StreamExecutorAllocator::Alloc");

    void* ptr = nullptr;

    if (num_bytes > 0) {
      auto allocation = memory_allocator_->Allocate(num_bytes);
      if (!allocation.ok()) {
        LOG(WARNING) << "could not allocate pinned host memory of size: "
                     << num_bytes;
        return nullptr;
      }

      ptr = (*allocation)->opaque();
      VisitAlloc(ptr, index_, num_bytes);

      absl::MutexLock lock(&mutex_);
      allocations_[ptr] = std::move(*allocation);
    }

    *bytes_received = num_bytes;

    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    tsl::profiler::TraceMe traceme("StreamExecutorAllocator::Free");

    if (ptr != nullptr) {
      VisitFree(ptr, index_, num_bytes);
      absl::MutexLock lock(&mutex_);
      allocations_.erase(ptr);
    }
  }

  bool SupportsCoalescing() const override { return false; }

  tsl::AllocatorMemoryType GetMemoryType() const override {
    if (memory_type_ == MemoryType::kHost) {
      return tsl::AllocatorMemoryType::kHostPinned;
    } else {
      return tsl::AllocatorMemoryType::kDevice;
    }
  }

 private:
  std::unique_ptr<MemoryAllocator> memory_allocator_;
  MemoryType memory_type_;
  int index_;

  StreamExecutorAllocator(const StreamExecutorAllocator&) = delete;
  void operator=(const StreamExecutorAllocator&) = delete;

  absl::Mutex mutex_;
  absl::flat_hash_map<void*, std::unique_ptr<MemoryAllocation>> allocations_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_STREAM_EXECUTOR_ALLOCATOR_H_
