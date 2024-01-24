/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME3_TMA_METADATA_H_
#define XLA_SERVICE_GPU_RUNTIME3_TMA_METADATA_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

// TmaMetadata interface.
//
// This allows passing around TmaMetadata without depending on a specific GPU
// library (such as CUDA).
struct TmaMetadata {
  TmaMetadata() = default;
  TmaMetadata(const TmaMetadata&) = delete;
  TmaMetadata& operator=(const TmaMetadata&) = delete;
  virtual ~TmaMetadata() = default;

  virtual std::unique_ptr<TmaMetadata> Clone() const = 0;
  virtual std::string ToString() const = 0;
};

#if GOOGLE_CUDA

// Information describing a CUDA tensor map.
struct CudaTensorMapInfo {
  std::string ToString() const;

  CUtensorMapDataType tensor_data_type;
  uint32_t tensor_rank;
  // The index of the kernel argument used for the pointer of the tensor.
  int global_address_arg_index;
  absl::InlinedVector<uint64_t, 4> global_dim;
  // `global_strides` doesn't include the stride for dim_index=0.
  absl::InlinedVector<uint64_t, 4> global_strides;
  absl::InlinedVector<uint32_t, 4> box_dim;
  absl::InlinedVector<uint32_t, 4> element_strides;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2_promotion;
  CUtensorMapFloatOOBfill oob_fill;
};

// Information describing a CUDA tensor map, after we have the tensor pointer.
// To be used with cuTensorMapEncodeTiled.
struct ConcreteCudaTensorMapInfo {
  ConcreteCudaTensorMapInfo();
  ConcreteCudaTensorMapInfo(CudaTensorMapInfo info, void* global_address);
  std::string ToString() const;

  bool operator==(const ConcreteCudaTensorMapInfo& other) const;
  template <typename H>
  friend H AbslHashValue(H h, const ConcreteCudaTensorMapInfo& info) {
    return H::combine(std::move(h), info.tensor_rank, info.global_address,
                      info.global_dim, info.global_strides, info.box_dim,
                      info.element_strides, info.interleave, info.swizzle,
                      info.l2_promotion, info.oob_fill);
  }

  CUtensorMapDataType tensor_data_type;
  uint32_t tensor_rank;
  void* global_address;
  absl::InlinedVector<uint64_t, 4> global_dim;
  // `global_strides` doesn't include the stride for dim_index=0.
  absl::InlinedVector<uint64_t, 4> global_strides;
  absl::InlinedVector<uint32_t, 4> box_dim;
  absl::InlinedVector<uint32_t, 4> element_strides;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2_promotion;
  CUtensorMapFloatOOBfill oob_fill;
};

// CUDA-specific TmaMetadata.
struct CudaTmaMetadata : public TmaMetadata {
  explicit CudaTmaMetadata(std::vector<CudaTensorMapInfo> tensor_map_infos);
  ~CudaTmaMetadata() override;

  std::unique_ptr<TmaMetadata> Clone() const override;
  std::string ToString() const override;

  std::vector<CudaTensorMapInfo> tensor_map_infos;
};

// A singleton class that uploads the tensor maps to the devices in a cached
// way.
//
// Each TensorMap is only uploaded once per device and never deallocated.
// TODO(tdanyluk): It would be better to upload all tensor maps together in one
// step.
class CudaTensorMapManager {
 public:
  static CudaTensorMapManager& GetInstance();

  // If the given tensor map was already uploaded to the device of `stream`,
  // then return the corresponding device pointer, otherwise upload it and then
  // return the device pointer.
  absl::StatusOr<stream_executor::DeviceMemoryBase> GetOrCreateDeviceTensorMap(
      ConcreteCudaTensorMapInfo info, stream_executor::Stream& stream)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  CudaTensorMapManager();

  absl::Mutex mu_;
  absl::flat_hash_map<
      std::pair<ConcreteCudaTensorMapInfo, int /*device_ordinal*/>,
      stream_executor::OwningDeviceMemory>
      tensor_maps_ ABSL_GUARDED_BY(mu_);
};

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME3_TMA_METADATA_H_
