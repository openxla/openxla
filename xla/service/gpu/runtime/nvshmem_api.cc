/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/nvshmem_api.h"

#include "absl/strings/str_format.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"
#include "third_party/nvshmem/nvshmem.h"
#include "third_party/nvshmem/nvshmemx.h"

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// Macros to return or warn on NVSHMEM errors.
//==-----------------------------------------------------------------------===//

static absl::Status NvshmemToStatus(int s, const char* file, int64_t line,
                                    const char* expr) {
  if (s == 0) return absl::OkStatus();

  return absl::InternalError(
      absl::StrFormat("%s:%d: NVSHMEM operation %s failed."
                      " For extra logging, rerun with 'NVSHMEM_DEBUG=INFO'.",
                      file, line, expr));
}

#define XLA_NVSHMEM_STATUS(expr) \
  xla::gpu::NvshmemToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_NVSHMEM_RETURN_IF_ERROR(expr)      \
  do {                                         \
    absl::Status s = XLA_NVSHMEM_STATUS(expr); \
    if (!s.ok()) {                             \
      return s;                                \
    }                                          \
  } while (0)

#define XLA_NVSHMEM_LOG_IF_ERROR(expr)         \
  do {                                         \
    absl::Status s = XLA_NVSHMEM_STATUS(expr); \
    if (!s.ok()) {                             \
      LOG(ERROR) << s.ToString();              \
    }                                          \
  } while (0)

#define XLA_NVSHMEM_CHECK(expr) CHECK(XLA_NVSHMEM_STATUS(expr).ok())

int NvshmemApi::process_id_ = -1;
size_t NvshmemApi::num_processes_ = 0;
size_t NvshmemApi::device_count_per_process_ = 0;
std::function<absl::StatusOr<std::string>(std::string_view)>
    NvshmemApi::kv_store_get_ = nullptr;
std::function<absl::Status(std::string_view, std::string_view)>
    NvshmemApi::kv_store_set_ = nullptr;

NvshmemApi& NvshmemApi::Default() {
  static NvshmemApi instance;
  return instance;
}

void NvshmemApi::SetEnvInfo(
    int process_id, size_t num_processes, size_t device_count_per_process,
    std::function<absl::StatusOr<std::string>(std::string_view)> kv_store_get,
    std::function<absl::Status(std::string_view, std::string_view)>
        kv_store_set) {
  process_id_ = process_id;
  num_processes_ = num_processes;
  device_count_per_process_ = device_count_per_process;
  kv_store_get_ = kv_store_get;
  kv_store_set_ = kv_store_set;
}

NvshmemApi::NvshmemApi() {
  // Initialize NVSHMEM here since code path
  // is already protected by singleton pattern
  if (process_id_ == -1) {
    LOG(FATAL)
        << "NvshmemApi::SetEnvInfo was not called before using NVSHMEM API";
  }
  if (device_count_per_process_ != 1) {
    LOG(FATAL) << "NVSHMEM API is only supported with one device per process";
  }
  CHECK(Initialize().ok());
}

NvshmemApi::~NvshmemApi() {
  VLOG(3) << absl::StreamFormat(
      "Finilizing NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  nvshmemx_hostlib_finalize();
}

absl::Status NvshmemApi::Initialize() {
  nvshmemx_init_attr_t nvshmem_init_attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_uniqueid_t nvshmem_id = NVSHMEMX_UNIQUEID_INITIALIZER;

  // Initialize NVSHMEM
  if (process_id_ == 0) {
    XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_get_uniqueid(&nvshmem_id));
    std::string_view nvshmem_id_str(reinterpret_cast<char*>(&nvshmem_id),
                                    sizeof(nvshmemx_uniqueid_t));
    TF_RETURN_IF_ERROR(kv_store_set_(kv_store_key_, nvshmem_id_str));
  } else {
    TF_ASSIGN_OR_RETURN(std::string id_str, kv_store_get_(kv_store_key_));
    std::copy(id_str.data(), id_str.data() + sizeof(nvshmemx_uniqueid_t),
              reinterpret_cast<char*>(&nvshmem_id));
  }

  XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_set_attr_uniqueid_args(
      process_id_, num_processes_, &nvshmem_id, &nvshmem_init_attr));
  XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_hostlib_init_attr(
      NVSHMEMX_INIT_WITH_UNIQUEID, &nvshmem_init_attr));

  VLOG(3) << absl::StreamFormat(
      "Initialized NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  return absl::OkStatus();
}

absl::StatusOr<void*> NvshmemApi::Allocate(uint64_t bytes) {
  VLOG(3) << absl::StreamFormat(
      "Start allocation of %s (%llu bytes) for NVSHMEM",
      tsl::strings::HumanReadableNumBytes(bytes), bytes);
  void* buffer = nvshmem_malloc(bytes);
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from NVSHMEM memory",
        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  return buffer;
}

absl::Status NvshmemApi::Deallocate(void* buffer) {
  VLOG(3) << absl::StreamFormat("Start de-allocation for NVSHMEM buffer: %p",
                                buffer);
  nvshmem_free(buffer);
  return absl::OkStatus();
}

}  // namespace xla::gpu
