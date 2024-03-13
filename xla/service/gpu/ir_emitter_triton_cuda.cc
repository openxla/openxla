/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_module_config.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h" // from @llvm-project
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"

namespace xla {
namespace gpu {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace ml = ::mlir::LLVM;
namespace mn = ::mlir::NVVM;
namespace mt = ::mlir::triton;

using ::llvm::SmallVector;
using mlir::ArrayRef;
using mlir::ImplicitLocOpBuilder;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;
using mlir::ValueRange;

absl::Status CreateTritonPipeline(
    mlir::OpPassManager& pm, const se::GpuComputeCapability& cc,
    const TritonGemmConfig& config,
    mt::nvidia_gpu::ClusterInfo& out_cluster_info) {
  auto ccCuda = std::get<se::CudaComputeCapability>(cc);
  const int ccAsInt = ccCuda.major * 10 + ccCuda.minor;
  const int threadsPerWarp = 32;

  // Based on make_ttir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mt::createRewriteTensorPointerPass());
  pm.addPass(mt::createCombineOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mt::createReorderBroadcastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // Based on make_ttgir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm.addPass(mt::createConvertTritonToTritonGPUPass(
      config.num_warps, threadsPerWarp, config.num_ctas, ccAsInt));
  pm.addPass(mt::gpu::createCoalescePass());
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&out_cluster_info));
  pm.addPass(mt::gpu::createRemoveLayoutConversionsPass());
  pm.addPass(mt::gpu::createOptimizeThreadLocalityPass());
  pm.addPass(mt::gpu::createAccelerateMatmulPass(ccAsInt));
  pm.addPass(mt::gpu::createRemoveLayoutConversionsPass());
  pm.addPass(mt::gpu::createOptimizeDotOperandsPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mt::gpu::createPipelinePass(config.num_stages, config.num_warps,
                                         config.num_ctas, ccAsInt));

  if (!ccCuda.IsAtLeastHopper()) {
    pm.addPass(mt::gpu::createPrefetchPass());
  }

  pm.addPass(mt::gpu::createOptimizeDotOperandsPass());
  pm.addPass(mt::gpu::createRemoveLayoutConversionsPass());
  pm.addPass(mt::gpu::createReduceDataDuplicationPass());
  pm.addPass(mt::gpu::createReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (ccCuda.IsAtLeastHopper()) {
    pm.addPass(mlir::createTritonNvidiaGPUFenceInsertionPass(ccAsInt));
  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Based on make_llir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm.addPass(mt::gpu::createDecomposeUnsupportedConversionsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mt::gpu::createAllocateSharedMemoryPass());
  pm.addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
  pm.addPass(mt::createConvertNVGPUToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.

  return absl::OkStatus();
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  return nvptx::LibDevicePath(
      hlo_config.debug_options().xla_gpu_cuda_data_dir());
}

}  // namespace gpu
}  // namespace xla
