/* Copyright 2021 The OpenXLA Authors.

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

#include <string>
#include <string_view>

#include "mhlo/transforms/passes.h"
#include "absl/status/statusor.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "nanobind/nanobind.h"  // from @nanobind
#include "nanobind/stl/string.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // from @nanobind  // IWYU pragma: keep
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/client/xla_computation.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/refine_polymorphic_shapes.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tsl/platform/errors.h"

namespace nb = nanobind;

namespace xla {
namespace {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseModule(
    mlir::MLIRContext* context, std::string_view str) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::mhlo::MhloDialect>();
  context->loadDialect<mlir::chlo::ChloDialect>();
  context->loadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  context->loadDialect<mlir::stablehlo::StablehloDialect>();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  context->appendDialectRegistry(registry);

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(str.data(), str.size()), context);
  if (!module) {
    return diagnostic_handler.ConsumeStatus();
  }
  if (failed(module->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module->dump();
    return diagnostic_handler.ConsumeStatus();
  }
  return module;
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(os, flags);
  return s;
}

void EnablePrintBeforeAndAfter(mlir::PassManager& pm) {
  auto print_before = [](mlir::Pass*, mlir::Operation*) { return true; };
  auto print_after = [](mlir::Pass*, mlir::Operation*) { return true; };
  pm.enableIRPrinting(print_before, print_after);
}

// Converts an XlaComputation to an MHLO or StableHLO mlir::Module string.
// Exists for backwards compatibility.
// TODO(phawkins): port remaining users of XlaComputations to use mlir::Modules
// instead and delete this function.
absl::StatusOr<std::string> PyXlaComputationToMlirModule(
    const XlaComputation& computation, bool emit_stable_hlo) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&context));
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  context.appendDialectRegistry(registry);

  TF_RETURN_IF_ERROR(ConvertHloToMlirHlo(*module, &computation.proto(),
                                         /*import_all_computations=*/true));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  if (emit_stable_hlo) {
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  }
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("MHLO => StableHLO failed");
  }
  return PrintModule(*module);
}

absl::StatusOr<XlaComputation> PyMlirModuleToXlaComputation(
    std::string_view mlir_module, bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseModule(&context, mlir_module));
  XlaComputation computation;
  TF_RETURN_IF_ERROR(
      MlirToXlaComputation(*module, computation, use_tuple_args, return_tuple));
  return computation;
}

absl::StatusOr<std::string> PyMhloToStablehlo(std::string_view mlir_module) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  // JAX can be customized in a way that involves operations from custom
  // dialects showing up in JAX IR.
  // `ParseModule` won't know about these dialects, but that's fine since we
  // just want to convert MHLO ops to StableHLO ops here and leave everything
  // else unchanged.
  // In order to achieve that, we're allowing unregistered dialects here.
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseModule(&context, mlir_module));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("MHLO => StableHLO failed");
  }
  return PrintModule(*module);
}

absl::StatusOr<std::string> PyStablehloToMhlo(const nb::bytes& mlir_module) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  // See PyMhloToStablehlo for an explanation of why we're allowing unregistered
  // dialects here.
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      ParseModule(&context,
                  std::string_view(mlir_module.c_str(), mlir_module.size())));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("StableHLO => MHLO failed");
  }
  return PrintModule(*module);
}

absl::StatusOr<nb::bytes> PySerializePortableArtifact(
    std::string_view mlir_module, std::string_view target) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseModule(&context, mlir_module));

  // Serialize portable artifact
  TF_ASSIGN_OR_RETURN(
      std::string bytecode,
      SerializeUsingVersionedStablehlo(*module, target, /*inplace=*/true));
  return nb::bytes(bytecode.data(), bytecode.size());
}

absl::StatusOr<std::string> PyDeserializePortableArtifact(
    const nb::bytes& bytecode_str) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::stablehlo::deserializePortableArtifact(
          std::string_view(bytecode_str.c_str(), bytecode_str.size()),
          &context);
  if (!module)
    return tsl::errors::InvalidArgument("Failed to deserialize StableHLO");
  return PrintModule(*module);
}

}  // namespace

void BuildMlirSubmodule(nb::module_& m) {
  nb::module_ mlir_module = m.def_submodule("mlir", "MLIR/XLA integration");

  mlir_module.def("xla_computation_to_mlir_module",
                  xla::ValueOrThrowWrapper(PyXlaComputationToMlirModule),
                  nb::arg("computation"), nb::arg("emit_stable_hlo") = true);
  mlir_module.def("mlir_module_to_xla_computation",
                  xla::ValueOrThrowWrapper(PyMlirModuleToXlaComputation),
                  nb::arg("mlir_module"), nb::arg("use_tuple_args") = false,
                  nb::arg("return_tuple") = false);
  mlir_module.def(
      "mhlo_to_stablehlo",
      [](const nb::bytes& mlir_module) {
        return xla::ValueOrThrow(PyMhloToStablehlo(
            std::string_view(mlir_module.c_str(), mlir_module.size())));
      },
      nb::arg("mlir_module"));
  mlir_module.def(
      "mhlo_to_stablehlo",
      [](std::string_view mlir_module) {
        return xla::ValueOrThrow(PyMhloToStablehlo(mlir_module));
      },
      nb::arg("mlir_module"));
  mlir_module.def("stablehlo_to_mhlo",
                  xla::ValueOrThrowWrapper(PyStablehloToMhlo),
                  nb::arg("mlir_module"));
  mlir_module.def(
      "serialize_portable_artifact",
      [](const nb::bytes& mlir_module, std::string_view target) {
        return xla::ValueOrThrow(PySerializePortableArtifact(
            std::string_view(mlir_module.c_str(), mlir_module.size()), target));
      },
      nb::arg("mlir_module"), nb::arg("target"));
  mlir_module.def("serialize_portable_artifact",
                  xla::ValueOrThrowWrapper(PySerializePortableArtifact),
                  nb::arg("mlir_module"), nb::arg("target"));
  mlir_module.def("deserialize_portable_artifact",
                  xla::ValueOrThrowWrapper(PyDeserializePortableArtifact),
                  nb::arg("mlir_module"));
  mlir_module.def(
      "refine_polymorphic_shapes",
      [](nb::bytes mlir_module, bool enable_shape_assertions,
         bool validate_static_shapes) -> nb::bytes {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        xla::ThrowIfError(RefinePolymorphicShapes(
            std::string_view(mlir_module.c_str(), mlir_module.size()), os,
            enable_shape_assertions, validate_static_shapes));
        return nb::bytes(buffer.data(), buffer.size());
      },
      nb::arg("mlir_module"), nb::arg("enable_shape_assertions") = true,
      nb::arg("validate_static_shapes") = true,
      R"(Refines the dynamic shapes for a module.
        The "main" function must have static shapes and all the
        intermediate dynamic shapes depend only on the input static
        shapes. Optionally, also validates that the resulting module has
        only static shapes.
      )");
}

}  // namespace xla
