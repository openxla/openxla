/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_MHLO_TRANSFORMS_FOLDERS_FOLDERS_H_
#define MLIR_HLO_MHLO_TRANSFORMS_FOLDERS_FOLDERS_H_

#include <cstdint>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace mhlo {

template <typename SourceOp>
class OpFolderRewritePattern : public OpRewritePattern<SourceOp> {
 public:
  explicit OpFolderRewritePattern(MLIRContext *context,
                                  int64_t opFoldLimit = 65536)
      : OpRewritePattern<SourceOp>(context), opFoldLimit(opFoldLimit) {}

  int64_t opFoldLimit;
};

class ConvertOpFolder : public OpFolderRewritePattern<ConvertOp> {
 public:
  using OpFolderRewritePattern::OpFolderRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp convertOp,
                                PatternRewriter &rewriter) const override;
};

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_FOLDERS_FOLDERS_H_
