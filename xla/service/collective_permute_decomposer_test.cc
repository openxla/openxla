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

#include "xla/service/collective_permute_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
namespace op = xla::testing::opcode_matchers;
using CollectivePermuteDecomposerTest = HloTestBase;

TEST_F(CollectivePermuteDecomposerTest, WithCycleNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] replica-id()
        ROOT cp = u32[] collective-permute(p), channel_id=1,
          source_target_pairs={{0,1}, {1,0}}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, WithContextDataNotTransformed) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = (u32[], u32[], u32[], u32[]) collective-permute(p), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, TransformedExplicitChannelId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = u32[] collective-permute(p), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
      metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);

  auto check_metadata = [](const HloInstruction* inst) {
    EXPECT_EQ(inst->metadata().op_name(), "op1/op2/add");
    EXPECT_EQ(inst->metadata().source_file(), "foo/bar/mysource.py");
    EXPECT_EQ(inst->metadata().source_line(), 35);
  };

  auto check_not_pipelined = [](const HloInstruction* instr) {
    const FrontendAttributes& attributes = instr->frontend_attributes();
    EXPECT_EQ(attributes.map().end(),
              attributes.map().find(kSendRecvPipelineAttr));
  };

  HloInstruction* after_all = FindInstruction(module.get(), "after-all");
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->operand(0), after_all);
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
  check_metadata(recv);
  check_not_pipelined(recv);
  HloInstruction* recv_done = FindInstruction(module.get(), "recv-done");
  EXPECT_EQ(recv_done->operand(0), recv);

  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_EQ(send->operand(1), after_all);
  EXPECT_EQ(send->channel_id().value(), 1);
  EXPECT_THAT(
      send->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
  check_metadata(send);
  check_not_pipelined(send);
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  EXPECT_EQ(send_done->operand(0), send);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(recv_done, 0));
}

TEST_F(CollectivePermuteDecomposerTest, NotTransformedDefaultChannelId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = u32[] collective-permute(p),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, ThresholdNotTransformed) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = u32[] collective-permute(p), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
      metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, Pipeline1) {
  const char* const kModuleStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    recv-data = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
      frontend_attributes={_xla_other_attribute="xyz"}

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_other_attribute=\"xyz\""));
  HloInstruction* recv_done = FindInstruction(module.get(), "recv-done");
  EXPECT_THAT(recv_done->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"0\""));

  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_EQ(send->channel_id().value(), 1);
  EXPECT_THAT(
      send->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_other_attribute=\"xyz\""));
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  EXPECT_THAT(send_done->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"0\""));

  EXPECT_FALSE(recv_done->control_predecessors().empty());
  EXPECT_EQ(recv_done->control_predecessors()[0], send);
}

TEST_F(CollectivePermuteDecomposerTest, ForwardPipeline2) {
  const char* const kModuleStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    recv-data.0 = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{3,0}}

    recv-data.1 = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,1}, {1,2}, {2,3}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(recv->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs={{3,0}}"));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_THAT(send->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs={{3,0}}"));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));

  HloInstruction* recv1 = FindInstruction(module.get(), "recv.1");
  EXPECT_EQ(recv1->channel_id().value(), 2);
  EXPECT_THAT(
      recv1->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}"));
  EXPECT_THAT(recv1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* recv_done1 = FindInstruction(module.get(), "recv-done.1");
  EXPECT_THAT(recv_done1->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* send1 = FindInstruction(module.get(), "send.1");
  EXPECT_THAT(
      send1->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}"));
  EXPECT_THAT(send1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* send_done1 = FindInstruction(module.get(), "send-done.1");
  EXPECT_THAT(send_done1->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"1\""));
}

TEST_F(CollectivePermuteDecomposerTest, ForwardPipelineWithMatmul) {
  // The HLO module below is generated by passing the HLO in
  // CollectiveOpsTest.CollectivePermute_CircularPipelinePreOptimization through
  // the collective_permute_cycle_decomposer.transformation.
  const char* const kModuleStr = R"(
  HloModule test

  while_body {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    partition-id = u32[] partition-id()
    zero = u32[] constant(0)
    compare = pred[] compare(partition-id, zero), direction=EQ
    broadcast = pred[2,2] broadcast(compare), dimensions={}

    weights = f32[2,2] get-tuple-element(inputs), index=2
    data = f32[2,2] get-tuple-element(inputs), index=1

    cp_back = f32[2,2] collective-permute(data), channel_id=1,
      source_target_pairs={{3,0}},
      frontend_attributes={_xla_send_recv_validation="{{3,10}}"}
    cp_forward = f32[2,2] collective-permute(data), channel_id=2,
      source_target_pairs={{0,1},{1,2},{2,3}},
      frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9}}"}

    select = f32[2,2] select(broadcast, cp_back, cp_forward)

    matmul = f32[2,2] dot(weights, select), lhs_contracting_dims={1}, rhs_contracting_dims={0}

    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)
  }

  while_cond {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    max_iter = u32[] constant(3)
    ROOT compare = pred[] compare(iter, max_iter), direction=LT
  }

  ENTRY test_computation {
    start_iter = u32[] constant(0)
    input_data = f32[2,2] parameter(0)
    input_weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data, input_weights)
    while_result = (u32[], f32[2,2], f32[2,2]) while(input), condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_result), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloModule* transformed_module = module.get();
  // Check the annotations and ordering of the decomposed send-recv pairs.
  // We expect the recv to come before the send in the while body, both for the
  // forward edge ({0,1},{1,2},{2,3}}) and the backward edge ({3,0}). This is
  // an XLA invariant that shouldn't be broken (see
  // https://openxla.org/xla/operation_semantics#send for details of the
  // semantics).
  HloInstruction* recv_bwd = FindInstruction(transformed_module, "recv");
  EXPECT_EQ(recv_bwd->channel_id().value(), 1);
  auto recv_bwd_frontend_attributes = recv_bwd->frontend_attributes().map();
  EXPECT_EQ(recv_bwd_frontend_attributes.size(), 3);
  EXPECT_EQ(recv_bwd_frontend_attributes.at(kSendRecvValidationAttr),
            "{{3,10}}");
  EXPECT_EQ(recv_bwd_frontend_attributes.at(kSendRecvPipelineAttr), "0");
  EXPECT_EQ(recv_bwd_frontend_attributes.at(kSendRecvSourceTargetPairsAttr),
            "{{3,0}}");

  HloInstruction* send_bwd = FindInstruction(transformed_module, "send");
  auto send_bwd_frontend_attributes = send_bwd->frontend_attributes().map();
  EXPECT_THAT(send_bwd_frontend_attributes.at(kSendRecvSourceTargetPairsAttr),
              "{{3,0}}");

  HloInstruction* recv_fwd = FindInstruction(transformed_module, "recv.1");
  EXPECT_EQ(recv_fwd->channel_id().value(), 2);
  auto recv_fwd_frontend_attributes = recv_fwd->frontend_attributes().map();
  EXPECT_EQ(recv_fwd_frontend_attributes.size(), 3);
  EXPECT_EQ(recv_fwd_frontend_attributes.at(kSendRecvPipelineAttr), "1");
  EXPECT_EQ(recv_fwd_frontend_attributes.at(kSendRecvSourceTargetPairsAttr),
            "{{0,1},{1,2},{2,3}}");

  HloInstruction* send_fwd = FindInstruction(transformed_module, "send.1");
  auto send_fwd_frontend_attributes = send_fwd->frontend_attributes().map();
  EXPECT_EQ(send_fwd_frontend_attributes.size(), 3);
  EXPECT_EQ(send_fwd_frontend_attributes.at(kSendRecvPipelineAttr), "1");
  EXPECT_EQ(send_fwd_frontend_attributes.at(kSendRecvSourceTargetPairsAttr),
            "{{0,1},{1,2},{2,3}}");

  HloComputation* while_body =
      FindComputation(transformed_module, "while_body");
  EXPECT_NE(while_body, nullptr);
  EXPECT_TRUE(hlo_query::IsBeforeInComputation(while_body, "recv", "send"));
  EXPECT_TRUE(
      hlo_query::IsBeforeInComputation(while_body, "recv", "recv-done"));
  EXPECT_TRUE(
      hlo_query::IsBeforeInComputation(while_body, "send", "recv-done"));
  EXPECT_TRUE(
      hlo_query::IsBeforeInComputation(while_body, "send", "send-done"));
  EXPECT_TRUE(
      hlo_query::IsBeforeInComputation(while_body, "send-done", "send-done.1"));
  EXPECT_TRUE(
      hlo_query::IsBeforeInComputation(while_body, "recv-done", "send-done.1"));
  EXPECT_TRUE(hlo_query::IsBeforeInComputation(while_body, "recv-done.1",
                                               "send-done.1"));
  auto recv_done_fwd = FindInstruction(transformed_module, "recv-done");
  auto recv_done_bwd = FindInstruction(transformed_module, "recv-done.1");

  // TODO: b/356201477 - Investigate potential NCCL deadlock in
  // collective_permute_decomposer
  EXPECT_EQ(recv_done_fwd->control_predecessors()[0], send_bwd);
  EXPECT_EQ(recv_done_bwd->control_predecessors()[0], send_fwd);
}

TEST_F(CollectivePermuteDecomposerTest, BackwardPipeline2) {
  const char* const kModuleStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    recv-data.0 = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{1,0},{2,1},{3,2}}

    recv-data.1 = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,3}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=NE
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs={{1,0},{2,1},{3,2}}"));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_THAT(
      send->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs={{1,0},{2,1},{3,2}}"));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));

  HloInstruction* recv1 = FindInstruction(module.get(), "recv.1");
  EXPECT_EQ(recv1->channel_id().value(), 2);
  EXPECT_THAT(recv1->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs={{0,3}}"));
  EXPECT_THAT(recv1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  HloInstruction* send1 = FindInstruction(module.get(), "send.1");
  EXPECT_THAT(send1->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs={{0,3}}"));
  EXPECT_THAT(send1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
}

}  // namespace
}  // namespace xla
