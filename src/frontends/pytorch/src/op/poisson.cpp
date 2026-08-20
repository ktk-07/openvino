// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/random_poisson.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov;
using namespace ov::op;

OutputVector translate_poisson(const NodeContext& context) {
    // Supported signatures:
    // 1. aten::poisson(Tensor input, Generator? generator=None) -> Tensor
    num_inputs_check(context, 1, 2);
    auto input = context.get_input(0);

    // Generator not supported as no torch.Generator in frontend. 
    // Similar approach when translating aten::multinomial.
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1),
                                "aten::poisson conversion with generator is not supported");

    uint64_t global_seed = 0;
    uint64_t op_seed = 0;
    // When both seeds are 0, backend/reference may use non-deterministic RNG (e.g. time-based).
    // Alignment PYTORCH matches PyTorch's Philox usage (same as aten::bernoulli and aten::multinomial).
    return {context.mark_node(std::make_shared<ov::op::v17::RandomPoisson>(input,
                                                                            global_seed,
                                                                            op_seed,
                                                                            ov::op::PhiloxAlignment::PYTORCH))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov