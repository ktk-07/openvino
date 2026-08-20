# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import pytest
from pytorch_layer_test_class import PytorchLayerTest

# Supported signatures:
# 1. aten::poisson(Tensor input, Generator? generator=None) -> Tensor

# Since poisson is a random operation, we need to test it statistically similar to 
# openvino/tests/layer_tests/pytorch_tests/test_rand.py
# We need to test the mean and variance of the output distribution.
# We need to test the output distribution with different input distributions.
# We need to test the output distribution with different input shapes.
# We need to test the output distribution with different input types.
# We need to test the output distribution with different input values.
# We need to test the output distribution with different input ranges.
# We need to test the output distribution with different input sizes.
# We need to test the output distribution with different input densities.

class PoissonModel(nn.Module):
    def __init__(self):
        super(PoissonModel, self).__init__()
        self.poisson = torch.poisson

    def forward(self, input):
        x = input.to(torch.float32)
        return self.poisson(x)

class TestPoisson(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.abs(np.random.randn(*self.input_shape)),)

    def create_model(self):
        return PoissonModel(), "aten::poisson"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [
        [3, 3],
        [2, 5, 10],
        [7, 7, 7],
        [1, 2, 3, 4],
        [1, 7, 69, 69],
    ])
    def test_poisson(self, input_shape, ie_device, precision, ir_version):
        if precision == "FP16":
            pytest.skip("Random op outputs cannot be compared across frameworks with FP16 tolerance due to framework hardcoding custom eps to 5e-2")
        self.input_shape = input_shape
        self._test(*self.create_model(), ie_device, precision, ir_version, custom_eps=1e30)

# Statistics tests
# We need to test the mean and variance of the output distribution.
# Mean and variance should be close to the input rates for Poisson distribution.

class TestPoissonStatistic():
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("model,inputs", [
        (PoissonModel(), (250.0,(1000,1000))), # Testing Hoermann's algorithm
        (PoissonModel(), (50.0,(1000,1000))), # Testing Hoermann's algorithm
        (PoissonModel(), (26.0,(100,100))), # Testing Hoermann's algorithm
        (PoissonModel(), (10.0,(10000,))), # Testing Knuth's algorithm
        (PoissonModel(), (1.0,(1000,1000))), # Testing Knuth's algorithm
        (PoissonModel(), (1.0,(10000,100))), # Testing Knuth's algorithm
        (PoissonModel(), (0.0,(10000,100))), # For zero rates, should return 0
    ])
    def test_poisson_statistics(self, model, inputs, ie_device, precision):
        import numpy.testing as npt
        import numpy as np 
        import openvino as ov
        rates, size = inputs
        lambda_tensor = torch.full(size, rates, dtype=torch.float32)

        example_input = (lambda_tensor,)
        input_size = [size]

        config = {}

        ov_model = ov.convert_model(input_model=model, example_input=example_input, input=input_size)
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)
        pt_res = model(*example_input).numpy()
        ov_res = compiled_model(example_input)[0]
        npt.assert_allclose(ov_res.mean(), pt_res.mean(), atol=0.5, rtol=0.1)
        npt.assert_allclose(ov_res.var(), pt_res.var(), atol=0.5, rtol=0.1)
        npt.assert_allclose(ov_res.mean(), rates, atol=0.5, rtol=0.1)
        npt.assert_allclose(ov_res.var(), rates, atol=0.5, rtol=0.1)