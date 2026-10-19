import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.vaes import wanvae


class TestWanVaeMacaLayout(unittest.TestCase):
    def test_maca_convolution_preserves_values_in_dense_ncdhw(self):
        weight = torch.randn(4, 3, 1, 1, 1)
        channels_last = torch.randn(1, 3, 2, 5, 7).contiguous(
            memory_format=torch.channels_last_3d
        )
        temporal_view = torch.randn(1, 2, 3, 5, 7).permute(0, 2, 1, 3, 4)

        with patch.object(wanvae, "_IS_MACA", True):
            for inputs in [channels_last, temporal_view]:
                with self.subTest(stride=inputs.stride()):
                    output = wanvae.match_conv3d_input_format(inputs, weight)
                    self.assertTrue(output.is_contiguous())
                    self.assertTrue(torch.equal(output, inputs))

    def test_maca_does_not_treat_pointwise_weights_as_channels_last(self):
        weight = torch.randn(4, 3, 1, 1, 1)
        self.assertTrue(weight.is_contiguous(memory_format=torch.channels_last_3d))

        with (
            patch.object(wanvae, "_IS_MACA", True),
            patch.object(wanvae.current_platform, "is_cuda", return_value=True),
        ):
            self.assertFalse(wanvae._conv3d_weight_is_channels_last_3d(weight))


if __name__ == "__main__":
    unittest.main()
