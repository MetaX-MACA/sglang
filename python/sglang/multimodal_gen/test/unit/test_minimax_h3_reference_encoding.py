# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for MiniMax H3 reference-image preparation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestMiniMaxH3ReferenceImageResize(unittest.TestCase):
    def test_shape_uses_fixed_2048_short_edge_nearest_32_and_upscales(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_resolve_reference_image_shape,
        )

        shape = minimax_h3_resolve_reference_image_shape(width=320, height=240)

        self.assertEqual((shape["width"], shape["height"]), (2720, 2048))
        self.assertEqual(shape["base_short_edge"], 2048)
        self.assertEqual(shape["multiple"], 32)
        self.assertEqual(shape["rounding"], "nearest")
        self.assertTrue(shape["allow_upscale"])

    def test_shape_rejects_non_finite_geometry(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_resolve_reference_image_shape,
        )

        for width, height in ((float("nan"), 1), (1, float("inf")), (0, 1)):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    minimax_h3_resolve_reference_image_shape(
                        width=width,
                        height=height,
                    )

    def test_shape_rejects_ratios_outside_one_to_four(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_resolve_reference_image_shape,
        )

        for width, height in ((5, 1), (1, 5), (401, 100), (100, 401)):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "inclusive range 1:4 to 4:1"):
                    minimax_h3_resolve_reference_image_shape(
                        width=width,
                        height=height,
                    )

    def test_prepared_image_consumes_precomputed_material_shape(self):
        from PIL import Image

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
            material_io,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_prepared_reference_image,
            minimax_h3_resolve_reference_image_shape,
        )

        resolved_shape = minimax_h3_resolve_reference_image_shape(width=320, height=240)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "small.png"
            Image.new("RGB", (320, 240), "red").save(source)
            condition_index = 3
            batch = SimpleNamespace(
                extra={
                    MINIMAX_H3_PROBE_FACTS_EXTRA_KEY: {
                        condition_index: {
                            "display_width": 320,
                            "display_height": 240,
                        }
                    },
                    MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY: {
                        condition_index: resolved_shape
                    },
                }
            )
            plan = SimpleNamespace(
                materials=[
                    SimpleNamespace(
                        material_chain="image.reference_preserve",
                        condition_index=condition_index,
                        condition_type="image",
                        uri="file:///ignored.png",
                    )
                ]
            )
            with patch.object(
                material_io,
                "minimax_h3_localize_material_uri",
                return_value=str(source),
            ):
                prepared = minimax_h3_prepared_reference_image(batch, plan)

        self.assertEqual(prepared["image"].size, (2720, 2048))
        self.assertEqual(prepared["condition_index"], condition_index)


class TestScopedEncodeRng(unittest.TestCase):
    """The encode recipes must not leak seed-42 state into the global RNG."""

    def test_scoped_encode_rng_restores_global_state(self):
        import torch

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
            minimax_h3_scoped_encode_rng,
        )

        torch.default_generator.manual_seed(1234)
        expected_after = torch.randn(4)  # what an outer consumer would draw
        torch.default_generator.manual_seed(1234)

        with minimax_h3_scoped_encode_rng(42):
            inner_first = torch.randn(4)
        with minimax_h3_scoped_encode_rng(42):
            inner_second = torch.randn(4)

        # Inside the scope the recipe seed is deterministic (deterministic recipe).
        self.assertTrue(torch.equal(inner_first, inner_second))
        # Outside the scope the pre-existing generator state is untouched.
        self.assertTrue(torch.equal(torch.randn(4), expected_after))

    def test_scoped_encode_rng_matches_global_seeding(self):
        import torch

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
            minimax_h3_scoped_encode_rng,
        )

        torch.manual_seed(42)  # the recipe's global seeding
        reference_draw = torch.randn(8)

        with minimax_h3_scoped_encode_rng(42):
            scoped_draw = torch.randn(8)

        self.assertTrue(torch.equal(scoped_draw, reference_draw))


if __name__ == "__main__":
    unittest.main()
