"""CPU dispatch checks for the opt-in MACA BF16 logits collective."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_processor_method(name):
    path = ROOT / "python/sglang/srt/layers/logits_processor.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LogitsProcessor")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace[name]


checks = SimpleNamespace(ROOT=ROOT, load_processor_method=load_processor_method,
                         copy_to_buffer=load_processor_method("_copy_logits_to_buffer"))


class BF16GatherTests(unittest.TestCase):
    def run_path(self, enabled, dtype=torch.bfloat16, capture=True, decode=True,
                 rows=3, specialized=None):
        get_logits = checks.load_processor_method("_get_logits")
        logits = torch.arange(rows * 5, dtype=dtype).view(rows, 5)
        output = torch.empty(rows, 20, dtype=torch.float32)
        metadata = SimpleNamespace(
            next_token_logits_buffer=output,
            forward_mode=SimpleNamespace(is_decode=lambda: decode, is_idle=lambda: False))
        seen = []

        def collective(value, *args):
            seen.append(value.dtype)
            return torch.cat([value] * 4, dim=1)

        processor = SimpleNamespace(
            vocab_size=20, use_bf16_logits_gather=enabled,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
            use_lm_head_tp_group=False, use_attn_tp_group=False,
            use_tp_lm_head_all_to_all=False,
            logit_scale=None, final_logit_softcapping=None,
            _gather_dp_attn_hidden_states=lambda x, meta: (x, x),
            _compute_lm_head=lambda *args: logits,
            _can_use_tp_lm_head_all_to_all=lambda *args: False,
            _gather_attn_tp_logits=collective,
            _logits_gatherer=collective,
            _scatter_dp_attn_logits=lambda value, *args: value)
        processor._copy_logits_to_buffer = lambda *args, **kw: checks.copy_to_buffer(
            processor, *args, **kw)
        if specialized:
            setattr(processor, specialized, True)
        get_logits.__globals__.update(
            _trace_e2e_logits=lambda *args, **kw: None,
            envs=SimpleNamespace(SGLANG_TRACE_LOGITS_E2E_SYNC=SimpleNamespace(get=lambda: False)),
            get_lm_head_tp_group=lambda: SimpleNamespace(all_gather=collective))
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=capture):
            result = get_logits(processor, torch.empty(rows, 7), None, metadata)
        self.assertEqual(result.dtype, torch.float32)
        self.assertIs(result, output)
        torch.testing.assert_close(result, torch.cat([logits.float()] * 4, dim=1),
                                   rtol=0, atol=0)
        return seen[0]

    def test_default_preserves_fp32_graph_collective(self):
        self.assertEqual(self.run_path(False), torch.float32)

    def test_opt_in_preserves_bf16_graph_collective(self):
        self.assertEqual(self.run_path(True), torch.bfloat16)

    def test_eager_and_empty_keep_existing_dtype(self):
        for enabled in (False, True):
            self.assertEqual(self.run_path(enabled, capture=False), torch.bfloat16)
            self.assertEqual(self.run_path(enabled, rows=0), torch.bfloat16)

    def test_other_paths_keep_fp32_workaround(self):
        self.assertEqual(self.run_path(True, dtype=torch.float16), torch.float32)
        self.assertEqual(self.run_path(True, dtype=torch.float32), torch.float32)
        self.assertEqual(self.run_path(True, decode=False), torch.float32)
        for name in ("use_lm_head_tp_group", "use_attn_tp_group",
                     "use_tp_lm_head_all_to_all", "do_tensor_parallel_all_gather_dp_attn"):
            self.assertEqual(self.run_path(True, specialized=name), torch.float32)

    def test_switch_defaults_off_and_is_maca_only(self):
        path = checks.ROOT / "python/sglang/srt/layers/logits_processor.py"
        tree = ast.parse(path.read_text())
        assignment = next(
            n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "use_bf16_logits_gather"
                    for t in n.targets))
        expression = compile(ast.Expression(body=assignment.value), str(path), "eval")
        for maca in (False, True):
            for enabled in (False, True):
                env = SimpleNamespace(SGLANG_LOGITS_GATHER_BF16=SimpleNamespace(get=lambda: enabled))
                self.assertEqual(eval(expression, {"is_maca": lambda: maca, "envs": env}),
                                 maca and enabled)
        path = checks.ROOT / "python/sglang/srt/environ.py"
        tree = ast.parse(path.read_text())
        assignment = next(
            n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SGLANG_LOGITS_GATHER_BF16"
                    for t in n.targets))
        self.assertEqual(ast.unparse(assignment.value), "EnvBool(False)")


if __name__ == "__main__":
    unittest.main()
