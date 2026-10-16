import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.entrypoints.cli import generate as cli


class TestGenerateCommandShutdown(unittest.TestCase):
    def _run_generate(self, generator):
        args = SimpleNamespace(config=None, output_file_path=None)
        server_args = SimpleNamespace(model_path="test-model")
        sampling_cls = Mock()
        sampling_cls.get_cli_args.return_value = {"prompt": "a cat"}
        with (
            patch.object(cli.ServerArgs, "from_cli_args", return_value=server_args),
            patch.object(
                cli, "_resolve_cli_sampling_params_cls", return_value=sampling_cls
            ),
            patch.object(cli.DiffGenerator, "from_pretrained", return_value=generator),
            patch.object(cli, "maybe_dump_performance"),
        ):
            cli.generate_cmd(args)

    def test_shutdown_after_successful_generation(self):
        generator = Mock()

        self._run_generate(generator)

        generator.generate.assert_called_once()
        generator.shutdown.assert_called_once_with()

    def test_shutdown_when_generation_raises(self):
        generator = Mock()
        generator.generate.side_effect = RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self._run_generate(generator)

        generator.shutdown.assert_called_once_with()

    def test_empty_generation_fails_the_command_and_shuts_down(self):
        generator = Mock()
        generator.generate.return_value = None

        with self.assertRaisesRegex(RuntimeError, "Generation failed"):
            self._run_generate(generator)

        generator.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
