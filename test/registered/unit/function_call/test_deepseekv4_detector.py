"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must not empty the turn, and the dropped
        buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        # No half-formed call: the failure can land between a tool's name and its
        # arguments, so an argument-less named call must not reach the client.
        self.assertEqual(first.calls, [])

    def test_stream_end_flushes_held_back_dsml_prefix(self):
        """A trailing DSML prefix is held back while a tool call may still arrive;
        the stream ending must release it as normal text."""
        detector = DeepSeekV4Detector()
        self.assertEqual(
            detector.parse_streaming_increment("plain text<｜DSML", self.tools).normal_text,
            "",
        )
        end = detector.finish(self.tools)
        self.assertEqual(end.normal_text, "plain text<｜DSML")
        self.assertEqual(end.calls, [])
        # State is cleared, so a second finish() is a no-op.
        self.assertEqual(detector.finish(self.tools).normal_text, "")

    def test_stream_end_does_not_leak_completed_call_markup(self):
        """A completed call leaves the closing `</｜DSML｜tool_calls>` in the buffer;
        the stream ending must not surface that raw markup as content."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            _weather_call() + f"\n</{DSML}tool_calls>", self.tools
        )
        end = detector.finish(self.tools)

        self.assertEqual(end.normal_text, "")
        self.assertNotIn(DSML, result.normal_text + end.normal_text)

    def test_stream_end_keeps_prose_after_tool_call(self):
        """Prose batched after the closing marker is still content, minus markup."""
        detector = DeepSeekV4Detector()
        detector.parse_streaming_increment(
            _weather_call() + f"\n</{DSML}tool_calls>\nI will check now", self.tools
        )
        end = detector.finish(self.tools)

        self.assertNotIn(DSML, end.normal_text)
        self.assertTrue(end.normal_text.endswith("I will check now"), end.normal_text)

    def test_stream_end_drops_truncated_call_block(self):
        """A call cut off mid-block must not leak its tags or half-formed args."""
        detector = DeepSeekV4Detector()
        detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="true">{DSML}',
            self.tools,
        )
        end = detector.finish(self.tools)

        self.assertNotIn(DSML, end.normal_text)


if __name__ == "__main__":
    import unittest

    unittest.main()
