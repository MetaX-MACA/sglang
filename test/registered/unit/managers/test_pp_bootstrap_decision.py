import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBootstrapDecision,
    SchedulerPPMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_req(rid, finished_reason=None):
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=1,
        finished_reason=finished_reason,
    )


class TestPPBootstrapDecision(unittest.TestCase):
    def make_scheduler(self, *, pp_rank, reqs=()):
        scheduler = SchedulerPPMixin()
        scheduler.ps = SimpleNamespace(pp_size=8)
        scheduler.pp_group = SimpleNamespace(is_first_rank=pp_rank == 0)
        scheduler.waiting_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=list(reqs),
            pop_bootstrapped=MagicMock(return_value=([], [])),
            ensure_metadata_buffer=MagicMock(return_value=True),
        )
        scheduler.abort_request = MagicMock()
        return scheduler

    def test_pp0_freezes_candidate_and_fences_pending_failure(self):
        scheduler = self.make_scheduler(pp_rank=0, reqs=[make_req("good")])
        scheduler._pp_pending_bootstrap_failures = {"bad": "local failure"}

        decision = scheduler.process_bootstrapped_queue([["good", "bad"], []])

        self.assertEqual(
            decision,
            PPBootstrapDecision(0, ("good",), ("bad",)),
        )
        scheduler.disagg_prefill_bootstrap_queue.pop_bootstrapped.assert_called_once_with(
            return_failed_reqs=True,
            pp_good_rids=("good",),
            pp_bad_rids=("bad",),
        )

    def test_downstream_forwards_exact_immutable_decision(self):
        scheduler = self.make_scheduler(pp_rank=3)
        decision = PPBootstrapDecision(0, ("good",), ())

        forwarded = scheduler.process_bootstrapped_queue(decision)

        self.assertIs(forwarded, decision)
        self.assertEqual(scheduler._pp_bootstrap_expected_decision_sequence, 1)

    def test_duplicate_is_ignored_and_sequence_gap_fails(self):
        scheduler = self.make_scheduler(pp_rank=4)
        decision = PPBootstrapDecision(0, (), ())

        scheduler.process_bootstrapped_queue(decision)
        scheduler.process_bootstrapped_queue(decision)
        with self.assertRaisesRegex(RuntimeError, "sequence gap"):
            scheduler.process_bootstrapped_queue(PPBootstrapDecision(2, (), ()))

    def test_downstream_defers_abort_until_its_boundary(self):
        scheduler = self.make_scheduler(pp_rank=2)
        abort_req = AbortReq(rid="request", pp_bootstrap_abort_after_sequence=0)
        scheduler.abort_request = MagicMock()

        self.assertTrue(scheduler._pp_order_or_defer_abort_request(abort_req))
        scheduler.process_bootstrapped_queue(PPBootstrapDecision(0, (), ()))

        scheduler.abort_request.assert_called_once_with(abort_req)

    def test_metadata_unready_candidate_is_withheld(self):
        req = make_req("ready")
        scheduler = self.make_scheduler(pp_rank=0, reqs=[req])
        scheduler.disagg_prefill_bootstrap_queue.ensure_metadata_buffer.return_value = False

        self.assertEqual(scheduler._pp_filter_metadata_ready_rids(["ready"]), [])


if __name__ == "__main__":
    unittest.main()
