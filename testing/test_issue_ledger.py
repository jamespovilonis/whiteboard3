import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "issue_ledger.py"
SPEC = importlib.util.spec_from_file_location("issue_ledger", SCRIPT_PATH)
issue_ledger = importlib.util.module_from_spec(SPEC)
sys.modules["issue_ledger"] = issue_ledger
SPEC.loader.exec_module(issue_ledger)


class IssueLedgerTest(unittest.TestCase):
    def run_cli(self, ledger_dir, *args):
        return issue_ledger.main(["--ledger-dir", str(ledger_dir), *args])

    def read_issues(self, ledger_dir):
        with (ledger_dir / "issues.json").open() as handle:
            return json.load(handle)

    def test_upsert_transition_fixture_and_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            audit_dir = ledger_dir / "audit_20260703T150000Z_example"
            audit_dir.mkdir()
            comparison_path = audit_dir / "comparison.json"
            comparison_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "fastProblemStatus": "incorrect",
                        "vlmProblemStatus": "correct",
                        "discrepancies": [
                            {"type": "line_latex_mismatch", "description": "Fast dropped equation tail"}
                        ],
                    }
                )
            )

            self.run_cli(
                ledger_dir,
                "upsert",
                "--key",
                "problem-input-fraction-repair-overstrips",
                "--title",
                "Problem-input fraction repair strips equation tails",
                "--severity",
                "P0",
                "--from-audit",
                str(comparison_path),
                "--acceptance-criteria",
                "Preserve equation tails",
            )
            data = self.read_issues(ledger_dir)
            self.assertEqual(data["issues"][0]["id"], "WB3-0001")
            self.assertEqual(data["issues"][0]["severity"], "P0")
            self.assertEqual(data["issues"][0]["evidence"][0]["audit_id"], audit_dir.name)

            self.run_cli(ledger_dir, "transition", "WB3-0001", "fixture_needed")
            self.run_cli(ledger_dir, "add-fixture", "WB3-0001", "testing/fixtures/example.json")
            self.run_cli(
                ledger_dir,
                "add-verification",
                "WB3-0001",
                "node --test testing/test_recognition_pipeline.mjs",
            )
            data = self.read_issues(ledger_dir)
            issue = data["issues"][0]
            self.assertEqual(issue["fixtures"][0]["path"], "testing/fixtures/example.json")
            self.assertEqual(issue["verification"]["replay"], "passed")
            self.assertEqual(issue["status"], "verified_by_replay")
            self.assertTrue((ledger_dir / "ISSUES.md").exists())

    def test_upsert_same_key_merges_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            for name in ["audit_1", "audit_2"]:
                audit_dir = ledger_dir / name
                audit_dir.mkdir()
                comparison_path = audit_dir / "comparison.json"
                comparison_path.write_text(json.dumps({"discrepancies": [{"type": "line_count_mismatch"}]}))
                self.run_cli(
                    ledger_dir,
                    "upsert",
                    "--key",
                    "line-count-fragmentation",
                    "--from-audit",
                    str(comparison_path),
                )
            data = self.read_issues(ledger_dir)
            self.assertEqual(len(data["issues"]), 1)
            self.assertEqual(len(data["issues"][0]["evidence"]), 2)

    def test_import_diagnostic_state_seeds_recurring_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            state_path = ledger_dir / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "ongoing_issues": [
                            {
                                "issue_key": "problem-input-fraction-repair-overstrips-tail-and-denominator",
                                "title": "Problem-input fraction repair over-strips numerators/denominators",
                                "status": "active_new_regression_after_last_fix",
                                "first_seen": "2026-07-03T14:57:57Z",
                                "last_seen": "2026-07-03T15:03:43Z",
                                "occurrence_count": 4,
                                "representative_audit_ids": ["audit_1", "audit_2"],
                                "likely_app_area": "src/recognition/studentWritingPipeline.js",
                                "recommended_fix": "Gate repair on structural split-fraction evidence.",
                            }
                        ]
                    }
                )
            )

            self.run_cli(ledger_dir, "import-diagnostic-state", str(state_path))
            data = self.read_issues(ledger_dir)
            issue = data["issues"][0]
            self.assertEqual(issue["id"], "WB3-0001")
            self.assertEqual(issue["severity"], "P0")
            self.assertEqual(issue["status"], "fixture_needed")
            self.assertEqual(issue["diagnostic_occurrence_count"], 4)
            self.assertEqual(len(issue["evidence"]), 2)


if __name__ == "__main__":
    unittest.main()
