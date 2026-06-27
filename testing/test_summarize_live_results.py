import unittest

from testing.summarize_live_results import aggregate_summaries


class LiveResultSummaryTests(unittest.TestCase):
    def test_aggregates_live_result_totals(self):
        result = aggregate_summaries([
            {
                "path": "a.json",
                "totals": {
                    "fixtures": 2,
                    "segmentationExact": 2,
                    "ocrLines": 8,
                    "ocrAcceptedMatches": 7,
                    "ocrMisses": 1,
                },
            },
            {
                "path": "b.json",
                "totals": {
                    "fixtures": 3,
                    "segmentationExact": 3,
                    "ocrLines": 10,
                    "ocrAcceptedMatches": 10,
                    "ocrMisses": 0,
                },
            },
        ])

        self.assertEqual(result["summaryFiles"], 2)
        self.assertEqual(result["totals"]["fixtures"], 5)
        self.assertEqual(result["totals"]["ocrLines"], 18)
        self.assertEqual(result["totals"]["ocrAcceptedMatches"], 17)
        self.assertEqual(result["totals"]["ocrMisses"], 1)


if __name__ == "__main__":
    unittest.main()
