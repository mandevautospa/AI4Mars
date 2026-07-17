import unittest

from research_agent.tools import _notebook_digest, _project_files, _segmentation_metrics


class ResearchAgentToolsTests(unittest.TestCase):
    def test_project_listing_excludes_secrets(self) -> None:
        files = _project_files()
        self.assertIn("README.md", files)
        self.assertNotIn(".env.local", files)

    def test_notebook_digest_reads_project_notebook(self) -> None:
        digest = _notebook_digest("notebooks/03_baseline_training.ipynb", max_cells=2)
        self.assertEqual(digest["included_cells"], 2)
        self.assertGreaterEqual(digest["total_cells"], 2)

    def test_segmentation_metrics_orientation(self) -> None:
        metrics = _segmentation_metrics([[8, 2], [1, 9]], ["soil", "rock"])
        self.assertAlmostEqual(metrics["pixel_accuracy"], 0.85)
        self.assertAlmostEqual(metrics["per_class"]["soil"]["iou"], 8 / 11)
        self.assertAlmostEqual(metrics["per_class"]["rock"]["iou"], 9 / 12)


if __name__ == "__main__":
    unittest.main()
