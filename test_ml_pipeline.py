
import json
import os
import unittest

class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_threshold(self):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        self.assertGreaterEqual(metrics["accuracy"], 0.85)


if __name__ == "__main__":
    unittest.main()
