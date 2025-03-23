import unittest

from datasets import Dataset

from src.harness.gemma_ner_processor import GemmaNER


class TestGemmaNER(unittest.TestCase):
    def setUp(self):
        self.gn = GemmaNER(ner_entity_list=['O', 'NO_NORMALIZABLES', 'NORMALIZABLES', 'PROTEINAS', 'UNCLEAR'])

    def test_process_dataset(self):
        """Test the process_dataset function with multiple cases."""
        dataset = Dataset.from_dict({
            "text": [
                "Aspirina es un medicamento.",
                "El paracetamol es común.",
                "No hay entidades aquí.",
                "Ibuprofeno y aspirina son medicamentos."
            ],
            "entities": [
                [{"start": 0, "end": 8, "label": "NORMALIZABLES"}],
                [{"start": 3, "end": 14, "label": "NORMALIZABLES"}],
                [],
                [{"start": 0, "end": 10, "label": "NORMALIZABLES"}, {"start": 13, "end": 21, "label": "NORMALIZABLES"}]
            ],
            "article_id": ["1", "2", "3", "4"]
        })

        processed_dataset = self.gn.process_dataset(dataset)
        self.assertEqual(len(processed_dataset), 4)  # Ensure all entries are processed
        self.assertEqual(processed_dataset[0]["target"], "Aspirina$NORMALIZABLES")
        self.assertEqual(processed_dataset[1]["target"], "paracetamol$NORMALIZABLES")
        self.assertEqual(processed_dataset[2]["target"], "&&NOENT&&")  # No entities case
        self.assertEqual(processed_dataset[3]["target"], "Ibuprofeno$NORMALIZABLES,aspirina$NORMALIZABLES")

    def test_exact_match(self):
        """Test when predictions perfectly match the ground truth."""
        doc = {"target": "Aspirina$NORMALIZABLES,Paracetamol$NORMALIZABLES"}
        pred_results = "Aspirina$NORMALIZABLES,Paracetamol$NORMALIZABLES"
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertEqual(output["f1"], ([2, 2], [2, 2]))

    def test_no_entities(self):
        """Test when both the ground truth and the predictions are empty."""
        doc = {"target": "&&NOENT&&"}
        pred_results = "&&NOENT&&"
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertEqual(output["f1"], ([0], [0]))

    def test_over_prediction(self):
        """Test when the model predicts more entities than present in the ground truth."""
        doc = {"target": "Aspirina$NORMALIZABLES"}
        pred_results = "Aspirina$NORMALIZABLES,Paracetamol$NORMALIZABLES"
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertEqual(len(output["f1"][0]), len(output["f1"][1]))  # Ensure lengths match

    def test_under_prediction(self):
        """Test when the model misses an entity present in the ground truth."""
        doc = {"target": "Aspirina$NORMALIZABLES,Paracetamol$NORMALIZABLES"}
        pred_results = "Aspirina$NORMALIZABLES"
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertEqual(len(output["f1"][0]), len(output["f1"][1]))  # Ensure alignment

    def test_type_mismatch(self):
        """Test when entity text matches but the type is wrong."""
        doc = {"target": "Aspirina$NORMALIZABLES"}
        pred_results = "Aspirina$UNCLEAR"
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertNotEqual(output["f1"][0], output["f1"][1])  # Different types

    def test_completely_wrong_predictions(self):
        """Test when the model's predictions are entirely incorrect."""
        doc = {"target": "Aspirina$NORMALIZABLES"}
        pred_results = "Ibuprofeno$UNCLEAR"
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertNotIn(self.gn.ner_mapping["NORMALIZABLES"], output["f1"][0])  # Model missed it

    def test_noise_handling(self):
        """Test how the model handles malformed input."""
        doc = {"target": "Aspirina$NORMALIZABLES, Paracetamol$NORMALIZABLES"}
        pred_results = "Aspirina$NORMALIZABLES, Paracetamol$PROTEINAS"  # Wrong type
        output = self.gn.ner_process_results(doc, pred_results)
        self.assertEqual(len(output["f1"][0]), len(output["f1"][1]))  # Alignment check


if __name__ == '__main__':
    unittest.main()
