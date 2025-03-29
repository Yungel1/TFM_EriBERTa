import unittest
from harness.cantemist_harness.ner_cantemist_process import (
    _process_example,
    _ner_process_raw_output,
    _ner_align_entities,
    ner_process_results,
    NO_ENT_STRING, NER_TYPE_SEPARATOR, NER_MAPPING
)


class TestNEREvaluation(unittest.TestCase):
    def test_process_example(self):
        example = {
            "id": "1",
            "text": "Tumor maligno detectado en el pulmón.",
            "entities": [
                {"start": 0, "end": 5, "ent_type": "MORFOLOGIA_NEOPLASIA"}
            ]
        }
        expected_output = {
            "id": "1",
            "text": "Tumor maligno detectado en el pulmón.",
            "target": f"Tumor{NER_TYPE_SEPARATOR}MORFOLOGIA_NEOPLASIA"
        }
        self.assertEqual(_process_example(example), expected_output)

    def test_ner_process_raw_output_valid(self):
        llm_output = f"tumor{NER_TYPE_SEPARATOR}MORFOLOGIA_NEOPLASIA, pulmon{NER_TYPE_SEPARATOR}O"
        expected = [("tumor", "MORFOLOGIA_NEOPLASIA"), ("pulmon", "O")]
        self.assertEqual(_ner_process_raw_output(llm_output), expected)

    def test_ner_process_raw_output_no_entities(self):
        self.assertEqual(_ner_process_raw_output(NO_ENT_STRING), [])

    def test_ner_process_raw_output_empty_string(self):
        self.assertEqual(_ner_process_raw_output(""), [("WRONG", "O")])

    def test_ner_align_entities_perfect_match(self):
        pred = [("tumor", "MORFOLOGIA_NEOPLASIA")]
        gold = [("tumor", "MORFOLOGIA_NEOPLASIA")]
        aligned_pred, aligned_gold = _ner_align_entities(pred, gold)
        self.assertEqual(aligned_pred, [NER_MAPPING["MORFOLOGIA_NEOPLASIA"]])
        self.assertEqual(aligned_gold, [NER_MAPPING["MORFOLOGIA_NEOPLASIA"]])

    def test_ner_align_entities_text_match_mismatch_type(self):
        pred = [("tumor", "O")]
        gold = [("tumor", "MORFOLOGIA_NEOPLASIA")]
        aligned_pred, aligned_gold = _ner_align_entities(pred, gold)
        self.assertEqual(aligned_pred, [NER_MAPPING["O"]])
        self.assertEqual(aligned_gold, [NER_MAPPING["MORFOLOGIA_NEOPLASIA"]])

    def test_ner_align_entities_extra_pred(self):
        pred = [("tumor", "MORFOLOGIA_NEOPLASIA"), ("pulmon", "O")]
        gold = [("tumor", "MORFOLOGIA_NEOPLASIA")]
        aligned_pred, aligned_gold = _ner_align_entities(pred, gold)
        self.assertEqual(aligned_pred, [NER_MAPPING["MORFOLOGIA_NEOPLASIA"], NER_MAPPING["O"]])
        self.assertEqual(aligned_gold, [NER_MAPPING["MORFOLOGIA_NEOPLASIA"], NER_MAPPING["O"]])

    def test_ner_process_results_empty(self):
        doc = {"target": NO_ENT_STRING}
        results = NO_ENT_STRING
        expected = {"f1": ([NER_MAPPING['O']], [NER_MAPPING['O']])}
        self.assertEqual(ner_process_results(doc, results), expected)

    def test_ner_process_results_valid(self):
        doc = {"target": f"tumor{NER_TYPE_SEPARATOR}MORFOLOGIA_NEOPLASIA"}
        results = f"tumor{NER_TYPE_SEPARATOR}MORFOLOGIA_NEOPLASIA"
        expected = {"f1": ([NER_MAPPING["MORFOLOGIA_NEOPLASIA"]], [NER_MAPPING["MORFOLOGIA_NEOPLASIA"]])}
        self.assertEqual(ner_process_results(doc, results), expected)


if __name__ == "__main__":
    unittest.main()
