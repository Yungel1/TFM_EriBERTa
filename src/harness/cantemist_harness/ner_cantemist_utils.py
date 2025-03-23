from src.harness.gemma_ner_processor import GemmaNER

gn = GemmaNER(ner_entity_list=['O', 'MORFOLOGIA_NEOPLASIA'])


def ner_process_docs(dataset):
    return gn.process_dataset(dataset)


def ner_process_results(doc, results):
    return gn.ner_process_results(doc, results)
