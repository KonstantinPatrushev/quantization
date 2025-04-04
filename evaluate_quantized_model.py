from time import time
from gliner import GLiNER
from datasets import load_dataset


def ner_tags_to_spans(samples, tag_to_id):
    """
    Converts NER tags in the dataset samples to spans (start, end, entity type).

    Args:
        samples (dict): A dictionary containing the tokens and NER tags.
        tag_to_id (dict): A dictionary mapping NER tags to IDs.

    Returns:
        dict: A dictionary containing tokenized text and corresponding NER spans.
    """
    ner_tags = samples["ner_tags"]
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    spans = []
    start_pos = None
    entity_name = None

    for i, tag in enumerate(ner_tags):
        if tag == 0:
            if entity_name is not None:
                spans.append((start_pos, i - 1, entity_name))
                entity_name = None
                start_pos = None
        else:
            tag_name = id_to_tag[tag]
            if tag_name.startswith('B-'):
                if entity_name is not None:
                    spans.append((start_pos, i - 1, entity_name))
                entity_name = tag_name[2:]
                start_pos = i
            elif tag_name.startswith('I-'):
                continue

    if entity_name is not None:
        spans.append((start_pos, len(samples["tokens"]) - 1, entity_name))

    return {"tokenized_text": samples["tokens"], "ner": spans}


def evaluate_model(model_name, device, format=None):
    if format:
        # Загрузка квантованной модели
        model = GLiNER.from_pretrained(model_name, load_onnx_model=True, onnx_model_file=f'onnx/model_{format}.onnx', load_tokenizer=False, map_location=device)
    else:
        model = GLiNER.from_pretrained(model_name, load_tokenizer=False, map_location=device)

    # Замер скорости и качества работы модели
    t0 = time()
    evaluation_results = model.evaluate(
    gliner_data_conll,
    flat_ner=True,
    entity_types=["person", "organization", "location", "others"],
    threshold=0.5,
    batch_size=16
    )
    print(time() - t0)
    print(evaluation_results)


# Загрузка тестового датасета
dataset = load_dataset("eriktks/conll2003")
tag_to_id = {
    'O': 0, 'B-person': 1, 'I-person': 2, 'B-organization': 3, 'I-organization': 4,
    'B-location': 5, 'I-location': 6, 'B-others': 7, 'I-others': 8
}
gliner_data_conll = [ner_tags_to_spans(i, tag_to_id) for i in dataset['train']]

# Замер скорости и качества работы неквантованной модели на CPU
evaluate_model("urchade/gliner_multi-v2.1", 'cpu')

# Замер скорости и качества работы неквантованной модели на GPU
evaluate_model("urchade/gliner_multi-v2.1", 'cuda')

# Замер скорости и качества работы квантованной в формат fp16 модели на GPU
evaluate_model("urchade/gliner_multi-v2.1", 'cuda', 'fp16')

# Замер скорости и качества работы квантованной в формат int8 модели на CPU
evaluate_model("urchade/gliner_multi-v2.1", 'cpu', 'int8')

# Замер скорости и качества работы квантованной в формат int8 модели на CPU
evaluate_model("urchade/gliner_multi-v2.1", 'cuda', 'q4f16')

# Замер скорости и качества работы квантованной в формат uint8 модели на CPU
evaluate_model("urchade/gliner_multi-v2.1", 'cpu', 'uint8')
