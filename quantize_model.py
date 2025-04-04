import os
import torch
from gliner import GLiNER
from onnxruntime.quantization import quantize_dynamic, QuantType


def convert_model(onnx_save_path, quantized_save_path, gliner_model):
    """
    Функция конвертации модели в ONNX и квантования в заданный формат
    """
    text = "Hi! My name is Konstantin"
    labels = ['format', 'model', 'tool', 'cat']
    inputs, _ = gliner_model.prepare_model_inputs([text], labels)

    if gliner_model.config.span_mode == 'token_level':
        all_inputs =  (inputs['input_ids'], inputs['attention_mask'],
                    inputs['words_mask'], inputs['text_lengths'])
        input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths']
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "logits": {0: "position", 1: "batch_size", 2: "sequence_length", 3: "num_classes"},
            }
    else:
        all_inputs =  (inputs['input_ids'], inputs['attention_mask'],
                    inputs['words_mask'], inputs['text_lengths'],
                    inputs['span_idx'], inputs['span_mask'])
        input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths', 'span_idx', 'span_mask']
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
            "span_mask": {0: "batch_size", 1: "num_spans"},
            "logits": {0: "batch_size", 1: "sequence_length", 2: "num_spans", 3: "num_classes"},
        }

    all_inputs = dict(zip(input_names,all_inputs))
    # Конвертация в ONNX формат
    torch.onnx.export(
        gliner_model.model,
        all_inputs,
        f=onnx_save_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    # Квантование модели
    quantize_dynamic(
        onnx_save_path,
        quantized_save_path,
        weight_type=QuantType.QUInt8  # Формат квантования
    )


# Загрузка предобученной модели
gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1", load_tokenizer=True)

onnx_save_path = os.path.join("gliner_multi.onnx")
quantized_save_path = os.path.join("model_quantized.onnx")

convert_model(onnx_save_path, quantized_save_path, gliner_model)
