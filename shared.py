import argparse
import importlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tokenizer_kwargs", action="store_true", default=True)

    # wandb stuff
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--wandb_group", type=str, default="mistral-7b")

    #
    parser.add_argument(
        "--decoder_layer_import",
        type=str,
        default="transformers.models.mistral.modeling_mistral,MistralDecoderLayer",
        # default="transformers.models.persimmon.modeling_persimmon,PersimmonDecoderLayer",
    )

    args = parser.parse_args()
    return args


def import_dec_layer(import_str):
    """
    import_str = 'path,class_name'
    use for strings like:
        `transformers.models.persimmon.modeling_persimmon,PersimmonDecoderLayer`

    rather than
        `from transformers.models.mistral.modeling_mistral import MistralDecoderLayer`

    """
    module_path, class_name = import_str.split(",")
    DecoderLayer = getattr(importlib.import_module(module_path), class_name)

    return DecoderLayer
