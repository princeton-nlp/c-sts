import argparse
import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          default_data_collator)

from utils.fewshot.generate_in_context_dataset import make_dataset
from utils.fewshot.openai_utils import (OPENAI_MODELS, authenticate,
                                        get_gpt_prediction)
from utils.fewshot.progress_logger import ProgressLogger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt": AutoModelForCausalLM,
    "t5": AutoModelForSeq2SeqLM,
}

NO_SKIP_MODULES = {
    "t5": ["T5Block"],
    "gpt": ["GPTJBlock"],
}

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "tf32": torch.float32,  # tf32 set by cuda.backend
}


def get_tokenizer_type(model):
    if (
        "t5" in model.lower()
        or "t0" in model.lower()
        or "tk-" in model.lower()
        or "ul2" in model.lower()
    ):
        return "t5"
    elif "gpt" in model.lower():
        return "gpt"
    else:
        raise ValueError(f"Unknown tokenizer type {model}")


def extract_float(s):
    match = re.search(r"(\d+\.\d+|\d+)", s)
    if match:
        return float(match.group(1))
    return s


def eval(
    dataset,
    model,
    tokenizer,
    prefix,
    tokenizer_type,
    min_similarity,
    max_similarity,
    dataloader_num_workers,
    batch_size,
):
    start_time = time.time()
    if model in OPENAI_MODELS:
        all_preds, all_labels, examples, non_numeric = openai_model_eval(
            model,
            dataset,
            min_similarity,
            max_similarity,
        )
    else:
        all_preds, all_labels, examples, non_numeric = non_openai_model_eval(
            model,
            tokenizer,
            tokenizer_type,
            dataset,
            dataloader_num_workers,
            batch_size,
            min_similarity,
            max_similarity,
        )
    eval_time = time.time() - start_time
    predictions = dict(enumerate(all_preds))
    logger.info(f"Example Preds: {all_preds[:3]}")
    logger.info(f"Example Labels: {all_labels[:3]}")
    results = process_results(
        prefix,
        eval_time,
        len(dataset),
        non_numeric,
        all_preds,
        all_labels,
        min_similarity,
        max_similarity,
    )
    return results, predictions, examples


def get_tokenizer_func(tokenizer, tokenizer_type):
    def tokenizer_func(example):
        return tokenizer(
            example["text"],
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=tokenizer_type == "t5",
        )
    return tokenizer_func


def openai_model_eval(
    model, dataset, min_similarity, max_similarity
):
    all_preds, all_labels, examples = [], [], []
    non_numeric = 0
    for ix, example in ProgressLogger.wrap_iter(
        "eval", dataset, len(dataset), return_ix=True
    ):
        raw_pred = get_gpt_prediction(model, example["text"])
        pred = extract_float(raw_pred)
        if type(pred) is not float:
            non_numeric += 1
            pred = torch.empty(1).uniform_(min_similarity, max_similarity).item()
        label = float(example["label"])
        all_preds.append(pred)
        all_labels.append(label)
        examples.append(
            {
                "id": ix,
                "example": example["text"],
                "raw_pred": raw_pred,
                "pred": pred,
                "label": label,
            }
        )
        if ix < 3:
            log_example(ix, example["text"], raw_pred, label)
    return all_preds, all_labels, examples, non_numeric


def non_openai_model_eval(
    model,
    tokenizer,
    tokenizer_type,
    dataset,
    dataloader_num_workers,
    batch_size,
    min_similarity,
    max_similarity,
):
    preprocess_func = get_tokenizer_func(tokenizer, tokenizer_type)
    dataset = dataset.map(
        preprocess_func, batched=True, batch_size=batch_size
    )
    generation_kwargs = {
        "gpt": {"max_new_tokens": 20, 'pad_token_id': tokenizer.eos_token_id},
        "t5": {"max_new_tokens": 20},
    }[tokenizer_type]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        num_workers=dataloader_num_workers,
        shuffle=False,
    )
    non_numeric = 0
    all_preds, all_labels, examples = [], [], []
    with torch.no_grad():
        for ix, example in ProgressLogger.wrap_iter(
            "eval", dataloader, len(dataloader), return_ix=True
        ):
            inputs = {
                k: v.to(model.device)
                for k, v in example.items()
                if k in ["input_ids", "attention_mask"]
            }
            output = model.generate(**inputs, **generation_kwargs)
            if tokenizer_type == "gpt":
                output = output[:, inputs["input_ids"].shape[-1]:, ...]
            raw_preds = tokenizer.batch_decode(output, skip_special_tokens=True)
            preds, non_numeric = process_preds(
                raw_preds,
                non_numeric,
                min_similarity,
                max_similarity,
            )
            labels = example["labels"].tolist()
            example_texts = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            if ix * batch_size < 3:
                log_examples(
                    ix, example_texts, raw_preds, labels, batch_size
                )
            all_preds.extend(preds)
            all_labels.extend(labels)
            examples.extend(
                [
                    {
                        "id": cix + ix * batch_size,
                        "example": example_text,
                        "raw_pred": raw_pred,
                        "pred": pred,
                        "label": label,
                    }
                    for cix, example_text, raw_pred, pred, label in zip(range(len(preds)), example_texts, raw_preds, preds, labels)
                ]
            )
    return all_preds, all_labels, examples, non_numeric


def process_preds(raw_preds, non_numeric, min_similarity, max_similarity):
    preds = list(map(extract_float, raw_preds))
    non_numeric += sum(1 for p in preds if type(p) is not float)
    preds = [
        p
        if type(p) is float
        else torch.empty(1).uniform_(min_similarity, max_similarity).item()
        for p in preds
    ]
    return preds, non_numeric


def log_example(ix, text, raw_pred, label):
    example_str = "Example %d:\n\t%sPRED=%s LABEL=%s" % (
        ix,
        text.replace("\n", "\n\t"),
        raw_pred,
        label,
    )
    logger.info(example_str)


def log_examples(ix, example_texts, raw_preds, labels, batch_size):
    for cix in range(min(len(raw_preds), 3)):
        log_example(
            ix * batch_size + cix,
            example_texts[cix],
            raw_preds[cix],
            labels[cix],
        )


def process_results(
    prefix,
    eval_time,
    samples,
    non_numeric,
    all_preds,
    all_labels,
    min_similarity,
    max_similarity,
):
    scaled_preds = np.array(all_preds)
    invalid_preds = sum(
        1 for p in scaled_preds if not min_similarity <= p <= max_similarity
    )
    scaled_labels = np.array(all_labels)
    results = {
        "pearsonr": pearsonr(scaled_preds, scaled_labels)[0],
        "spearmanr": spearmanr(scaled_preds, scaled_labels)[0],
        "runtime": eval_time,
        "samples": samples,
        "samples_per_second": samples / eval_time,
        "non_numeric": non_numeric,
        "non_numeric_percent": non_numeric / samples,
        "mse": ((torch.tensor(all_preds) - torch.tensor(all_labels)) ** 2)
        .mean()
        .item(),
        "out_of_range": invalid_preds,
        "out_of_range_percent": invalid_preds / samples,
    }
    return {f"{prefix}_{k}": v for k, v in results.items()}


def load_model_and_tokenizer(model_name_or_path, tokenizer_type, api_key, dtype):
    if model_name_or_path not in OPENAI_MODELS:
        if not torch.cuda.is_available() and dtype != 'fp32':
            logger.info("Using CPU, overriding dtype to fp32")
        dtype = torch.float32 if not torch.cuda.is_available() else DTYPES[dtype]
        model_cls = MODEL_CLASSES[tokenizer_type]
        weights_location = get_weights_location(model_name_or_path)
        config = AutoConfig.from_pretrained(weights_location)
        index_location = get_index_location(weights_location)
        with init_empty_weights():
            logger.info(f"Instantiating model from config")
            model = model_cls.from_config(config)
        model = load_model_weights(model, index_location, dtype, tokenizer_type)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left" if tokenizer_type == "gpt" else "right")
        if tokenizer_type == "gpt":
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if not os.path.exists(api_key):
            raise ValueError("api_key must be a file containing your OpenAI API key")
        authenticate(api_key)
        model = model_name_or_path
        tokenizer = None
    return model, tokenizer


def get_weights_location(model_name_or_path):
    if not os.path.exists(model_name_or_path):
        return snapshot_download(
            repo_id=model_name_or_path,
            ignore_patterns=["*h5*", "*msgpack*", "*safetensors*", '*tflite*', '*rust_model.ot*'],  # only download pytorch weights
        )
    elif os.path.isdir(model_name_or_path):
        return model_name_or_path
    else:
        return os.path.dirname(model_name_or_path)


def get_index_location(weights_location):
    index_location = os.path.join(weights_location, "pytorch_model.bin.index.json")
    if not os.path.exists(index_location):
        index_location = os.path.join(weights_location, "pytorch_model.bin")
    return index_location


def load_model_weights(model, index_location, dtype, tokenizer_type):
    logger.info("Loading model weights with load_checkpoint_and_dispatch")
    model = load_checkpoint_and_dispatch(
        model,
        index_location,
        device_map="balanced",
        no_split_module_classes=NO_SKIP_MODULES[tokenizer_type],
        dtype=dtype,
    )
    logger.info(f"Loaded model with load_checkpoint_and_dispatch from {index_location}")
    return model


def save_results(results, predictions, examples, output_dir, output_file_prefix):
    logger.info(f"{output_file_prefix} results: %s" % json.dumps(results, indent=4))
    logger.info("Writing eval_results to %s" % output_dir)
    with open(Path(output_dir, f"{output_file_prefix}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(Path(output_dir, f"{output_file_prefix}_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=4)
    with open(Path(output_dir, f"{output_file_prefix}_examples.json"), "w") as f:
        json.dump(examples, f, indent=4)


def main(
    model_name_or_path,
    tokenizer_type,
    output_dir,
    train_file,
    validation_file,
    test_file,
    k_shot,
    prompt_name,
    seed,
    overwrite_output_dir,
    dataloader_num_workers,
    max_eval_samples,
    api_key,
    dtype,
    batch_size,
):
    skip_validation = overwrite_output_dir is False and Path(output_dir, "eval_results.json").exists()
    skip_test = overwrite_output_dir is False and Path(output_dir, "test_results.json").exists()
    if skip_validation:
        logger.info(f"Skipping validation, found eval_results.json in {output_dir}.\nSet overwrite_output_dir=True to override.")
    if skip_test:
        logger.info(f"Skipping test, found test_results.json in {output_dir}.\nSet overwrite_output_dir=True to override.")
    if skip_validation and skip_test:
        return
    if validation_file is None and test_file is None:
        logger.info("No validation or test file provided. Exiting.")
        return
    if model_name_or_path in OPENAI_MODELS:
        assert api_key is not None, "api_key path must be provided for OpenAI models"
    if dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    if tokenizer_type is None:
        tokenizer_type = get_tokenizer_type(model_name_or_path)
    else:
        assert tokenizer_type in {'gpt', 't5'}, f"tokenizer_type must be one of 'gpt' or 't5', got {tokenizer_type}"
    logger.info(f"Using {tokenizer_type} tokenizer")
    config_key = f"{tokenizer_type}_k{k_shot}_prompt{prompt_name}"
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path, tokenizer_type, api_key, dtype
    )
    max_similarity = 5.0
    min_similarity = 1.0 if "csts" in Path(train_file).name else 0.0
    is_stsb = "stsb" in Path(train_file).name
    logging.info("Loading dataset %s" % config_key)
    dataset = make_dataset(
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        tokenizer_type=tokenizer_type,
        k_shot=k_shot,
        prompt_name=prompt_name,
        seed=seed,
        is_stsb=is_stsb,
    )
    train_dataset = dataset["train"]
    eval_dataset, test_dataset = None, None
    if validation_file is not None:
        eval_dataset = dataset["validation"]
    if test_file is not None:
        test_dataset = dataset["test"]
    if max_eval_samples is not None and 'validation' in dataset:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    logger.info(
        "Loaded %d train examples, %d validation examples, %d test examples"
        % (len(train_dataset), len(eval_dataset) if eval_dataset is not None else 0, len(test_dataset) if test_dataset is not None else 0)
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if validation_file is not None:
        logger.info("Evaluating validation dataset")
        eval_results, eval_predictions, eval_examples = eval(
            dataset=eval_dataset,
            model=model,
            tokenizer=tokenizer,
            prefix='eval',
            tokenizer_type=tokenizer_type,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            dataloader_num_workers=dataloader_num_workers,
            batch_size=batch_size,
        )
        save_results(eval_results, eval_predictions, eval_examples, output_dir, "eval")
    if test_file is not None:
        logger.info("Predicting on test dataset")
        test_results, test_predictions, test_examples = eval(
            dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            prefix='test',
            tokenizer_type=tokenizer_type,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            dataloader_num_workers=dataloader_num_workers,
            batch_size=batch_size,
        )
        save_results(test_results, test_predictions, test_examples, output_dir, "test")
    logger.info("Done!")


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_type", type=str, help="Tokenizer type (gpt or t5). If not provided, will be inferred from model_name_or_path.")
    parser.add_argument("--k_shot", type=int, required=True, help="Number of examples to use in prompt.")
    parser.add_argument("--prompt_name", type=str, required=True, help="Name of prompt to use. See utils/fewshot/generate_in_context_dataset.py for options.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_file", type=str, required=True, help="Path to train file.")
    parser.add_argument("--validation_file", type=str, required=False, help="Path to validation file. If not provided, will not run validation.")
    parser.add_argument("--test_file", type=str, required=False, help="Path to test file. If not provided, will not run test.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=string_to_bool,
        default=False,
        nargs="?",
        const=True,
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--api_key", type=str, required=False, help="Path to OpenAI API key"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32", "tf32"],
        help="Data used for model. TF32 and BF16 are recommended but only supported for NVIDIA GPUs with Ampere architecture or later.",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()
    main(**vars(args))
