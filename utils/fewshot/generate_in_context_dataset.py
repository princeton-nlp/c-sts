import torch
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import random
from pathlib import Path
from functools import partial
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SEP_TOKENS = {
    't5': '</s>',
    'gpt': ' ',
}
LABEL_KEY_FUNCS = {
    't5': lambda x: f'{x.capitalize()}:',
    'gpt': lambda x: f'{x.capitalize()}: ',
}

prompts = {
    'none': None,
    'short': 'On a scale between 1 and 5, how similar are the following two sentences with respect to the condition provided? Respond only with a score between 1 and 5.',
    'long': 'Definition: Evaluate the similarity between the two sentences, with respect to the condition. Assign the pair a score between 1 and 5 as follows: 1 : The two sentences are completely dissimilar with respect to the condition. 2 : The two sentences are dissimilar, but are on a similar topic with respect to the condition. 3 : The two sentences are roughly equivalent, but some important information differs or is missing with respect to the condition. 4 : The two sentences are mostly equivalent, but some unimportant details differ with respect to the condition. 5 : The two sentences are completely equivalent with respect to the condition.',
}
stsb_prompts = {
    'none': None,
    'short': 'On a scale between 0 and 5, how similar are the following two sentences? Respond only with a score between 0 and 5.',
    'long': 'Definition: Evaluate the similarity between them and classify them into classes from 0-5 as follows: 0 : The two sentences are completely dissimilar. 1 : The two sentences are not equivalent, but are on the same topic. 2 : The two sentences are not equivalent, but share some details. 3 : The two sentences are roughly equivalent, but some important information differs/missing. 4 : The two sentences are mostly equivalent, but some unimportant details differ. 5 : The two sentences are completely equivalent, as they mean the same thing.',
}
def get_prompt(prompt_name, is_stsb=False):
    if prompt_name is None:
        return None
    else:
        if is_stsb:
            return stsb_prompts[prompt_name]
        else:
            return prompts[prompt_name]


def convert_text(
    example,
    sep_token,
    label_key_func=lambda x: f'{x.capitalize()}: ',
    sentence_1_label='sentence1',
    sentence_2_label='sentence2',
    condition_label='condition',
    answer_label='label',
    is_stsb=False,
):
    sent_1 = example[sentence_1_label].strip()
    sent_2 = example[sentence_2_label].strip()
    condition = example[condition_label] if example[condition_label] is not None else ''  # bug some conditions are None
    similarity = example[answer_label]
    if is_stsb:
            ex_list = [
        label_key_func('input'),
        ' '.join([label_key_func('sentence 1'), sent_1, ]),
        ' '.join([label_key_func('sentence 2'), sent_2, ]),
        label_key_func('output'),
    ]
    else:
        ex_list = [
            label_key_func('input'),
            ' '.join([label_key_func('sentence 1'), sent_1, ]),
            ' '.join([label_key_func('sentence 2'), sent_2, ]),
            ' '.join([label_key_func('condition'), condition, ]),
            label_key_func('output'),
        ]
    ex_str = ' '.join(map(str, ex_list))
    return ex_str


def add_context(
    example,
    context,
    prompt,
    sep_token,
    answer_label='label',
    label_func=lambda x: f'{float(x)}',
):
    if prompt is not None:
        ex_list = [
            prompt.strip(' :'),
        ]
    else:
        ex_list = []
    for ex in context:
        entry = ex['original_text'] + label_func(ex[answer_label])
        ex_list.extend([entry, ])
    ex_list.append(example['original_text'])  # don't add a label to the last example
    return '\n'.join(ex_list)


def add_in_context_examples(dataset, context_dataset, model, k, prompt, tokenizer_type, pairs=None):
    contexts = list()
    context_ids = list()
    for ix, entry in enumerate(dataset):
        if pairs is not None:
            random_pairs = random.sample(range(len(pairs)), k=(k+1)//2)
            context_example_ids = [x for pair in random_pairs for x in pairs[pair]][:k]
        else:
            context_example_ids = random.sample(list(set(range(len(context_dataset))) - {ix}), k=k)
        context_ids.append(context_example_ids)
        context_examples = [context_dataset[idx] for idx in context_example_ids]
        contexts.append(add_context(entry, context_examples, prompt, SEP_TOKENS[tokenizer_type]))
    dataset = dataset.add_column('context_ids', context_ids)
    dataset = dataset.add_column('text', contexts)
    return dataset

def get_idx_pairs(
        dataset,
        sentence_1_label='sentence1',
        sentence_2_label='sentence2',
        condition_label='condition',
        answer_label='label',
        ):
    from collections import defaultdict
    pairs = defaultdict(list)
    for ix, datum in enumerate(dataset):
        pairs[datum[sentence_1_label] + '<-SEP->' + datum[sentence_2_label]].append(ix)
    pair_idxs = list(pairs.keys())
    drop_count = 0
    for pair_idx in pair_idxs:
        if len(pairs[pair_idx]) != 2:
            drop_count += len(pairs[pair_idx])
            pairs.pop(pair_idx)
    logger.warning('Dropping %d indices for missing pairs. Dataset has %d pairs total' % (drop_count, len(pair_idxs)))
    pairs = list(map(lambda x: sorted(pairs[x], key=lambda idx: -dataset[idx][answer_label]), pairs.keys()))
                    # negative because we want to sort in descending order (highest similarity first)
    for idx1, idx2 in pairs:
        if (dataset[idx1][sentence_1_label] != dataset[idx2][sentence_1_label]) or (dataset[idx1][sentence_2_label] != dataset[idx2][sentence_2_label]):
            raise ValueError('Pairing of indices is incorrect, sentences do not match for pair %d and %d' % (idx1, idx2))
        if (dataset[idx1][answer_label] < dataset[idx2][answer_label]):
            raise ValueError('Pairing of indices is incorrect, similarity is not in descending order for pair %d and %d' % (idx1, idx2))
    return pairs

def make_dataset(
        train_file,
        validation_file,
        test_file,
        tokenizer_type,
        k_shot,
        prompt_name,
        seed,
        is_stsb=False,
        ):
    convert_func = partial(convert_text, sep_token=SEP_TOKENS[tokenizer_type], label_key_func=LABEL_KEY_FUNCS[tokenizer_type], is_stsb=is_stsb)
    data_files = {'train': train_file}
    if validation_file is not None:
        data_files['validation'] = validation_file
    if test_file is not None:
        data_files['test'] = test_file
    raw_datasets = load_dataset('csv', data_files=data_files, keep_in_memory=True)
    raw_datasets = raw_datasets.map(lambda x: {'original_text': convert_func(x)}, batched=False, keep_in_memory=True)
    prompt = get_prompt(prompt_name, is_stsb)
    random.seed(seed)
    pairs = None
    if not is_stsb:
        pairs = get_idx_pairs(raw_datasets['train'])
    raw_datasets['train'] = add_in_context_examples(raw_datasets['train'], raw_datasets['train'], tokenizer_type, k_shot, prompt, tokenizer_type, pairs)
    if validation_file is not None:
        raw_datasets['validation'] = add_in_context_examples(raw_datasets['validation'], raw_datasets['train'], tokenizer_type, k_shot, prompt, tokenizer_type, pairs)
    if test_file is not None:
        raw_datasets['test'] = add_in_context_examples(raw_datasets['test'], raw_datasets['train'], tokenizer_type, k_shot, prompt, tokenizer_type, pairs)
    return raw_datasets

def main(train_file, test_file, tokenizer_type, output_dir, k_shot, prompt_name, seed, is_stsb):
    dataset = make_dataset(train_file, test_file, tokenizer_type, k_shot, prompt_name, seed, is_stsb)
    output_file = Path(output_dir, Path(train_file).stem + f'_{tokenizer_type}_k{k_shot}_prompt{prompt_name}_seed{seed}')
    dataset.save_to_disk(output_file)
    logger.info(f'Saved to {output_file}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, choices=sorted(LABEL_KEY_FUNCS.keys()), required=True)
    parser.add_argument('--output_dir', type=str, default='./in_context_datasets')
    parser.add_argument('--k_shot', type=int, required=True)
    parser.add_argument('--prompt_name', type=str)
    parser.add_argument('--is_stsb', action='store_true')
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    main(**vars(args))
