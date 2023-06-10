def scale_to_range(labels, _min, _max):
    return list(map(lambda x: (x - _min) / (_max - _min), labels))


def get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        condition_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        scale=None,
        condition_only=False,
        sentences_only=False,
        ):
    'Returns a the preprocessing function for each encoding type'
    if model_args.encoding_type == 'bi_encoder':
        if condition_only or sentences_only:
            raise ValueError('condition_only and sentences_only doesn\'t apply to bi_encoder')
        def preprocess_function(examples):
            sent1_args = (examples[sentence1_key], examples[condition_key])
            sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent2_args = (examples[sentence2_key], examples[condition_key])
            sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent1_result['input_ids_2'] = sent2_result['input_ids']
            sent1_result['attention_mask_2'] = sent2_result['attention_mask']
            if 'token_type_ids' in sent2_result:
                sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']
            sent1_result['labels'] = examples[similarity_key]
            if scale is not None:
                _min, _max = scale
                for label in sent1_result['labels']:
                    if (label < _min or label > _max) and label != -1:
                        raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
                sent1_result['labels'] = scale_to_range(sent1_result['labels'], _min, _max)
            return sent1_result
    elif model_args.encoding_type == 'cross_encoder':
        def preprocess_function(examples):
            if condition_only:
                input_args = examples[condition_key]
            elif sentences_only:
                input_args = list(map(lambda x: ' '.join([x[0], tokenizer.sep_token, x[1]]), zip(examples[sentence1_key], examples[sentence2_key])))
            else:
                input_args = list(map(lambda x: ' '.join([x[0], tokenizer.sep_token, x[1], tokenizer.sep_token, x[2]]), zip(examples[sentence1_key], examples[sentence2_key], examples[condition_key])))
            result = tokenizer(input_args, padding=padding, max_length=max_seq_length, truncation=True)
            result['labels'] = examples[similarity_key]
            if scale is not None:
                _min, _max = scale
                for label in result['labels']:
                    if (label < _min or label > _max) and label != -1:
                        raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
                result['labels'] = scale_to_range(result['labels'], _min, _max)
            return result
    elif model_args.encoding_type == 'tri_encoder':
        if condition_only or sentences_only:
            raise ValueError('condition_only and sentences_only doesn\'t apply to tri_encoder')
        def preprocess_function(examples):
            sent1_args = (examples[sentence1_key], )
            sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent2_args = (examples[sentence2_key], )
            sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent3_args = (examples[condition_key], )
            sent3_result = tokenizer(*sent3_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent1_result['input_ids_2'] = sent2_result['input_ids']
            sent1_result['attention_mask_2'] = sent2_result['attention_mask']
            sent1_result['input_ids_3'] = sent3_result['input_ids']
            sent1_result['attention_mask_3'] = sent3_result['attention_mask']
            if 'token_type_ids' in sent2_result:
                sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']
                sent1_result['token_type_ids_3'] = sent3_result['token_type_ids']
            sent1_result['labels'] = examples[similarity_key]
            if scale is not None:
                _min, _max = scale
                for label in sent1_result['labels']:
                    if (label < _min or label > _max) and label != -1:
                        raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
                sent1_result['labels'] = scale_to_range(sent1_result['labels'], _min, _max)
            return sent1_result
    else:
        raise ValueError(f'Invalid model type: {model_args.encoding_type}')
    return preprocess_function
