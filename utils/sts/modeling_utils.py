from .modeling_encoders import BiEncoderForClassification, CrossEncoderForClassification, TriEncoderForClassification
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DataCollatorWithPadding:
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # get max length of all sequences in features
        max_length = max(max(len(feature[key]) for feature in features) for key in features[0] if key.startswith('input_ids'))
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        # pad all sequences to max length
        out_features = {}
        for key in features[0].keys():
            if key.startswith('input_ids') or key.startswith('attention_mask'):
                pad_token = self.pad_token_id if key.startswith('input_ids') else 0
                out_features[key] = [feature[key] + [pad_token] * (max_length - len(feature[key])) for feature in features]
            else:
                out_features[key] = [feature[key] for feature in features]
        if self.return_tensors == 'pt':
            out_features = {key: torch.tensor(value) for key, value in out_features.items()}
        elif self.return_tensors == 'np':
            out_features = {key: np.array(value) for key, value in out_features.items()}
        return out_features


def get_model(model_args):
    if model_args.encoding_type == 'bi_encoder':
        return BiEncoderForClassification
    if model_args.encoding_type == 'cross_encoder':
        return CrossEncoderForClassification
    if model_args.encoding_type == 'tri_encoder':
        return TriEncoderForClassification
    raise ValueError(f'Invalid model type: {model_args.encoding_type}')

def add_args_to_config(config, model_args, data_args):
    '''
    Add important arguments to the config
    '''
    prefix = 'csts'
    for arg in vars(model_args):
        if prefix in arg:
            setattr(config, arg, getattr(model_args, arg))
    for arg in vars(data_args):
        if prefix in arg:
            setattr(config, arg, getattr(data_args, arg))
    return config