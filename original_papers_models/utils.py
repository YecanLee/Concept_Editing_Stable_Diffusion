from os.path import exists
from os import makedirs

from urllib.parse import urlparse

import torch

def on_change_single_global_state(keys, values, global_state, map_transforms=None):
    if map_transforms is not None:
        values = map_transforms(values)

    curr_state = global_state
    if isinstance(keys, str):
        last_key = keys
    else:
        for k in keys[:-1]:
            curr_state = curr_state[k]
        last_key = keys[-1]

    curr_state[last_key] = values
    return global_state

def is_local(url:str)->bool:
    """
    check if the file is already in the local folder
    """
    url_parsed = urlparse(url)
    if url_parsed.scheme in ('file', ''):
        return exists(url_parsed.path)
    return False

class TrainableLatent:
    def __init__(self, latent:torch.Tensor, trainable_w_dim:int = 6, dim:int=1):
        """
        a small slash before a variable means this value is a default value inside the community
        trainable_w_dim is the trainable w inside the stylegan2 model        
        """
        self._trainable_w_dim = 6
        self._dim = 1

        self._latent = latent
        self._trainable_latent = None
        self._fix_latent = None

    def _compute_latent(self):
        """
        a very good example about how to split a sequence of layers, and make part of it trainble, part of it none-trainable
        """
        self._trainable_latent, self._fix_latent = torch.split(
            self._latent, [self._trainable_latent, self._latent.shape[self._dim] - self._trainable_latent], dim=self._dim
        )
        self._trainable_latent.requires_grad = True
        self._fix_latent = False
    
    @property
    def latent(self):
        return torch.cat(self._trainable_latent, self._fix_latent, dim=self._dim)
    
    @property 
    def trainable_latent(self):
        if self._trainable_latent is None:
            self._compute_latent()
        return self._trainable_latent
    # this upper property equals to getter method in python
    # if the trainable_latent did not get a value during the initialization, by calling 
    # train = TrainableLatent(), this function and the compute_latent function will be automatically called
    
    @property
    def fix_latent(self):
        if self._fix_latent is None:
            self._compute_latent()
        return self._fix_latent
    
    @property
    def trainable_w_dim(self):
        return self._trainable_latent
    
    @trainable_w_dim.setter
    def set_trainable_w_dims(self, value:int):
        self._trainable_latent = value
        self._compute_latent()
    # this method will change the value of the trainable_latent
    # train.trainable_latent = 5 will change the value into 5

    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def set_dim(self, value:int):
        self._dim = value
        self._compute_latent()