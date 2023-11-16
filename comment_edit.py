from copy import deepcopy # This won't change the original object
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import torch
import torch.nn as nn
from mmengine import ProgressBar

from diffusers import StableDiffusionPipeline
from mmengine import Registry

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.__version__)

# define the Config Class
class Config:
    change_k = True 
    MAX_LENGTH = 'max_length'
    LAMBDA = 1


sd_model = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base').to(device)
sub_nets = sd_model.unet.named_children()
# the sub_nets is a generator, so we need to convert it to a list   
sub_nets = list(sub_nets)
cross_attention : List[nn.Module] = []
sub_nets = sd_model.unet.named_children()
sub_nets = list(sub_nets)
# Check the following code to see the structure of sub_nets

"""
print(sub_nets[3][0])
print(sub_nets[4][0])
print(sub_nets[5][0])
"""

for net in sub_nets:
    if 'up' in net[0] or 'down' in net[0]:
        for block in net[1]:
            if 'Cross' in block.__class__.__name__:
                for attention in block.attentions:
                    for transformer in attention.transformer_blocks:
                        cross_attention.append(transformer.attn2)

    # This is True in our case
    if 'mid' in net[0]:
        for attention in net[1].attentions:
            for transformer in attention.transformer_blocks:
                cross_attention.append(transformer.attn2)

# As mentioned in the paper, the og_matrices are the original matrices of the cross attention layers, 
# W^old, C^i is the old concept, and C*^i is the new concept, C^j is the concept we need to preserve
proj_layers = [layer.to_v for layer in cross_attention]
og_matrices = [deepcopy(layer.to_v) for layer in cross_attention]

# print(proj_layers[1].weight.shape)  
# Return the shape of the weight matrix, torch.Size([320, 1024]) in our case
# 320 is the dimension of the embedding, 1024 is the dimension of the hidden states
# Mentioned in the paper, check the part under the Figure 2, the first equation
if Config.change_k:
    proj_layers.extend([layer.to_k for layer in cross_attention])
    og_matrices.extend([deepcopy(layer.to_k) for layer in cross_attention])

# print(len(cross_attention))
# print(len(proj_layers))

old_concept : List[str]
edit_concept : List[str]
preserve_concept : List[str]
preserve_concept = None
old_concept = ['The cat is on the mat']
edit_concept = ['The green cat is on the blue mat']

proc_old_concept : List[str] = []
proc_edit_concept : List[str] = []
proc_preserve_concept : List[str] = [''] if preserve_concept is None else deepcopy(preserve_concept)

for old_c, edit_c in zip(old_concept, edit_concept):
    proc_old_concept.append(old_c)
    proc_edit_concept.append(edit_c if edit_c != '' else ' ')


for i in range(len(proj_layers)):
    matrix_1 = proj_layers[i].weight
    matrix_2 = torch.nn.Parameter(torch.eye(proj_layers[i].weight.shape[1], device=device, requires_grad=True))
    
    for old_c, edit_c in zip(proc_old_concept, proc_edit_concept):  
        text = [old_c, edit_c]
        text_input = sd_model.tokenizer(text, 
                                        return_tensors='pt', 
                                        padding=Config.MAX_LENGTH,
                                        truncation=True)
        text_embedding = sd_model.text_encoder(text_input.input_ids.to(device))[0]
        
        # print(text_input.attention_mask[0].shape)
        # print(text_input.attention_mask[1].shape)   
        # print(text_input.attention_mask[0].sum().item())    
        # print(text_input.attention_mask[1].sum().item())
        # final_token_ind = text_input.attention_mask[0].sum().item() - 1
        # The reason why we need to minus 1 is that the attention mask is a list of 0 and 1,
        # and the 1 is the padding token, which is the last token in the sentence
        # final_token_ind_new = text_input.attention_mask[1].sum().item() - 1
        # In our case, this would be -2 since we want to align the last token in the old concept
        # and the last token in the new concept
        final_token_ind = text_input.attention_mask[0].sum().item() - 2
        final_token_ind_new = text_input.attention_mask[1].sum().item() - 2
        farthest = max([final_token_ind, final_token_ind_new])

        old_text_embedding = text_embedding[0]
        old_text_embedding = old_text_embedding[final_token_ind:len(old_text_embedding) - max(farthest - final_token_ind, 0)]
        new_text_embedding = text_embedding[1]  
        new_text_embedding = new_text_embedding[final_token_ind_new:len(new_text_embedding) - max(farthest - final_token_ind_new, 0)]
        # print(old_text_embedding.shape)
        # print(new_text_embedding.shape)

        # freeze the old_text_embedding 
        context = old_text_embedding.detach()
        
        # freeze the new_text_embedding
        value: List[torch.Tensor] = []
        for layer in proj_layers:
            value.append(layer(new_text_embedding).detach())
        
        context_vec = context.reshape(context.shape[0], context.shape[1], 1)
        context_vec_t = context.reshape(context.shape[0], 1, context.shape[1])
        value_vec = value[i].reshape(value[i].shape[0], value[i].shape[1], 1)

        matrix_1_first_part = torch.matmul(value_vec,context_vec_t).sum(dim = 0)
        matrix_2_first_part = torch.matmul(context_vec, context_vec_t).sum(dim = 0)
        
        # You need this otherwise we are adding grad = True to the matrix 1 and matrix 2
        with torch.no_grad():
           matrix_1 += Config.LAMBDA * matrix_1_first_part
           matrix_2 += Config.LAMBDA * matrix_2_first_part
        

    for old_c, edit_c in zip(proc_preserve_concept, proc_preserve_concept):
        text = [old_c, edit_c]
        text_input = sd_model.tokenizer(text, 
                                        return_tensors='pt', 
                                        padding=Config.MAX_LENGTH,
                                        truncation=True)
        text_embedding = sd_model.text_encoder(text_input.input_ids.to(device))[0]
        
        old_text_embedding, new_text_embedding = text_embedding
        # print(old_text_embedding.shape)
        # print(new_text_embedding.shape)
        
        # freeze the old_text_embedding 
        context = old_text_embedding.detach()
        
        # freeze the new_text_embedding
        value: List[torch.Tensor] = []
        for layer in proj_layers:
            value.append(layer(new_text_embedding).detach())
        
        context_vec = context.reshape(context.shape[0], context.shape[1], 1)
        context_vec_t = context.reshape(context.shape[0], 1, context.shape[1])
        value_vec = value[i].reshape(value[i].shape[0], value[i].shape[1], 1)

        matrix_1_first_part = torch.matmul(value_vec, context_vec_t).sum(dim = 0)
        matrix_2_first_part = torch.matmul(context_vec, context_vec_t).sum(dim = 0)
        
        # print(type(matrix_1))
        # print(type(matrix_2))
        with torch.no_grad():
            matrix_1 += Config.LAMBDA * matrix_1_first_part
            matrix_2 += Config.LAMBDA * matrix_2_first_part
        

        # print(type(matrix_1))
        # print(type(matrix_2))
        # Update the weight matrix, this does not require gradient to retrain the model
        proj_layers[i].weight = torch.nn.Parameter(torch.matmul(matrix_1, torch.inverse(matrix_2)))
        