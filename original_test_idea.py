import numpy as np
import torch

# Load the alphas, this is the weight for each sub-concept
alphas_dict = torch.load(f'{folder}/output/best_alphas.pt').detach_().requires_grad_(False)

# Normalize the alphas, so that they sum to 1
total_alpha = alphas_dict.sum()
normalized_alphas = alphas_dict / total_alpha

# Sort tokens by normalized alpha values
sorted_indices = torch.argsort(normalized_alphas, descending=True)
sorted_alphas = normalized_alphas[sorted_indices]

# Calculate cumulative sum of sorted alphas
cumulative_alphas = torch.cumsum(sorted_alphas, dim=0)

# Identify top subconcepts
threshold = 0.8
top_indices = torch.where(cumulative_alphas <= threshold)[0]
top_tokens = [dictionary[idx] for idx in sorted_indices[top_indices]]

for token_id in top_tokens:
    subconcept = pipe.tokenizer.decode([token_id])
    prompt = f'a photo of a {subconcept}'
    generator = torch.Generator("cuda").manual_seed(target_seed)
    image = pipe(prompt, guidance_scale=7.5,
                 generator=generator,
                 return_dict=False,
                 num_images_per_prompt=1,
                 num_inference_steps=num_inference_steps)[0][0]
    # Display the image
    display(image.resize((224, 224)))

