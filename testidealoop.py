from diffusers import StableDiffusionPipeline
import torch
from compel import Compel

model_id = "emilianJR/majicMIX_realistic"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

pipe = pipe.to("cuda")

# Define the prompts
prompt = "1girl, smile,(long leg:1.2), heart-shaped pupils,(Slim, slender figure:1.2), mer1, tiara, sailor mercury, sailor senshi uniform, sexy red sailor moon bra, almost nude breast, blue sailor collar, bow, red supergirl boots, white gloves, blue choker, elbow gloves, jewelry, earrings, blue T-back (thong), beautiful lighting, deep shadow, nebula , unparalleled masterpiece, ultra realistic 8k CG, perfect artwork, ((perfect female figure)), narrow waist, chinese deity, looking at viewer, seductive posture, sexy pose, clean, beautiful face, pure face, divine goddess, glint , Show belly button , Sea wave, (best quality, masterpiece:1.2), official art, unity 8k wallpaper, ultra detailed, beautiful and aesthetic, masterpiece, best quality"
negative_prompt = "extra limbs, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# Number of iterations
num_iterations = 100  

# Initial random seed
initial_seed = 89627

for i in range(num_iterations):
    random_seed = initial_seed + i
    image_file_name = f"image_file/image{42 + i}.png"

    conditioning = compel.build_conditioning_tensor(prompt)
    conditioning = conditioning.to("cuda")
    conditioning_neg = compel.build_conditioning_tensor(negative_prompt)
    conditioning_neg = conditioning_neg.to("cuda")

    image = pipe(
        prompt_embeds=conditioning,
        negative_prompt_embeds=conditioning_neg,
        width=1024,
        height=1024,
        guidance_scale=12,
        num_inference_steps=200,
        random_seed=random_seed,
    ).images[0]

    image.save(image_file_name)
