from diffusers import StableDiffusionPipeline
import torch
from compel import Compel

model_id = "emilianJR/majicMIX_realistic"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

pipe = pipe.to("cuda")

#prompt = "souryuu asuka langley, scifi armor, 8k, High detail RAW color photo, realistic, (photo realism:1. 4), highly detailed CG unified 8K wallpapers, physics-based rendering, cinematic lighting, (entire body), beautiful detailed eyes, ultra highres, photorealistic, 8k, hyperrealism, cinematic lighting, photography,"
#negative_prompt = "cgi, 3d, doll, blurry, lowres, text, cropped, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
prompt = "1girl, smile, (long legs:1.2), (fire:1.2), (Slim, slender figure:1.2), heart-shaped pupils ,mars, tiara, sailor moon uniform, red sailor collar, bow, knee boots,  garter, choker, white gloves, red choker, elbow gloves, jewelry, earrings, red skirt, beautiful lighting, deep shadow, unparalleled masterpiece, ultra realistic 8k CG, perfect artwork, ((perfect female figure)), narrow waist, chinese deity, looking at viewer, seductive posture, sexy pose, clean, beautiful face, pure face, divine goddess, glint , Show belly button , (best quality, masterpiece:1.2), official art, unity 8k wallpaper, ultra detailed, beautiful and aesthetic, beautiful, masterpiece, best quality, ultra-detailed"

conditioning = compel.build_conditioning_tensor(prompt)
conditioning = conditioning.to("cuda")
conditioning_neg = compel.build_conditioning_tensor(negative_prompt)
conditioning_neg = conditioning_neg.to("cuda")

image = pipe(
    prompt_embeds = conditioning, 
    negative_prompt_embeds = conditioning_neg, 
    width=1024,
    height=1024,
    guidance_scale=12,
    num_inference_steps=800,
    random_seed= 89618,
).images[0]

image.save("generated_images/hentai140.png")
