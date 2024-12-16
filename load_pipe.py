import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from diffusers import StableDiffusionPipeline

device = "cuda:5"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
).to(device)
