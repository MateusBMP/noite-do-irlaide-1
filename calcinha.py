import os
from sys import argv

FLAVIO_IA = False;

for arg in argv:
    if arg == "--flavio-ia":
        FLAVIO_IA = True
        break

if (FLAVIO_IA):
    import os
    from PIL import Image
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image = Image.open("./flavio-ia.png").convert("RGB")
    prompt = "Replace the underpants with panties."

    with torch.inference_mode():
        output = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1)
        output_image = output.images[0]
        output_image.save("flavio-ia-calcinha.png")
        print("image saved at", os.path.abspath("flavio-ia-calcinha.png"))
