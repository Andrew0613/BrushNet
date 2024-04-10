from diffusers import StableDiffusionBrushNetDDIMInversionPipeline, BrushNetModel, UniPCMultistepScheduler, AutoencoderKL, UNet2DConditionAdaINModel
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image

base_model_path = "pretrained/sd_checkpoints/Realistic_Vision_V6.0_B1_noVAE"
brushnet_path = "runs/logs/brushnet_adain_segmentationmask/checkpoint-300000/brushnet"
vae_model_path = "pretrained/sd_checkpoints/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float32)

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float32)
unet = UNet2DConditionAdaINModel.from_pretrained(base_model_path, subfolder="unet")
pipe = StableDiffusionBrushNetDDIMInversionPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, vae=vae, unet=unet, torch_dtype=torch.float32, low_cpu_mem_usage=False
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

# image_path="examples/brushnet/src/girl/girl.jpeg"
# mask_path="examples/brushnet/src/girl/girl_mask.png"
# caption="adventurer girl, best quality, 4K"

# image_path="examples/brushnet/src/368667/368667-input.png"
# mask_path="examples/brushnet/src/368667/368667-mask1.png"
# caption = "Horses grazing by a stream at a ranch"

# image_path="examples/brushnet/src/242679/242679-input.png"
# mask_path="examples/brushnet/src/242679/242679-mask1.png"
# caption = "A cat lounges on a seat under a mirror onboard a train, next to a cluttered counter."

# image_path="examples/brushnet/src/cute_girl/cute_girl.jpeg"
# mask_path="examples/brushnet/src/cute_girl/cute_girl_mask2.png"
# caption = "score_7_up, 1girl with glasses, solo, expressive, hearts, dynamic, brown hair, brown eyes, pale skin girl, pale skin, freckles, green turtleneck, cozy, blush, in_office, looking_longingly, seductive."

# image_path="examples/brushnet/src/1/1.png"
# mask_path="examples/brushnet/src/1/1_mask.jpeg.png"
# caption = "minimal dark moody portrait of a menacing hooded Emperor Palpatine emerging from the shadows."


# image_path="examples/brushnet/src/2/2.png"
# mask_path="examples/brushnet/src/2/2_mask3.png"
# caption = "a flower "
# caption = "3d render of a cute simba from the lion king with a hat, disney pixar animation, game render, lion king movie still, very detailed, 4k resolution, specular lighting, 35mm Canon f/8, outstanding beauty, masterpiece."

# image_path="examples/brushnet/mask_test/images/8/8.jpeg"
# mask_path="examples/brushnet/mask_test/images/8/8_mask2.png"
# caption = "empty"

# image_path="examples/brushnet/src/girl/girl.jpeg"
# mask_path="examples/brushnet/src/girl/10_mask2.png"

# image_path="examples/brushnet/src/test_image.jpg"
# mask_path="examples/brushnet/src/test_mask.jpg"
# caption = "empty"


image_path="examples/brushnet/src/example_4.jpeg"
mask_path="examples/brushnet/src/example_4_mask.jpg"
caption = "car in the forest"

init_image = cv2.imread(image_path)
init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
# # downsample 
# mask = cv2.resize(mask, (mask.shape[1]//2,mask.shape[0]//2))
#resize by mask
init_image = cv2.resize(init_image, (mask.shape[1],mask.shape[0]))
mask_image = np.zeros_like(mask)
mask_image[np.all(mask == [0, 0, 0], axis=-1)] = [1.,1.,1.]

# mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
mask_image = 1.*(mask.sum(-1)>255)[:,:,np.newaxis]
# init_image = init_image * (1-mask_image)
init_image = init_image * mask_image

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
# mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(1,-1)*255).convert("RGB")

init_image.save("init_image.png")
mask_image.save("mask_image.png")

generator = torch.Generator("cuda").manual_seed(-1)

image = pipe(
    caption, 
    init_image, 
    mask_image, 
    num_inference_steps=50, 
    generator=generator,
    paintingnet_conditioning_scale=1.0
).images[0]

image.save("output_adain.png")