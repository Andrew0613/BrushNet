from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler, AutoencoderKL
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image
import json
import glob
import os
from tqdm import tqdm

img_folder_dir = "examples/brushnet/magicbrush/test/images"
description_path = "examples/brushnet/magicbrush/test/edit_turns.json"
output_dir = "output/magicbrush_test/instruction"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
base_model_path = "pretrained/sd_checkpoints/Realistic_Vision_V4_noVAE"
brushnet_path = "pretrained/brushnet"
vae_model_path = "pretrained/sd_checkpoints/sd-vae-ft-mse"

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, vae=vae, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

resize_512 = False

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


img_folder_list = sorted(os.listdir(img_folder_dir))

edit_turns = json.load(open(description_path))

for edit_turn in tqdm(edit_turns):
    input_img_name = edit_turn["input"]
    mask_img_name = edit_turn["mask"]
    output_img_name = edit_turn["output"]
    instruction = edit_turn["instruction"]
    img_idx = input_img_name.split('-')[0]
    
    init_img_path = os.path.join(img_folder_dir, img_idx, input_img_name)
    mask_img_path = os.path.join(img_folder_dir, img_idx, mask_img_name)
    
    init_img = cv2.imread(init_img_path)
    init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(mask_img_path)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    
    if resize_512:
        mask_img = cv2.resize(mask_img, (512,512))
        
    init_img = cv2.resize(init_img, (mask_img.shape[1],mask_img.shape[0]))
    mask_image = np.zeros_like(mask_img)
    mask_image[np.all(mask_img == [0, 0, 0], axis=-1)] = [1.,1.,1.]
    
    init_img = init_img * (1-mask_image)

    init_img = Image.fromarray(init_img.astype(np.uint8)).convert("RGB")
    # mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(1,-1)*255).convert("RGB")

    generator = torch.Generator("cuda").manual_seed(1234)

    image = pipe(
        instruction, 
        init_img, 
        mask_image, 
        num_inference_steps=50, 
        generator=generator,
        paintingnet_conditioning_scale=1.0
    ).images[0]
    
    output_path = output_dir
    # output_path = os.path.join(output_dir, img_name)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    img_save_path = os.path.join(output_path, output_img_name)
    image.save(img_save_path)
    print(f"saved {img_save_path}")
    
    torch.cuda.empty_cache()
    

