from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler, AutoencoderKL, UNet2DConditionAdaINModel
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image
import json
import glob
import os

img_folder_dir = "data/mask_test/images"
caption_path = "data/mask_test/test_description.json"
output_dir = "output/brushnet_adain/test_description"
os.makedirs(output_dir, exist_ok=True)
base_model_path = "pretrained/sd_checkpoints/Realistic_Vision_V6.0_B1_noVAE"
brushnet_path = "runs/logs/brushnet_adain_segmentationmask/checkpoint-300000/brushnet"
vae_model_path = "pretrained/sd_checkpoints/sd-vae-ft-mse"

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float32)
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float32)
unet = UNet2DConditionAdaINModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=torch.float32)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, vae=vae, unet=unet, torch_dtype=torch.float32, low_cpu_mem_usage=False
)

resize_512 = False

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


img_folder_list = sorted(os.listdir(img_folder_dir))

description = json.load(open(caption_path))
for img_name in img_folder_list:
    if not img_name in description.keys():
        continue
    img_folder_path =  os.path.join(img_folder_dir, img_name)
    init_img_path = glob.glob(os.path.join(img_folder_path, img_name + ".*"))[0]
    mask_img_paths = glob.glob(os.path.join(img_folder_path, img_name + "*mask*"))
    for mask_img_path in mask_img_paths:
        mask_img_name = os.path.basename(mask_img_path)
        # output_img_name = mask_img_name.replace("mask", "output")
        caption =  description[img_name][mask_img_name]
        
        init_img = cv2.imread(init_img_path)
        init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_img_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        if resize_512:
            mask_img = cv2.resize(mask_img, (512,512))
            
        if mask_img.shape[0] > 1024 or mask_img.shape[1] > 1024:
            mask_img = cv2.resize(mask_img, (mask_img.shape[1]//2,mask_img.shape[0]//2))
        
        if mask_img.shape[0] > 1700 or mask_img.shape[1] > 1700:
            mask_img = cv2.resize(mask_img, (mask_img.shape[1]//2,mask_img.shape[0]//2))
        init_img = cv2.resize(init_img, (mask_img.shape[1],mask_img.shape[0]))
        mask_image = np.zeros_like(mask_img)
        mask_image[np.all(mask_img == [0, 0, 0], axis=-1)] = [1.,1.,1.]
        
        masked_init_img = init_img * (1-mask_image)

        init_img = Image.fromarray(init_img.astype(np.uint8)).convert("RGB")
        masked_init_img = Image.fromarray(masked_init_img.astype(np.uint8)).convert("RGB")
        # mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(1,-1)*255).convert("RGB")

        generator = torch.Generator("cuda").manual_seed(1234)

        image = pipe(
            caption, 
            masked_init_img, 
            mask_image, 
            num_inference_steps=50, 
            generator=generator,
            paintingnet_conditioning_scale=1.0
        ).images[0]
        
        output_path = output_dir
        # output_path = os.path.join(output_dir, img_name)
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        output_name = mask_img_name.replace("mask", "output")
        init_img_name = mask_img_name.split("_")[0] + '.png'
        img_save_path = os.path.join(output_path, f"{output_name}")
        mask_img_save_path = os.path.join(output_path, f"{mask_img_name}")
        init_img_save_path =  os.path.join(output_path, f"{init_img_name}")
        image.save(img_save_path)
        mask_image.save(mask_img_save_path)
        init_img.save(init_img_save_path)
        print(f"saved {img_save_path}")
        
        torch.cuda.empty_cache()
        

