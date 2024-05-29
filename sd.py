import torch  #, tomesd
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline
)
logging.set_verbosity_error()
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, 
                 device, 
                 mode='geometry', 
                 text= '', 
                 add_directional_text= False, 
                 batch = 1, 
                 guidance_weight = 100, 
                 sds_weight_strategy = 0,
                 early_time_step_range = [0.02, 0.5],
                 late_time_step_range = [0.02, 0.5],
                 sd_version = '2.1',
                 negative_text = ''):
        super().__init__()

        self.device = device
        self.mode = mode
        self.text= text
        self.add_directional_text = add_directional_text
        self.batch = batch 
        self.sd_version = sd_version
        print(f'[INFO] loading stable diffusion...')
        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        if self.mode == 'geometry_modeling':
            negative_text = "low quality"
        if self.mode == 'appearance_modeling':
            negative_text = "shadow, oversaturated, low quality, unrealistic" 
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae",torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer",torch_dtype=torch.float16 )
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",torch_dtype=torch.float16 ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",torch_dtype=torch.float16 ).to(self.device)
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        self.negative_text = negative_text
        if add_directional_text:
            self.text_z = []
            self.uncond_z = []
            for d in ['front', 'side', 'back', 'side']:
                text = f"{self.text}, {d} view"
                # text = f"{d} view of {self.text}"
                negative_text = f"{self.negative_text}"
                # if d == 'back': negative_text += "face"
                text_z = self.get_text_embeds([text], batch = 1)
                uncond_z =self.get_uncond_embeds([negative_text], batch = 1)
                self.text_z.append(text_z)
                self.uncond_z.append(uncond_z)
            self.text_z = torch.cat(self.text_z)
            self.uncond_z = torch.cat(self.uncond_z)
        else: 
            self.text_z = self.get_text_embeds([self.text], batch = self.batch)
            self.uncond_z =self.get_uncond_embeds([self.negative_text], batch = self.batch)
        del self.text_encoder
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=torch.float16)
        # self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler", torch_dtype=torch.float16)
        # self.scheduler = DDPMScheduler.from_config(model_key, subfolder="scheduler", torch_dtype=torch.float16)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step_early = int(self.num_train_timesteps * early_time_step_range[0])
        self.max_step_early = int(self.num_train_timesteps * early_time_step_range[1])
        self.min_step_late = int(self.num_train_timesteps *  late_time_step_range[0])
        self.max_step_late = int(self.num_train_timesteps *  late_time_step_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.guidance_weight = guidance_weight
        self.sds_weight_strategy = sds_weight_strategy
        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, batch=1):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        if batch > 1:
            text_embeddings = text_embeddings.repeat(batch, 1, 1)
      
        return text_embeddings
    
    def get_uncond_embeds(self, negative_prompt, batch):
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    
        if batch > 1:
            uncond_embeddings = uncond_embeddings.repeat(batch, 1, 1)
        return uncond_embeddings

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        if self.mode == 'appearance_modeling':
            
            imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs



