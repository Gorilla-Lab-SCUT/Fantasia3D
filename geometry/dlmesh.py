# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

from render import mesh
from render import render
from render import regularizer
from render import util
from torch.cuda.amp import custom_bwd, custom_fwd 
import numpy as np

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(torch.nn.Module):
    def __init__(self, initial_guess, FLAGS):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        # self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad= True)
        # self.register_parameter('vertex_pos', self.mesh.v_pos)

     

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None,if_normal=False, mode = 'appearance_modeling', if_flip_the_normal = False, if_use_bump = False):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, 
                                  opt_mesh,
                                  target['mvp'],
                                  target['campos'],
                                  lgt,
                                  target['resolution'], 
                                  spp=target['spp'], 
                                  msaa=True,
                                  background= target['background'] ,
                                  bsdf= bsdf,
                                  if_normal=if_normal,
                                  normal_rotate=target['normal_rotate'], 
                                  mode = mode,
                                  if_flip_the_normal = if_flip_the_normal,
                                  if_use_bump = if_use_bump
                                   )

    def tick(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance,  mode, if_flip_the_normal, if_use_bump):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers= self.render(glctx, target, lgt, opt_material, if_normal = if_normal, mode = mode,  if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
        if self.FLAGS.add_directional_text:
            text_embeddings = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.text_z[target['prompt_index']]])
        else:
            text_embeddings = torch.cat([guidance.uncond_z, guidance.text_z])
            
        if iteration <= self.FLAGS.coarse_iter:
            srgb =  buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            # t = torch.randint( guidance.min_step_early, guidance.max_step_early, [1], dtype=torch.long, device='cuda')
            t = torch.randint( guidance.min_step_early, guidance.max_step_early+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
        else:
            srgb =   buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            # t = torch.randint(guidance.min_step_late, guidance.max_step_late, [1], dtype=torch.long, device='cuda')
            t = torch.randint( guidance.min_step_late, guidance.max_step_late+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]

        pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        latents = guidance.encode_imgs(pred_rgb_512)
       
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = guidance.unet(latent_model_input, tt, encoder_hidden_states= text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance.guidance_weight * (noise_pred_text - noise_pred_uncond)
    
        if guidance.sds_weight_strategy == 0:
            w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
        elif guidance.sds_weight_strategy == 1:
            # w = 1 / torch.sqrt(1 - guidance.alphas[t])
            # w = (1 - guidance.alphas[t]) 
            w = 1 / (1 - guidance.alphas[t])
        elif guidance.sds_weight_strategy == 2:
            if iteration <= self.FLAGS.coarse_iter:
                w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
            else:
                w = 1 / (1 - guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad = w* (noise_pred -noise) 
        grad = torch.nan_to_num(grad)
        sds_loss = SpecifyGradient.apply(latents, grad) 
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        # reg_loss = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) *100
     
        return sds_loss, img_loss, reg_loss
    