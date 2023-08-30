# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import light
import numpy as np
import cv2
import itertools
from tqdm import tqdm

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material,
        bsdf,
        if_normal,
        normal_rotate,
        mode,
        if_flip_the_normal,
        if_use_bump
    ):

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    if 'kd_ks_normal' in material and mode == 'appearance_modeling':
        # Combined texture, used for MLPs because lookups are expensive
        # all_tex_jitter = material['kd_ks_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
        all_tex = material['kd_ks_normal'].sample(gb_pos)
        assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
        kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]
        # Compute albedo (kd) gradient, used for material regularizer
        # kd_grad    = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 3
    elif mode == 'geometry_modeling':
        kd =  torch.ones_like(gb_pos, dtype=torch.float32, device='cuda') *0.5 
    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])  #[1,512,512,1]
    kd = kd[..., 0:3]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    # if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
    # 
    if if_use_bump == False or mode == 'geometry_modeling':
        perturbed_nrm = None
    if if_normal:
        bsdf = 'normal'
    #produces a final normal used for shading  [B, 512, 512, 3]
    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)
    # gb_normal1 = gb_normal 
    gb_normal1 = gb_normal @ normal_rotate[:,None,...] # Randomly rotate the normals to change the color gamut of nomral at the same angle. I found this help to deform the shape
    
    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material or bsdf is not None, "Material must specify a BSDF type"
    bsdf = material['bsdf'] if bsdf is None else bsdf
    
   
    if bsdf == 'pbr':
        if mode == 'geometry_modeling':
            shaded_col = kd * ru.lambert(gb_normal, util.safe_normalize(view_pos - gb_pos)) * 5
        elif mode == 'appearance_modeling':
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular = True)
        else:
            assert False, "Invalid mode type"
    elif bsdf == 'diffuse':
        if mode == 'geometry_modeling':
            shaded_col = kd * ru.lambert(gb_normal, util.safe_normalize(view_pos - gb_pos)) * 5
        elif mode == 'appearance_modeling':
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular = False)
        else:
            assert False, "Invalid mode type"
    elif bsdf == 'normal':
        shaded_col = gb_normal1
        if if_flip_the_normal:
            shaded_col[...,0][shaded_col[...,0]>0]= shaded_col[...,0][shaded_col[...,0]>0]*(-1) # Flip the x-axis positive half-axis of Normal. I found this process helps to alleviate the Janus problem.
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf
    
    buffers = {
        'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
        # 'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1),
        # 'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1)  #it is similar to a simple ambient occlusion term and does not account for directional visibility
    }
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        if_normal,
        normal_rotate,
        mode,
        if_flip_the_normal,
        if_use_bump
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast    #[u,v,z,triangle_id]
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int()) 
    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :] 
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :] 
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :] 
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0)) 
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3) #[10688,3] 三角面片每个顶点的法线的索引
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())
    # Compute tangent space
    # 
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    if if_use_bump == False or mode == 'geometry_modeling':
        gb_tangent = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    else:
        assert mesh.v_nrm is not None and mesh.v_tng is not None
        gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents
    # 
    # Texture coordinate
    # assert mesh.v_tex is not None
    # gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)
    gb_texc, gb_texc_deriv = 0, 0

    ################################################################################
    # Shade
    ################################################################################

    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, lgt, mesh.material, bsdf,if_normal,normal_rotate, mode, if_flip_the_normal, if_use_bump)
        
    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None, 
        bsdf        = None,
        if_normal = False,
        normal_rotate = None,
        mode = 'geometry_modeling',
        if_flip_the_normal = False,
        if_use_bump = False
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:] # [1,512,512,1] 保留有物体的像素的alpha为1，没有物体的像素alpha为0
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha) #[1,512,512,4] 最后一个通道是alpha通道，若像素有物体则为1，无物体则为0  outi=starti+weighti×(endi−starti)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)
    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)
    v_pos_clip = v_pos_clip.cuda()
    mesh.t_pos_idx = mesh.t_pos_idx.cuda()
    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf, if_normal, normal_rotate, mode, if_flip_the_normal, if_use_bump), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers


# ==============================================================================================
#  Render UVs
# ==============================================================================================

def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    
    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)

def uv_padding(image, hole_mask, padding = 2, uv_padding_block = 4):
        uv_padding_size = padding
        image1 = (image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        hole_mask = (hole_mask[0].detach().cpu().numpy() * 255).astype(np.uint8)
        block = uv_padding_block
        res = image1.shape[0]
        chunk = res // block 
        inpaint_image = np.zeros_like(image1)
        prods = list(itertools.product(range(block), range(block)))
        for (i, j) in tqdm(prods):
            patch = cv2.inpaint(
                image1[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk],
                hole_mask[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk],
                uv_padding_size,
                cv2.INPAINT_TELEA,
            )
            inpaint_image[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk] = patch
        inpaint_image = inpaint_image / 255.0
        return torch.from_numpy(inpaint_image).to(image)
    
def render_uv1(ctx, mesh, resolution, mlp_texture, uv_padding_block):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)
    hole_mask = ~(rast[..., 3:] > 0)
    
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    
    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    print(f'[INFO] UV padding for Kd...')
    kd = uv_padding(all_tex[..., :-6] , hole_mask, uv_padding_block)
    print(f'[INFO] UV padding for Ks...')
    ks = uv_padding(all_tex[..., -6:-3], hole_mask, uv_padding_block)
    print(f'[INFO] UV padding for perturbed normal...')
    perturbed_nrm = uv_padding(util.safe_normalize(all_tex[..., -3:]), hole_mask, uv_padding_block)
    
    # kd = all_tex[..., :-6] 
    # ks = all_tex[..., -6:-3]
    # perturbed_nrm = util.safe_normalize(all_tex[..., -3:])
    return (rast[..., -1:] > 0).float(), kd, ks, perturbed_nrm 
