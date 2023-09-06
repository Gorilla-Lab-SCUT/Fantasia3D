import torch
from tqdm import tqdm
import tinycudann as tcnn
import numpy as np

# MLP + Positional Encoding
class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()
        self.net.apply(self._init_weights)
        
    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

class Decoder(torch.nn.Module):
    def __init__(self, input_dims = 3, internal_dims = 128, output_dims = 4, hidden = 2, multires = 2, AABB=None, mesh_scale = 2.1):
        super().__init__()
        self.mesh_scale = mesh_scale
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
        self.AABB= AABB
        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
	    }
        gradient_scaling = 1.0 #128
        self.encoder = tcnn.Encoding(3, enc_cfg)
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_output_dims" : 4,
            "n_hidden_layers" : 2,
            "n_neurons" : 32
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
     
    def forward(self, p):
        _texc = (p.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        p_enc = self.encoder(_texc.contiguous())
        out = self.net(p_enc)
        return out

    def pre_train_ellipsoid(self, it, scene_and_vertices):
        if it% 100 ==0:
            print (f"Initialize SDF; it: {it}")
        loss_fn = torch.nn.MSELoss()
        scene = scene_and_vertices[0]
        points_surface = scene_and_vertices[1].astype(np.float32)
        points_surface_disturbed = points_surface + np.random.normal(loc=0.0, scale=0.05, size=points_surface.shape).astype(np.float32)   
        point_rand = (np.random.rand(3000,3).astype(np.float32)-0.5)* self.mesh_scale
        query_point = np.concatenate((points_surface, points_surface_disturbed, point_rand))
        signed_distance = scene.compute_signed_distance(query_point)
        ref_value = torch.from_numpy(signed_distance.numpy()).float().cuda()
        query_point = torch.from_numpy(query_point).float().cuda()
        output = self(query_point)
        
        loss = loss_fn(output[...,0], ref_value)
    
        return loss

