import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.gn2 = nn.GroupNorm(4, channels)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return x + h

class TimestepEmbedder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t):
        return self.net(t)

class ConditionalResBlock(nn.Module):
    def __init__(self, channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.gn2 = nn.GroupNorm(4, channels)
        self.act = nn.LeakyReLU(0.2)
        self.time_proj = nn.Linear(embed_dim, 2 * channels)
    def forward(self, x, temb):
        h = self.act(self.gn1(self.conv1(x)))
        scale_shift = self.time_proj(temb).view(temb.shape[0], -1, 1, 1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = h * (1 + scale) + shift
        h = self.gn2(self.conv2(self.act(h)))
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(4, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.gn(x)).chunk(3, dim=1)
        q = q.view(b, 1, c, h * w).transpose(2, 3)
        k = k.view(b, 1, c, h * w).transpose(2, 3)
        v = v.view(b, 1, c, h * w).transpose(2, 3)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).view(b, c, h, w)
        return x + self.proj_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, latent):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.conv = nn.Sequential(
            nn.Conv2d(1, latent, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResBlock(latent)
        )
    def forward(self, x):
        """ x --> z """
        z_out = self.conv(x)
        return z_out
    
class Decoder(nn.Module):
    def __init__(self, dim, latent):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.deconv = nn.Sequential(
            ResBlock(latent),
            nn.ConvTranspose2d(latent, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, z):
        """ z --> x """
        x_out = self.deconv(z)
        return x_out

class Autoencoder(nn.Module):
    def __init__(self, dim, latent=16):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.encoder = Encoder(dim, latent)
        self.decoder = Decoder(dim, latent)
    def forward(self, x):
        z = self.encoder(x)
        x_out = self.decoder(z)
        return x_out
    def encode(self, x):
        z_out = self.encoder(x)
        return z_out
    def loss(self, x):
        x_recon = self.forward(x)
        loss = F.mse_loss(x, x_recon)
        return loss
    

class FlowMatcher(nn.Module):
    def __init__(self, dim, latent=16, num_classes=10):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.autoencoder = Autoencoder(dim, latent)
        
        self.time_embed = TimestepEmbedder(128)
        self.class_embed = nn.Embedding(num_classes + 1, 128)
        self.in_proj = nn.Conv2d(self.latent, 128, kernel_size=3, padding=1)
        
        self.block1 = ConditionalResBlock(128, 128)
        self.attn1 = AttentionBlock(128)
        self.block2 = ConditionalResBlock(128, 128)
        self.block3 = ConditionalResBlock(128, 128)
        
        self.out_proj = nn.Conv2d(128, self.latent, kernel_size=3, padding=1)
        
    @property
    def device(self):
        device = next(self.parameters()).device
        return device
        
    def forward(self, zt, t, y):
        """ zt, t, y --> flow velocity """
        assert zt.shape[1] == self.latent
        
        t_flat = t.view(-1, 1)
        temb = self.time_embed(t_flat) + self.class_embed(y)
        
        h = self.in_proj(zt)
        h = self.block1(h, temb)
        h = self.attn1(h)
        h = self.block2(h, temb)
        h = self.block3(h, temb)
        flow_out = self.out_proj(h)
        return flow_out
        
    def _loss(self, z1, y):
        z0 = torch.randn_like(z1)
        flow_target = z1 - z0
        t = torch.rand((z1.shape[0], 1, 1, 1)).to(z1.device)
        zt = (1 - t) * z0 + t * z1
        
        flow_pred = self.forward(zt, t, y)
        loss = F.mse_loss(flow_target, flow_pred)
        return loss
        
    def loss(self, x1, y):
        z1 = self.autoencoder.encode(x1)
        loss = self._loss(z1, y)
        return loss
        
    def generate(self, y, cfg_scale=3.0, num_steps=10):
        N = y.shape[0]
        z0 = torch.randn((N, self.latent, 14, 14)).to(self.device)
        zt = z0
        t = torch.zeros((N, 1, 1, 1)).to(self.device)
        dt = 1 / num_steps
        
        y_null = torch.full_like(y, 10).to(self.device)
        
        for step in range(num_steps):
            zt_double = torch.cat([zt, zt], dim=0)
            t_double = torch.cat([t, t], dim=0)
            y_double = torch.cat([y, y_null], dim=0)
            
            flow_double = self.forward(zt_double, t_double, y_double)
            flow_cond, flow_uncond = flow_double.chunk(2, dim=0)
            
            flow_pred = flow_uncond + cfg_scale * (flow_cond - flow_uncond)
            zt = zt + dt * flow_pred
            t = t + dt
            
        z1 = zt
        x = self.autoencoder.decoder(z1)
        return x