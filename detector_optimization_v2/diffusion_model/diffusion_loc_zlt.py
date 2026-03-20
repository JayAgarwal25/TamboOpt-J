# diffusion_planes_single.py - Single Head (Diffusion Only) - STANDARDIZED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(v, t, x_shape):
    """Extract values from v at indices t and reshape for broadcasting"""
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    """
    Trainer for single-head plane-based diffusion model
    
    IMPORTANT: Works with STANDARDIZED data (zero mean, unit variance)
    No clipping is applied to maintain the unbounded nature of standardized data.
    """
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        
        # Create beta schedule
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, p_energy, class_id, sin_zenith, cos_zenith, 
                sin_azimuth, cos_azimuth, plane_idx, past_plane):
        """
        Training forward pass with diffusion objective only
        
        Args:
            x_0: (B, 3, H, W) - clean target plane (STANDARDIZED)
            [other args]: conditioning parameters
        
        Returns:
            loss: scalar tensor - MSE loss between predicted and true noise
        """
        # Sample random timestep for each sample in batch
        t = torch.randint(self.T, (x_0.shape[0],), device=x_0.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Create noisy version of x_0
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        # Predict noise
        predicted_noise = self.model(
            x_t, t, p_energy, class_id, sin_zenith, cos_zenith,
            sin_azimuth, cos_azimuth, plane_idx, past_plane
        )
        
        # Compute diffusion loss (MSE between predicted and true noise)
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        return loss


class GaussianDiffusionSampler(nn.Module):
    """
    Sampler for generating planes
    
    IMPORTANT: Generates STANDARDIZED data (no clipping applied)
    """
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w  # classifier-free guidance weight
        
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        return (extract(self.coeff1, t, x_t.shape) * x_t - 
                extract(self.coeff2, t, x_t.shape) * eps)

    def p_mean_variance(self, x_t, t, p_energy, class_id, sin_zenith, cos_zenith,
                       sin_azimuth, cos_azimuth, plane_idx, past_plane):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        
        # Predict noise with conditioning
        eps = self.model(x_t, t, p_energy, class_id, sin_zenith, cos_zenith,
                        sin_azimuth, cos_azimuth, plane_idx, past_plane)
        
        # Classifier-free guidance
        if self.w > 0:
            # Create null conditioning for ALL parameters
            null_p_energy = torch.zeros_like(p_energy)
            null_class_id = torch.zeros_like(class_id)
            null_sin_zenith = torch.zeros_like(sin_zenith)
            null_cos_zenith = torch.zeros_like(cos_zenith)
            null_sin_azimuth = torch.zeros_like(sin_azimuth)
            null_cos_azimuth = torch.zeros_like(cos_azimuth)
            
            nonEps = self.model(x_t, t, null_p_energy, null_class_id, 
                               null_sin_zenith, null_cos_zenith,
                               null_sin_azimuth, null_cos_azimuth, 
                               plane_idx, past_plane)
            eps = (1. + self.w) * eps - self.w * nonEps
        
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        
        return xt_prev_mean, var

    def forward(self, x_T, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth, plane_idx, past_plane):
        """
        Generate a plane from noise
        
        Returns:
            plane: (B, 3, H, W) - generated plane (STANDARDIZED, unbounded)
        """
        x_t = x_T
        
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones((x_T.shape[0],), dtype=torch.long) * time_step
            
            mean, var = self.p_mean_variance(
                x_t, t, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth, plane_idx, past_plane
            )
            
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        
        # NO CLIPPING - return standardized (unbounded) data
        x_0 = x_t
        return x_0


class DDIMSamplerPlanes(nn.Module):
    """
    DDIM sampler for faster generation
    
    IMPORTANT: Generates STANDARDIZED data (no clipping applied)
    """
    def __init__(self, model, beta_1, beta_T, T, eta=0.0, ddim_steps=50, w=0.):
        super().__init__()
        self.model = model
        self.T = T
        self.eta = eta
        self.ddim_steps = ddim_steps
        self.w = w
        
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", 
                           torch.sqrt(1 - self.alphas_cumprod))
        
        self.ddim_timesteps = self.get_ddim_timesteps()

    def get_ddim_timesteps(self):
        return torch.linspace(0, self.T - 1, self.ddim_steps).long()

    def forward(self, x, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth, plane_idx, past_plane):
        """
        DDIM sampling for faster generation
        
        Returns:
            plane: (B, 3, H, W) - generated plane (STANDARDIZED, unbounded)
        """
        for i in reversed(range(len(self.ddim_timesteps))):
            t = int(self.ddim_timesteps[i].item())
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            
            # Predict noise
            eps = self.model(
                x, t_tensor, p_energy, class_id, sin_zenith, 
                cos_zenith, sin_azimuth, cos_azimuth, plane_idx, 
                past_plane
            )
            
            # Classifier-free guidance
            if self.w > 0:
                null_p_energy = torch.zeros_like(p_energy)
                null_class_id = torch.zeros_like(class_id)
                null_sin_zenith = torch.zeros_like(sin_zenith)
                null_cos_zenith = torch.zeros_like(cos_zenith)
                null_sin_azimuth = torch.zeros_like(sin_azimuth)
                null_cos_azimuth = torch.zeros_like(cos_azimuth)
                
                nonEps = self.model(
                    x, t_tensor, null_p_energy, null_class_id,
                    null_sin_zenith, null_cos_zenith,
                    null_sin_azimuth, null_cos_azimuth,
                    plane_idx, past_plane
                )
                eps = (1. + self.w) * eps - self.w * nonEps
            
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Predict x0
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
            
            if i > 0:
                t_prev = int(self.ddim_timesteps[i - 1].item())
                alpha_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_prev = torch.tensor(1.0, device=x.device)
            
            sigma = self.eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            noise = torch.randn_like(x) if self.eta > 0 else 0
            
            x = (torch.sqrt(alpha_prev) * x0_pred + 
                 torch.sqrt(1 - alpha_prev - sigma**2) * eps + 
                 sigma * noise)
        
        # NO CLIPPING - return standardized (unbounded) data
        return x


class AutoregressivePlaneGenerator:
    """
    Generates all 24 planes autoregressively
    
    IMPORTANT: Generates STANDARDIZED data (unbounded)
    """
    def __init__(self, sampler):
        self.sampler = sampler
    
    def generate_all_planes(self, batch_size, p_energy, class_id, sin_zenith, 
                          cos_zenith, sin_azimuth, cos_azimuth, device, img_size=64):
        """
        Generate all 24 planes autoregressively
        
        Returns:
            all_planes: (batch_size, 24, 3, img_size, img_size) - STANDARDIZED
        """
        all_planes = []
        
        # Start with zeros for "past plane" for first plane
        past_plane = torch.zeros(batch_size, 3, img_size, img_size, device=device)
        
        for plane_idx in range(24):
            # Create plane index tensor
            plane_idx_tensor = torch.full((batch_size,), plane_idx, 
                                         device=device, dtype=torch.long)
            
            # Sample noise
            x_T = torch.randn(batch_size, 3, img_size, img_size, device=device)
            
            # Generate current plane
            current_plane = self.sampler(
                x_T, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth, plane_idx_tensor, past_plane
            )
            
            all_planes.append(current_plane.unsqueeze(1))  # (B, 1, 3, H, W)
            
            # Update past plane for next iteration
            past_plane = current_plane
        
        # Stack all planes: (B, 24, 3, H, W) - STANDARDIZED
        all_planes = torch.cat(all_planes, dim=1)
        
        return all_planes
