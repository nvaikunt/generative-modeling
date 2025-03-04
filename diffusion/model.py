import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        # TODO 3.1: compute the cumulative products for current and previous timesteps
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1]).cuda(), self.alphas_cumprod[:-1]])

        # TODO 3.1: pre-compute values needed for forward process
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1 / torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = torch.sqrt(1 - self.alphas_cumprod) / torch.sqrt(self.alphas_cumprod)

        # TODO 3.1: compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        denom = 1 - self.alphas_cumprod
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_cumprod_prev) * self.betas) / denom
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = (torch.sqrt(alphas) * (1 - self.alphas_cumprod_prev)) / denom

        # TODO 3.1: compute posterior variance
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = ((1 - self.alphas_cumprod_prev) / denom) * self.betas
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # TODO 3.1: Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0
        # hint: can use extract function from utils.py

        x_coeff_1 = extract(self.posterior_mean_coef1, t, x_0.size())
        x_coeff_2 = extract(self.posterior_mean_coef2, t, x_t.size())

        posterior_mean = x_coeff_1 * x_0 + x_coeff_2 * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.size())
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.size())

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # TODO 3.1: given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py.
        # clamp x_0 to [-1, 1]
        pred_noise = self.model(x_t, t)

        coeff_1 = extract(self.x_0_pred_coef_1, t, x_t.size())
        coeff_2 = extract(self.x_0_pred_coef_2, t, pred_noise.size())
        x_0 = torch.clamp(coeff_1 * x_t - coeff_2 * pred_noise, min=-1, max=1)
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO 3.1: given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?
        pred_noise, x_0 = self.model_predictions(x, t)
        posterior_mean, posterior_variance, _ = self.get_posterior_parameters(x_0, x, t)

        z = torch.normal(mean=0., std = 1., size=posterior_mean.size()).cuda()

        posterior_std = torch.sqrt(posterior_variance)
        pred_img = posterior_mean + posterior_std * z
        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        # TODO 3.2: Generate a list of times to sample from.
        return list(reversed(torch.linspace(1 , total_timesteps - 1, sampling_timesteps).tolist()))

    def get_time_pairs(self, times):
        # TODO 3.2: Generate a list of adjacent time pairs to sample from.
        pairs = []
        for i in range(1, len(times)):
            pairs.append((times[i - 1], times[i]))
        return pairs

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # TODO 3.2: Compute the output image for a single step of the DDIM sampling process.

        # predict x_0 and the additive noise for tau_i
       
        batched_taus = torch.full((img.shape[0],), tau_i, device=device, dtype=torch.long)
        pred_noise, x_0 = model_predictions(img, batched_taus)

        # extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}
        alpha_tau_i = alphas_cumprod[int(tau_i)]
        alpha_tau_isub1 = alphas_cumprod[int(tau_isub1)]
        beta_tau_isub1 = self.betas[int(tau_isub1)]
        beta_hat_tau_i = ((1 - alpha_tau_isub1) / (1 - alpha_tau_i)) * beta_tau_isub1

        # compute \sigma_{\tau_{i}}
        variance_tau_i = eta * beta_hat_tau_i

        # compute the coefficient of \epsilon_{\tau_{i}}
        epsilon_coeff = torch.sqrt(1 - alpha_tau_isub1 - variance_tau_i)
        # sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: use the reparameterization trick
        mu_tau_i = torch.sqrt(alpha_tau_isub1) * x_0 + epsilon_coeff * pred_noise
        std_tau_i = torch.sqrt(variance_tau_i)
        if tau_i > 0: 
            z = torch.normal(mean=0., std=1., size=mu_tau_i.size()).cuda()
        else: 
            z = torch.zeros_like(mu_tau_i)

        img = mu_tau_i + std_tau_i * z
        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)
        
        # print(time_pairs)
        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        #TODO 3.3: fill out based on the sample function above
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(shape)
        img = sample_fn(shape, z) * 256
        return img
