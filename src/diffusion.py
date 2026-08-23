import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from improved DDPM.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5
    ) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-4, 0.999)

    return betas


def extract(a, t, x_shape):
    """
    Extract coefficients according to timestep t.

    a: shape (T,)
    t: shape (B,)
    return: shape (B, 1, 1, 1)
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DDPM:
    """
    DDPM wrapper.

    Model prediction target:
        epsilon prediction

    Training:
        x_0 -> q_sample -> x_t
        model(x_t, t, cond) predicts noise epsilon
        loss = MSE(pred_noise, true_noise)

    Sampling:
        x_T ~ N(0, I)
        reverse process from T-1 to 0
    """

    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type="linear",
        device="cuda",
    ):
        self.timesteps = timesteps
        self.device = device

        if schedule_type == "linear":
            betas = linear_beta_schedule(
                timesteps=timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
            )
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps=timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.betas = betas.to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1],
            pad=(1, 0),
            value=1.0,
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process.

        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_start: clean image x_0, shape (B, 3, H, W), range [-1, 1]
            t: timestep, shape (B,)
            noise: random Gaussian noise, same shape as x_start

        Return:
            noisy image x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape,
        )

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def predict_x0_from_noise(self, x_t, t, noise):
        """
        Estimate x_0 from x_t and predicted noise.
        """
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod,
            t,
            x_t.shape,
        )

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )

        x0 = (
            x_t - sqrt_one_minus_alphas_cumprod_t * noise
        ) / sqrt_alphas_cumprod_t

        return x0

    def p_sample(self, model, x_t, t, cond):
        """
        One reverse denoising step.

        Args:
            model: ConditionalUNet
            x_t: noisy image at timestep t
            t: shape (B,)
            cond: shape (B, 24)

        Return:
            x_{t-1}
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas,
            t,
            x_t.shape,
        )

        pred_noise = model(x_t, t, cond)

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = extract(
            self.posterior_variance,
            t,
            x_t.shape,
        )

        noise = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().reshape(
            x_t.shape[0],
            *((1,) * (len(x_t.shape) - 1)),
        )

        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model,
        cond,
        image_size=64,
        img_channels=3,
    ):
        """
        Generate images from pure Gaussian noise.

        Args:
            cond: shape (B, 24)

        Return:
            generated images, shape (B, 3, image_size, image_size), range roughly [-1, 1]
        """
        model.eval()

        batch_size = cond.shape[0]

        x = torch.randn(
            batch_size,
            img_channels,
            image_size,
            image_size,
            device=self.device,
        )

        cond = cond.to(self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full(
                (batch_size,),
                i,
                device=self.device,
                dtype=torch.long,
            )
            x = self.p_sample(model, x, t, cond)

        x = torch.clamp(x, -1.0, 1.0)

        return x

    @torch.no_grad()
    def sample_with_process(
        self,
        model,
        cond,
        image_size=64,
        img_channels=3,
        save_steps=None,
    ):
        """
        Generate one image and save intermediate denoising results.

        Used for denoising process grid.

        Args:
            cond: shape (1, 24)
            save_steps: list of timesteps to save.
                        Example:
                        [999, 800, 600, 400, 200, 100, 50, 0]

        Return:
            process_images: list of tensors, each shape (1, 3, H, W)
        """
        model.eval()

        if cond.shape[0] != 1:
            raise ValueError("sample_with_process expects cond batch size = 1")

        if save_steps is None:
            save_steps = [
                self.timesteps - 1,
                int(self.timesteps * 0.8),
                int(self.timesteps * 0.6),
                int(self.timesteps * 0.4),
                int(self.timesteps * 0.2),
                int(self.timesteps * 0.1),
                int(self.timesteps * 0.05),
                0,
            ]

        save_steps = set(save_steps)

        x = torch.randn(
            1,
            img_channels,
            image_size,
            image_size,
            device=self.device,
        )

        cond = cond.to(self.device)

        process_images = []

        for i in reversed(range(self.timesteps)):
            t = torch.full(
                (1,),
                i,
                device=self.device,
                dtype=torch.long,
            )

            x = self.p_sample(model, x, t, cond)

            if i in save_steps:
                process_images.append(torch.clamp(x.detach().cpu(), -1.0, 1.0))

        return process_images
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        cond,
        image_size=64,
        img_channels=3,
        ddim_steps=50,
        eta=0.0,
    ):
        """
        DDIM sampling using the trained DDPM noise prediction model.
        """
        model.eval()

        batch_size = cond.shape[0]
        cond = cond.to(self.device)

        x = torch.randn(
            batch_size,
            img_channels,
            image_size,
            image_size,
            device=self.device,
        )

        times = torch.linspace(
            self.timesteps - 1,
            0,
            steps=ddim_steps,
            device=self.device,
        ).long()

        for i in range(len(times)):
            t = times[i]
            t_batch = torch.full(
                (batch_size,),
                t.item(),
                device=self.device,
                dtype=torch.long,
            )

            pred_noise = model(x, t_batch, cond)

            alpha_bar_t = extract(
                self.alphas_cumprod,
                t_batch,
                x.shape,
            )

            x0_pred = (
                x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i == len(times) - 1:
                x = x0_pred
                break

            t_prev = times[i + 1]
            t_prev_batch = torch.full(
                (batch_size,),
                t_prev.item(),
                device=self.device,
                dtype=torch.long,
            )

            alpha_bar_prev = extract(
                self.alphas_cumprod,
                t_prev_batch,
                x.shape,
            )

            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)
            ) * pred_noise

            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + dir_xt
                + sigma_t * noise
            )

        x = torch.clamp(x, -1.0, 1.0)

        return x
    
    @torch.no_grad()
    def ddim_sample_with_process(
        self,
        model,
        cond,
        image_size=64,
        img_channels=3,
        ddim_steps=50,
        eta=0.0,
        save_indices=None,
    ):
        """
        DDIM sampling and save intermediate process images.
        """
        model.eval()

        if cond.shape[0] != 1:
            raise ValueError("ddim_sample_with_process expects cond batch size = 1")

        cond = cond.to(self.device)

        x = torch.randn(
            1,
            img_channels,
            image_size,
            image_size,
            device=self.device,
        )

        times = torch.linspace(
            self.timesteps - 1,
            0,
            steps=ddim_steps,
            device=self.device,
        ).long()

        if save_indices is None:
            save_indices = torch.linspace(
                0,
                ddim_steps - 1,
                steps=8,
            ).long().tolist()

        save_indices = set(save_indices)

        process_images = []

        for i in range(len(times)):
            t = times[i]
            t_batch = torch.full(
                (1,),
                t.item(),
                device=self.device,
                dtype=torch.long,
            )

            pred_noise = model(x, t_batch, cond)

            alpha_bar_t = extract(
                self.alphas_cumprod,
                t_batch,
                x.shape,
            )

            x0_pred = (
                x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i == len(times) - 1:
                x = x0_pred
            else:
                t_prev = times[i + 1]
                t_prev_batch = torch.full(
                    (1,),
                    t_prev.item(),
                    device=self.device,
                    dtype=torch.long,
                )

                alpha_bar_prev = extract(
                    self.alphas_cumprod,
                    t_prev_batch,
                    x.shape,
                )

                sigma_t = eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                )

                dir_xt = torch.sqrt(
                    torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)
                ) * pred_noise

                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

                x = (
                    torch.sqrt(alpha_bar_prev) * x0_pred
                    + dir_xt
                    + sigma_t * noise
                )

            if i in save_indices:
                process_images.append(torch.clamp(x.detach().cpu(), -1.0, 1.0))

        return process_images
    def training_loss(self, model, x_start, cond):
        """
        Compute DDPM noise prediction loss.

        Args:
            x_start: clean image, shape (B, 3, H, W), range [-1, 1]
            cond: multi-hot label, shape (B, 24)

        Return:
            loss
        """
        batch_size = x_start.shape[0]

        t = torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        )

        pred_noise = model(
            x_noisy,
            t,
            cond,
        )

        loss = F.mse_loss(pred_noise, noise)

        return loss


if __name__ == "__main__":
    from model import ConditionalUNet

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalUNet(
        img_channels=3,
        num_classes=24,
        base_channels=64,
        time_emb_dim=256,
        cond_emb_dim=256,
    ).to(device)

    ddpm = DDPM(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type="linear",
        device=device,
    )

    batch_size = 4
    x_start = torch.randn(batch_size, 3, 64, 64).to(device)
    cond = torch.zeros(batch_size, 24).to(device)
    cond[:, 0] = 1
    cond[:, 5] = 1

    loss = ddpm.training_loss(model, x_start, cond)
    print("Training loss:", loss.item())

    with torch.no_grad():
        samples = ddpm.sample(
            model=model,
            cond=cond,
            image_size=64,
            img_channels=3,
        )

    print("Sample shape:", samples.shape)

    process = ddpm.sample_with_process(
        model=model,
        cond=cond[:1],
        image_size=64,
        img_channels=3,
    )

    print("Number of process images:", len(process))
    print("Each process image shape:", process[0].shape)