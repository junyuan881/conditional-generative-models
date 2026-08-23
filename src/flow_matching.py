import torch
import torch.nn.functional as F


class FlowMatching:
    """
    Conditional Flow Matching wrapper.

    Model prediction target:
        velocity field v_t

    Training:
        x_0: clean image, range [-1, 1]
        x_1: Gaussian noise
        t ~ Uniform(0, 1)

        x_t = (1 - t) * x_0 + t * x_1
        target_v = x_1 - x_0

        model(x_t, t, cond) predicts target_v
        loss = MSE(pred_v, target_v)

    Sampling:
        Start from x_1 ~ N(0, I)
        Integrate ODE backward from t = 1 to t = 0:

        x_{t-dt} = x_t - v_theta(x_t, t, cond) * dt
    """

    def __init__(
        self,
        num_steps=100,
        time_scale=1000,
        device="cuda",
    ):
        self.num_steps = num_steps
        self.time_scale = time_scale
        self.device = device

    def training_loss(self, model, x_start, cond):
        """
        Compute Flow Matching loss.

        Args:
            model: ConditionalUNet
            x_start: clean image x_0, shape (B, 3, H, W), range [-1, 1]
            cond: multi-hot label, shape (B, 24)

        Return:
            loss
        """
        batch_size = x_start.shape[0]

        x_0 = x_start
        x_1 = torch.randn_like(x_0)

        t = torch.rand(
            batch_size,
            device=self.device,
            dtype=torch.float32,
        )

        t_view = t.view(batch_size, 1, 1, 1)

        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        target_v = x_1 - x_0

        # model.py 的 time embedding 原本設計給 DDPM timestep，
        # 所以這裡把 continuous t in [0,1] scale 到 [0,1000]。
        t_model = (t * self.time_scale).long()

        pred_v = model(
            x_t,
            t_model,
            cond,
        )

        loss = F.mse_loss(pred_v, target_v)

        return loss

    @torch.no_grad()
    def sample(
        self,
        model,
        cond,
        image_size=64,
        img_channels=3,
    ):
        """
        Generate images by solving the reverse ODE.

        Args:
            model: ConditionalUNet
            cond: shape (B, 24)

        Return:
            generated images, shape (B, 3, H, W), range roughly [-1, 1]
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

        dt = 1.0 / self.num_steps

        for step in range(self.num_steps):
            # t: 1 -> 0
            t_cont = 1.0 - step / self.num_steps

            t_model = torch.full(
                (batch_size,),
                int(t_cont * self.time_scale),
                device=self.device,
                dtype=torch.long,
            )

            pred_v = model(x, t_model, cond)

            # backward integration: x_{t-dt} = x_t - v * dt
            x = x - pred_v * dt

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
        Generate one image and save intermediate flow process.

        Args:
            cond: shape (1, 24)
            save_steps: step indices to save.
                        Example:
                        [0, 10, 25, 40, 60, 80, 90, 99]

        Return:
            process_images: list of tensors, each shape (1, 3, H, W)
        """
        model.eval()

        if cond.shape[0] != 1:
            raise ValueError("sample_with_process expects cond batch size = 1")

        cond = cond.to(self.device)

        if save_steps is None:
            save_steps = [
                0,
                int(self.num_steps * 0.1),
                int(self.num_steps * 0.25),
                int(self.num_steps * 0.4),
                int(self.num_steps * 0.6),
                int(self.num_steps * 0.8),
                int(self.num_steps * 0.9),
                self.num_steps - 1,
            ]

        save_steps = set(save_steps)

        x = torch.randn(
            1,
            img_channels,
            image_size,
            image_size,
            device=self.device,
        )

        process_images = []

        dt = 1.0 / self.num_steps

        for step in range(self.num_steps):
            t_cont = 1.0 - step / self.num_steps

            t_model = torch.full(
                (1,),
                int(t_cont * self.time_scale),
                device=self.device,
                dtype=torch.long,
            )

            pred_v = model(x, t_model, cond)
            x = x - pred_v * dt

            if step in save_steps:
                process_images.append(torch.clamp(x.detach().cpu(), -1.0, 1.0))

        return process_images


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

    fm = FlowMatching(
        num_steps=100,
        time_scale=1000,
        device=device,
    )

    batch_size = 4

    x_start = torch.randn(batch_size, 3, 64, 64).to(device)
    x_start = torch.clamp(x_start, -1.0, 1.0)

    cond = torch.zeros(batch_size, 24).to(device)
    cond[:, 0] = 1
    cond[:, 5] = 1

    loss = fm.training_loss(
        model=model,
        x_start=x_start,
        cond=cond,
    )

    print("Flow Matching training loss:", loss.item())

    samples = fm.sample(
        model=model,
        cond=cond,
        image_size=64,
        img_channels=3,
    )

    print("Sample shape:", samples.shape)

    process = fm.sample_with_process(
        model=model,
        cond=cond[:1],
        image_size=64,
        img_channels=3,
    )

    print("Number of process images:", len(process))
    print("Each process image shape:", process[0].shape)