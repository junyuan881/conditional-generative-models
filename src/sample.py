import os
import torch
from torch.utils.data import DataLoader
import time
from config import *
from dataset import ICLEVRConditionDataset, labels_to_multihot, load_object_mapping
from model import ConditionalUNet
from diffusion import DDPM
from utils import (
    set_seed,
    ensure_dir,
    load_checkpoint,
    save_image_batch,
    make_grid_and_save,
    save_process_grid,
    write_log,
    save_tensor_image,
)

def format_time(seconds):
    return f"{seconds:.2f} sec"

# @torch.no_grad()
# def generate_for_dataset(
#     model,
#     ddpm,
#     dataset,
#     save_dir,
#     batch_size=32,
# ):
#     """
#     Generate images for test.json or new_test.json.

#     Output image names follow json order:
#         0.png, 1.png, 2.png, ...
#     """
#     ensure_dir(save_dir)

#     def collate_fn(batch):
#         labels = torch.stack([item[0] for item in batch], dim=0)
#         indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
#         return labels, indices

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn,
#     )

#     all_images = []

#     for labels, indices in loader:
#         labels = labels.to(DEVICE)

#         samples = ddpm.sample(
#             model=model,
#             cond=labels,
#             image_size=IMAGE_SIZE,
#             img_channels=IMG_CHANNELS,
#         )
#         if SAMPLING_METHOD == "ddpm":
#             samples = ddpm.sample(
#                 model=model,
#                 cond=labels,
#                 image_size=IMAGE_SIZE,
#                 img_channels=IMG_CHANNELS,
#             )
#         elif SAMPLING_METHOD == "ddim":
#             samples = ddpm.ddim_sample(
#                 model=model,
#                 cond=labels,
#                 image_size=IMAGE_SIZE,
#                 img_channels=IMG_CHANNELS,
#             )

#         # 用真正的 json index 存檔，確保順序正確
#         samples_cpu = samples.detach().cpu()

#         for i in range(samples_cpu.shape[0]):
#             save_path = os.path.join(save_dir, f"{indices[i].item()}.png")
#             save_tensor_image(samples_cpu[i], save_path)

#         all_images.append(samples_cpu)

#     all_images = torch.cat(all_images, dim=0)

#     return all_images

@torch.no_grad()
def generate_for_dataset(
    model,
    ddpm,
    dataset,
    save_dir,
    batch_size=32,
):
    ensure_dir(save_dir)

    start_time = time.time()

    def collate_fn(batch):
        labels = torch.stack([item[0] for item in batch], dim=0)
        indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return labels, indices

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_images = []
    total_images = 0

    for labels, indices in loader:
        labels = labels.to(DEVICE)

        if SAMPLING_METHOD == "ddpm":
            samples = ddpm.sample(
                model=model,
                cond=labels,
                image_size=IMAGE_SIZE,
                img_channels=IMG_CHANNELS,
            )
        elif SAMPLING_METHOD == "ddim":
            samples = ddpm.ddim_sample(
                model=model,
                cond=labels,
                image_size=IMAGE_SIZE,
                img_channels=IMG_CHANNELS,
            )

        samples_cpu = samples.detach().cpu()

        for i in range(samples_cpu.shape[0]):
            save_path = os.path.join(save_dir, f"{indices[i].item()}.png")
            save_tensor_image(samples_cpu[i], save_path)

        all_images.append(samples_cpu)
        total_images += samples_cpu.shape[0]

    elapsed = time.time() - start_time
    sec_per_image = elapsed / max(total_images, 1)

    all_images = torch.cat(all_images, dim=0)

    return all_images, elapsed, sec_per_image

@torch.no_grad()
def generate_denoising_process(model, ddpm):
    """
    Generate denoising process for:
        ["red sphere", "cyan cylinder", "cyan cube"]
    """
    object_to_idx, _ = load_object_mapping(OBJECT_JSON)

    cond = labels_to_multihot(
        labels=DENOISING_LABEL_SET,
        object_to_idx=object_to_idx,
        num_classes=NUM_CLASSES,
    )

    cond = cond.unsqueeze(0).to(DEVICE)

    process_images = ddpm.sample_with_process(
        model=model,
        cond=cond,
        image_size=IMAGE_SIZE,
        img_channels=IMG_CHANNELS,
        save_steps=DENOISING_SAVE_STEPS,
    )

    save_process_grid(
        process_images=process_images,
        save_path=DENOISING_PROCESS_PATH,
        nrow=GRID_NROW,
    )


def sample():
    # set_seed(SEED)

    ensure_dir(TEST_OUTPUT_DIR)
    ensure_dir(NEW_TEST_OUTPUT_DIR)
    ensure_dir(GRID_OUTPUT_DIR)
    ensure_dir(LOG_DIR)

    log_path = os.path.join(LOG_DIR, "sample_log.txt")

    write_log("=" * 70, log_path)
    write_log("Start sampling images from Conditional DDPM", log_path)
    write_log("=" * 70, log_path)

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    model = ConditionalUNet(
        img_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        time_emb_dim=TIME_EMB_DIM,
        cond_emb_dim=COND_EMB_DIM,
    ).to(DEVICE)

    ddpm = DDPM(
        timesteps=TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        schedule_type=NOISE_SCHEDULE,
        device=DEVICE,
    )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    if not os.path.exists(BEST_CKPT_PATH):
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_CKPT_PATH}\n"
            f"Please run train.py first."
        )

    model, _, start_epoch, ckpt_loss = load_checkpoint(
        model=model,
        optimizer=None,
        path=BEST_CKPT_PATH,
        device=DEVICE,
    )

    model.eval()

    write_log(f"Loaded checkpoint: {BEST_CKPT_PATH}", log_path)
    write_log(f"Checkpoint epoch: {start_epoch}", log_path)
    write_log(f"Checkpoint loss: {ckpt_loss}", log_path)

    # ------------------------------------------------------------
    # Load test datasets
    # ------------------------------------------------------------
    test_dataset = ICLEVRConditionDataset(
        json_path=TEST_JSON,
        object_json_path=OBJECT_JSON,
        num_classes=NUM_CLASSES,
    )

    new_test_dataset = ICLEVRConditionDataset(
        json_path=NEW_TEST_JSON,
        object_json_path=OBJECT_JSON,
        num_classes=NUM_CLASSES,
    )

    write_log(f"Number of test conditions: {len(test_dataset)}", log_path)
    write_log(f"Number of new_test conditions: {len(new_test_dataset)}", log_path)

    # ------------------------------------------------------------
    # Generate images for test.json
    # ------------------------------------------------------------
    write_log("Generating images for test.json...", log_path)

    test_images, test_time, test_sec_per_img = generate_for_dataset(
        model=model,
        ddpm=ddpm,
        dataset=test_dataset,
        save_dir=TEST_OUTPUT_DIR,
        batch_size=SAMPLE_BATCH_SIZE,
    )

    make_grid_and_save(
        images=test_images[:32],
        save_path=TEST_GRID_PATH,
        nrow=GRID_NROW,
    )
    write_log(
        f"test.json sampling time: {format_time(test_time)} | "
        f"{test_sec_per_img:.4f} sec/image",
        log_path,
    )
    write_log(f"Saved test images to: {TEST_OUTPUT_DIR}", log_path)
    write_log(f"Saved test grid to: {TEST_GRID_PATH}", log_path)

    # ------------------------------------------------------------
    # Generate images for new_test.json
    # ------------------------------------------------------------
    write_log("Generating images for new_test.json...", log_path)

    new_test_images, new_test_time, new_test_sec_per_img = generate_for_dataset(
        model=model,
        ddpm=ddpm,
        dataset=new_test_dataset,
        save_dir=NEW_TEST_OUTPUT_DIR,
        batch_size=SAMPLE_BATCH_SIZE,
    )

    make_grid_and_save(
        images=new_test_images[:32],
        save_path=NEW_TEST_GRID_PATH,
        nrow=GRID_NROW,
    )
    write_log(
        f"new_test.json sampling time: {format_time(new_test_time)} | "
        f"{new_test_sec_per_img:.4f} sec/image",
        log_path,
    )
    write_log(f"Saved new_test images to: {NEW_TEST_OUTPUT_DIR}", log_path)
    write_log(f"Saved new_test grid to: {NEW_TEST_GRID_PATH}", log_path)
    
    total_sampling_time = test_time + new_test_time
    total_images = len(test_dataset) + len(new_test_dataset)

    write_log("-" * 70, log_path)
    write_log("Sampling Time Summary", log_path)
    write_log(f"Method: DDPM", log_path)
    write_log(f"Total generated images: {total_images}", log_path)
    write_log(f"Total sampling time: {format_time(total_sampling_time)}", log_path)
    write_log(
        f"Average sec/image: {total_sampling_time / max(total_images, 1):.4f}",
        log_path,
    )
    write_log("-" * 70, log_path)
    # ------------------------------------------------------------
    # Generate denoising process
    # ------------------------------------------------------------
    write_log("Generating denoising process grid...", log_path)

    generate_denoising_process(
        model=model,
        ddpm=ddpm,
    )

    write_log(f"Saved denoising process to: {DENOISING_PROCESS_PATH}", log_path)

    write_log("=" * 70, log_path)
    write_log("Sampling finished.", log_path)
    write_log("=" * 70, log_path)


if __name__ == "__main__":
    sample()