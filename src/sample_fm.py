import os
import torch
from torch.utils.data import DataLoader
import time
from config import *
from dataset import ICLEVRConditionDataset, labels_to_multihot, load_object_mapping
from model import ConditionalUNet
from flow_matching import FlowMatching
from utils import (
    set_seed,
    ensure_dir,
    load_checkpoint,
    make_grid_and_save,
    save_process_grid,
    save_tensor_image,
    write_log,
)


FM_BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "fm_best.pth")

FM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_fm")
FM_TEST_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "test")
FM_NEW_TEST_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "new_test")
FM_GRID_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "grids")
FM_LOG_DIR = os.path.join(FM_OUTPUT_DIR, "logs")

FM_TEST_GRID_PATH = os.path.join(FM_GRID_OUTPUT_DIR, "fm_test_grid.png")
FM_NEW_TEST_GRID_PATH = os.path.join(FM_GRID_OUTPUT_DIR, "fm_new_test_grid.png")
FM_PROCESS_PATH = os.path.join(FM_GRID_OUTPUT_DIR, "fm_generation_process.png")

FM_NUM_STEPS = 100
FM_TIME_SCALE = 1000

def format_time(seconds):
    return f"{seconds:.2f} sec"

# @torch.no_grad()
# def generate_for_dataset(model, fm, dataset, save_dir, batch_size=32):
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

#         samples = fm.sample(
#             model=model,
#             cond=labels,
#             image_size=IMAGE_SIZE,
#             img_channels=IMG_CHANNELS,
#         )

#         samples_cpu = samples.detach().cpu()

#         for i in range(samples_cpu.shape[0]):
#             save_path = os.path.join(save_dir, f"{indices[i].item()}.png")
#             save_tensor_image(samples_cpu[i], save_path)

#         all_images.append(samples_cpu)

#     return torch.cat(all_images, dim=0)

@torch.no_grad()
def generate_for_dataset(
    model,
    fm,
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

        samples = fm.sample(
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
def generate_flow_process(model, fm):
    object_to_idx, _ = load_object_mapping(OBJECT_JSON)

    cond = labels_to_multihot(
        labels=DENOISING_LABEL_SET,
        object_to_idx=object_to_idx,
        num_classes=NUM_CLASSES,
    ).unsqueeze(0).to(DEVICE)

    process_images = fm.sample_with_process(
        model=model,
        cond=cond,
        image_size=IMAGE_SIZE,
        img_channels=IMG_CHANNELS,
    )

    save_process_grid(
        process_images=process_images,
        save_path=FM_PROCESS_PATH,
        nrow=GRID_NROW,
    )


def sample_fm():
    set_seed(SEED)

    ensure_dir(FM_TEST_OUTPUT_DIR)
    ensure_dir(FM_NEW_TEST_OUTPUT_DIR)
    ensure_dir(FM_GRID_OUTPUT_DIR)
    ensure_dir(FM_LOG_DIR)

    log_path = os.path.join(FM_LOG_DIR, "sample_fm_log.txt")

    write_log("=" * 70, log_path)
    write_log("Start sampling images from Conditional Flow Matching", log_path)
    write_log("=" * 70, log_path)

    model = ConditionalUNet(
        img_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        time_emb_dim=TIME_EMB_DIM,
        cond_emb_dim=COND_EMB_DIM,
    ).to(DEVICE)

    fm = FlowMatching(
        num_steps=FM_NUM_STEPS,
        time_scale=FM_TIME_SCALE,
        device=DEVICE,
    )

    if not os.path.exists(FM_BEST_CKPT_PATH):
        raise FileNotFoundError(
            f"Flow Matching checkpoint not found: {FM_BEST_CKPT_PATH}\n"
            f"Please run train_fm.py first."
        )

    model, _, start_epoch, ckpt_loss = load_checkpoint(
        model=model,
        optimizer=None,
        path=FM_BEST_CKPT_PATH,
        device=DEVICE,
    )

    model.eval()

    write_log(f"Loaded checkpoint: {FM_BEST_CKPT_PATH}", log_path)
    write_log(f"Checkpoint epoch: {start_epoch}", log_path)
    write_log(f"Checkpoint loss: {ckpt_loss}", log_path)

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

    write_log("Generating Flow Matching images for test.json...", log_path)

    test_images, test_time, test_sec_per_img = generate_for_dataset(
        model=model,
        fm=fm,
        dataset=test_dataset,
        save_dir=FM_TEST_OUTPUT_DIR,
        batch_size=SAMPLE_BATCH_SIZE,
    )

    make_grid_and_save(
        images=test_images[:32],
        save_path=FM_TEST_GRID_PATH,
        nrow=GRID_NROW,
    )

    write_log(f"Saved test images to: {FM_TEST_OUTPUT_DIR}", log_path)
    write_log(f"Saved test grid to: {FM_TEST_GRID_PATH}", log_path)

    write_log("Generating Flow Matching images for new_test.json...", log_path)

    new_test_images, new_test_time, new_test_sec_per_img = generate_for_dataset(
        model=model,
        fm=fm,
        dataset=new_test_dataset,
        save_dir=FM_NEW_TEST_OUTPUT_DIR,
        batch_size=SAMPLE_BATCH_SIZE,
    )

    make_grid_and_save(
        images=new_test_images[:32],
        save_path=FM_NEW_TEST_GRID_PATH,
        nrow=GRID_NROW,
    )
    write_log(
        f"new_test.json sampling time: {format_time(new_test_time)} | "
        f"{new_test_sec_per_img:.4f} sec/image",
        log_path,
    )
    write_log(f"Saved new_test images to: {FM_NEW_TEST_OUTPUT_DIR}", log_path)
    write_log(f"Saved new_test grid to: {FM_NEW_TEST_GRID_PATH}", log_path)

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

    write_log("Generating Flow Matching generation process grid...", log_path)

    generate_flow_process(
        model=model,
        fm=fm,
    )

    write_log(f"Saved generation process to: {FM_PROCESS_PATH}", log_path)

    write_log("=" * 70, log_path)
    write_log("Flow Matching sampling finished.", log_path)
    write_log("=" * 70, log_path)


if __name__ == "__main__":
    sample_fm()