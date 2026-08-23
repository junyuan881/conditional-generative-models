import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import *
from dataset import ICLEVRDataset, ICLEVRConditionDataset
from model import ConditionalUNet
from diffusion import DDPM
from utils import (
    set_seed,
    ensure_dir,
    save_checkpoint,
    write_log,
    clear_log,
    count_parameters,
    make_grid_and_save,
)


def import_evaluator():
    """
    Import TA-provided evaluator.py without modifying it.

    Important:
    evaluator.py internally uses:
        torch.load('./checkpoint.pth')

    Therefore, we temporarily change cwd to EVALUATOR_DIR.
    """
    old_cwd = os.getcwd()
    sys.path.insert(0, EVALUATOR_DIR)

    try:
        os.chdir(EVALUATOR_DIR)
        from evaluator import evaluation_model
        evaluator = evaluation_model()
    finally:
        os.chdir(old_cwd)

    return evaluator


@torch.no_grad()
def evaluate_by_classifier(
    model,
    ddpm,
    evaluator,
    test_dataset,
    max_eval_samples=32,
    batch_size=16,
):
    """
    Generate images from conditions and evaluate them by TA evaluator.

    Args:
        model: ConditionalUNet
        ddpm: DDPM object
        evaluator: TA-provided evaluation_model
        test_dataset: ICLEVRConditionDataset
        max_eval_samples: use subset for fast model selection
        batch_size: sampling batch size

    Return:
        mean accuracy
    """
    model.eval()

    total_acc = 0.0
    total_batches = 0

    n = min(len(test_dataset), max_eval_samples)
    indices = list(range(n))

    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]

        labels = []
        for idx in batch_indices:
            label, _, _ = test_dataset[idx]
            labels.append(label)

        labels = torch.stack(labels, dim=0).to(DEVICE)

        samples = ddpm.sample(
            model=model,
            cond=labels,
            image_size=IMAGE_SIZE,
            img_channels=IMG_CHANNELS,
        )

        # evaluator expects images normalized with mean=0.5, std=0.5
        # our DDPM samples are already in [-1, 1], so they are valid.
        acc = evaluator.eval(samples.to(DEVICE), labels.to(DEVICE))

        total_acc += acc
        total_batches += 1

    mean_acc = total_acc / max(total_batches, 1)

    model.train()

    return mean_acc


@torch.no_grad()
def save_training_preview(model, ddpm, test_dataset, epoch):
    """
    Save preview generated images during training.
    """
    model.eval()

    n = min(16, len(test_dataset))
    labels = []

    for i in range(n):
        label, _, _ = test_dataset[i]
        labels.append(label)

    labels = torch.stack(labels, dim=0).to(DEVICE)

    samples = ddpm.sample(
        model=model,
        cond=labels,
        image_size=IMAGE_SIZE,
        img_channels=IMG_CHANNELS,
    )

    preview_path = os.path.join(GRID_OUTPUT_DIR, f"preview_epoch_{epoch:03d}.png")
    make_grid_and_save(samples, preview_path, nrow=8)

    model.train()


def train():
    set_seed(SEED)

    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(GRID_OUTPUT_DIR)
    ensure_dir(LOG_DIR)

    clear_log(TRAIN_LOG_PATH)

    write_log("=" * 70, TRAIN_LOG_PATH)
    write_log("Start training Conditional DDPM for Lab6", TRAIN_LOG_PATH)
    write_log("=" * 70, TRAIN_LOG_PATH)
    write_log(f"Device: {DEVICE}", TRAIN_LOG_PATH)
    write_log(f"Image size: {IMAGE_SIZE}", TRAIN_LOG_PATH)
    write_log(f"Batch size: {BATCH_SIZE}", TRAIN_LOG_PATH)
    write_log(f"Epochs: {EPOCHS}", TRAIN_LOG_PATH)
    write_log(f"Learning rate: {LR}", TRAIN_LOG_PATH)
    write_log(f"Timesteps: {TIMESTEPS}", TRAIN_LOG_PATH)
    write_log(f"Noise schedule: {NOISE_SCHEDULE}", TRAIN_LOG_PATH)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    train_dataset = ICLEVRDataset(
        json_path=TRAIN_JSON,
        image_dir=IMAGE_DIR,
        object_json_path=OBJECT_JSON,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

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

    write_log(f"Number of training images: {len(train_dataset)}", TRAIN_LOG_PATH)
    write_log(f"Number of test conditions: {len(test_dataset)}", TRAIN_LOG_PATH)
    write_log(f"Number of new_test conditions: {len(new_test_dataset)}", TRAIN_LOG_PATH)

    # ------------------------------------------------------------
    # Model
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    write_log(f"Trainable parameters: {count_parameters(model):,}", TRAIN_LOG_PATH)

    # ------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------
    write_log("Loading TA evaluator...", TRAIN_LOG_PATH)
    evaluator = import_evaluator()
    write_log("TA evaluator loaded.", TRAIN_LOG_PATH)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    best_score = -1.0
    best_epoch = 0

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            loss = ddpm.training_loss(
                model=model,
                x_start=images,
                cond=labels,
            )

            optimizer.zero_grad()
            loss.backward()

            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                msg = (
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )
                write_log(msg, TRAIN_LOG_PATH)

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time

        write_log(
            f"Epoch [{epoch}/{EPOCHS}] finished | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Time: {elapsed:.2f}s",
            TRAIN_LOG_PATH,
        )

        # --------------------------------------------------------
        # Save last checkpoint
        # --------------------------------------------------------
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            path=LAST_CKPT_PATH,
        )

        # --------------------------------------------------------
        # Evaluate by TA evaluator
        # --------------------------------------------------------
        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            write_log("Evaluating generated images by TA evaluator...", TRAIN_LOG_PATH)

            test_acc = evaluate_by_classifier(
                model=model,
                ddpm=ddpm,
                evaluator=evaluator,
                test_dataset=test_dataset,
                max_eval_samples=32,
                batch_size=16,
            )

            new_test_acc = evaluate_by_classifier(
                model=model,
                ddpm=ddpm,
                evaluator=evaluator,
                test_dataset=new_test_dataset,
                max_eval_samples=32,
                batch_size=16,
            )

            score = (test_acc + new_test_acc) / 2.0

            write_log(
                f"Evaluator accuracy | "
                f"test.json: {test_acc:.4f} | "
                f"new_test.json: {new_test_acc:.4f} | "
                f"mean score: {score:.4f}",
                TRAIN_LOG_PATH,
            )

            save_training_preview(
                model=model,
                ddpm=ddpm,
                test_dataset=test_dataset,
                epoch=epoch,
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    path=BEST_CKPT_PATH,
                )

                write_log(
                    f"New best model saved at epoch {epoch} | "
                    f"Best score: {best_score:.4f}",
                    TRAIN_LOG_PATH,
                )

        # --------------------------------------------------------
        # Periodic checkpoint
        # --------------------------------------------------------
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_epoch_{epoch:03d}.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                path=ckpt_path,
            )

    write_log("=" * 70, TRAIN_LOG_PATH)
    write_log("Training finished.", TRAIN_LOG_PATH)
    write_log(f"Best epoch: {best_epoch}", TRAIN_LOG_PATH)
    write_log(f"Best evaluator score: {best_score:.4f}", TRAIN_LOG_PATH)
    write_log(f"Best checkpoint path: {BEST_CKPT_PATH}", TRAIN_LOG_PATH)
    write_log("=" * 70, TRAIN_LOG_PATH)


if __name__ == "__main__":
    train()