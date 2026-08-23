import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from dataset import ICLEVRDataset, ICLEVRConditionDataset
from model import ConditionalUNet
from flow_matching import FlowMatching
from utils import (
    set_seed,
    ensure_dir,
    save_checkpoint,
    write_log,
    clear_log,
    count_parameters,
    make_grid_and_save,
)


# ============================================================
# Flow Matching specific paths
# ============================================================

FM_BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "fm_best.pth")
FM_LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "fm_last.pth")

FM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_fm")
FM_GRID_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "grids")
FM_LOG_DIR = os.path.join(FM_OUTPUT_DIR, "logs")
FM_TRAIN_LOG_PATH = os.path.join(FM_LOG_DIR, "train_fm_log.txt")

FM_NUM_STEPS = 100
FM_TIME_SCALE = 1000


def import_evaluator():
    """
    Import TA-provided evaluator.py without modifying it.

    evaluator.py internally loads:
        ./checkpoint.pth

    Therefore, temporarily change cwd to EVALUATOR_DIR.
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
    fm,
    evaluator,
    test_dataset,
    max_eval_samples=32,
    batch_size=16,
):
    """
    Generate images by Flow Matching and evaluate them by TA evaluator.
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

        samples = fm.sample(
            model=model,
            cond=labels,
            image_size=IMAGE_SIZE,
            img_channels=IMG_CHANNELS,
        )

        acc = evaluator.eval(samples.to(DEVICE), labels.to(DEVICE))

        total_acc += acc
        total_batches += 1

    mean_acc = total_acc / max(total_batches, 1)

    model.train()

    return mean_acc


@torch.no_grad()
def save_training_preview(model, fm, test_dataset, epoch):
    """
    Save preview generated images during Flow Matching training.
    """
    model.eval()

    n = min(16, len(test_dataset))
    labels = []

    for i in range(n):
        label, _, _ = test_dataset[i]
        labels.append(label)

    labels = torch.stack(labels, dim=0).to(DEVICE)

    samples = fm.sample(
        model=model,
        cond=labels,
        image_size=IMAGE_SIZE,
        img_channels=IMG_CHANNELS,
    )

    preview_path = os.path.join(FM_GRID_OUTPUT_DIR, f"fm_preview_epoch_{epoch:03d}.png")
    make_grid_and_save(samples, preview_path, nrow=8)

    model.train()


def train_fm():
    set_seed(SEED)

    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(FM_OUTPUT_DIR)
    ensure_dir(FM_GRID_OUTPUT_DIR)
    ensure_dir(FM_LOG_DIR)

    clear_log(FM_TRAIN_LOG_PATH)

    write_log("=" * 70, FM_TRAIN_LOG_PATH)
    write_log("Start training Conditional Flow Matching for Lab6", FM_TRAIN_LOG_PATH)
    write_log("=" * 70, FM_TRAIN_LOG_PATH)
    write_log(f"Device: {DEVICE}", FM_TRAIN_LOG_PATH)
    write_log(f"Image size: {IMAGE_SIZE}", FM_TRAIN_LOG_PATH)
    write_log(f"Batch size: {BATCH_SIZE}", FM_TRAIN_LOG_PATH)
    write_log(f"Epochs: {EPOCHS}", FM_TRAIN_LOG_PATH)
    write_log(f"Learning rate: {LR}", FM_TRAIN_LOG_PATH)
    write_log(f"FM sampling steps: {FM_NUM_STEPS}", FM_TRAIN_LOG_PATH)
    write_log(f"FM time scale: {FM_TIME_SCALE}", FM_TRAIN_LOG_PATH)

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

    write_log(f"Number of training images: {len(train_dataset)}", FM_TRAIN_LOG_PATH)
    write_log(f"Number of test conditions: {len(test_dataset)}", FM_TRAIN_LOG_PATH)
    write_log(f"Number of new_test conditions: {len(new_test_dataset)}", FM_TRAIN_LOG_PATH)

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

    fm = FlowMatching(
        num_steps=FM_NUM_STEPS,
        time_scale=FM_TIME_SCALE,
        device=DEVICE,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    write_log(f"Trainable parameters: {count_parameters(model):,}", FM_TRAIN_LOG_PATH)

    # ------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------
    write_log("Loading TA evaluator...", FM_TRAIN_LOG_PATH)
    evaluator = import_evaluator()
    write_log("TA evaluator loaded.", FM_TRAIN_LOG_PATH)

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

            loss = fm.training_loss(
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
                write_log(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{batch_idx + 1}/{len(train_loader)}] "
                    f"FM Loss: {loss.item():.6f}",
                    FM_TRAIN_LOG_PATH,
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time

        write_log(
            f"Epoch [{epoch}/{EPOCHS}] finished | "
            f"Avg FM Loss: {avg_loss:.6f} | "
            f"Time: {elapsed:.2f}s",
            FM_TRAIN_LOG_PATH,
        )

        # --------------------------------------------------------
        # Save last checkpoint
        # --------------------------------------------------------
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            path=FM_LAST_CKPT_PATH,
        )

        # --------------------------------------------------------
        # Evaluate by TA evaluator
        # --------------------------------------------------------
        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            write_log("Evaluating Flow Matching generated images...", FM_TRAIN_LOG_PATH)

            test_acc = evaluate_by_classifier(
                model=model,
                fm=fm,
                evaluator=evaluator,
                test_dataset=test_dataset,
                max_eval_samples=32,
                batch_size=16,
            )

            new_test_acc = evaluate_by_classifier(
                model=model,
                fm=fm,
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
                FM_TRAIN_LOG_PATH,
            )

            save_training_preview(
                model=model,
                fm=fm,
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
                    path=FM_BEST_CKPT_PATH,
                )

                write_log(
                    f"New best Flow Matching model saved at epoch {epoch} | "
                    f"Best score: {best_score:.4f}",
                    FM_TRAIN_LOG_PATH,
                )

        # --------------------------------------------------------
        # Periodic checkpoint
        # --------------------------------------------------------
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"fm_epoch_{epoch:03d}.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                path=ckpt_path,
            )

    write_log("=" * 70, FM_TRAIN_LOG_PATH)
    write_log("Flow Matching training finished.", FM_TRAIN_LOG_PATH)
    write_log(f"Best epoch: {best_epoch}", FM_TRAIN_LOG_PATH)
    write_log(f"Best evaluator score: {best_score:.4f}", FM_TRAIN_LOG_PATH)
    write_log(f"Best checkpoint path: {FM_BEST_CKPT_PATH}", FM_TRAIN_LOG_PATH)
    write_log("=" * 70, FM_TRAIN_LOG_PATH)


if __name__ == "__main__":
    train_fm()