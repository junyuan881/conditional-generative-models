import os
import re
import sys
import glob
import csv
import time

import torch
import matplotlib.pyplot as plt

from config import *
from dataset import ICLEVRConditionDataset
from model import ConditionalUNet
from diffusion import DDPM
from flow_matching import FlowMatching
from utils import ensure_dir


PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SUMMARY_CSV = os.path.join(PLOT_DIR, "training_summary.csv")

MAX_EVAL_SAMPLES = 32
EVAL_SAMPLE_BATCH_SIZE = 16

MODEL_GROUPS = {
    "DDPM-linear": {
        "pattern": ["ddpm_epoch*.pt", "ddpm_epoch*.pth"],
        "method": "ddpm",
        "exclude": ["ddpm_cosine"],
    },
    "DDPM-cosine": {
        "pattern": ["ddpm_cosine_epoch*.pt", "ddpm_cosine_epoch*.pth"],
        "method": "ddpm",
        "exclude": [],
    },
    "Flow Matching": {
        "pattern": ["fm_epoch*.pt", "fm_epoch*.pth"],
        "method": "fm",
        "exclude": [],
    },
}


def import_evaluator():
    old_cwd = os.getcwd()
    sys.path.insert(0, EVALUATOR_DIR)

    try:
        os.chdir(EVALUATOR_DIR)
        from evaluator import evaluation_model
        evaluator = evaluation_model()
    finally:
        os.chdir(old_cwd)

    return evaluator


def extract_epoch_from_name(path):
    name = os.path.basename(path)

    nums = re.findall(r"\d+", name)

    if len(nums) == 0:
        return None

    return int(nums[-1])


def find_checkpoints(group_cfg):
    paths = []

    for pattern in group_cfg["pattern"]:
        paths.extend(glob.glob(os.path.join(CHECKPOINT_DIR, pattern)))

    filtered = []

    for path in paths:
        name = os.path.basename(path)

        skip = False
        for ex in group_cfg.get("exclude", []):
            if ex in name:
                skip = True
                break

        if not skip:
            filtered.append(path)

    filtered = sorted(
        filtered,
        key=lambda p: extract_epoch_from_name(p) if extract_epoch_from_name(p) is not None else 10**9,
    )

    return filtered


def load_model_from_checkpoint(ckpt_path):
    model = ConditionalUNet(
        img_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        time_emb_dim=TIME_EMB_DIM,
        cond_emb_dim=COND_EMB_DIM,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    epoch = ckpt.get("epoch", extract_epoch_from_name(ckpt_path)) if isinstance(ckpt, dict) else extract_epoch_from_name(ckpt_path)
    loss = ckpt.get("loss", None) if isinstance(ckpt, dict) else None

    model.eval()

    return model, epoch, loss


@torch.no_grad()
def evaluate_checkpoint(model, method, evaluator, test_dataset, new_test_dataset):
    if method == "ddpm":
        sampler = DDPM(
            timesteps=TIMESTEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            schedule_type=NOISE_SCHEDULE,
            device=DEVICE,
        )
    elif method == "fm":
        sampler = FlowMatching(
            num_steps=100,
            time_scale=1000,
            device=DEVICE,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    test_acc = evaluate_dataset(model, sampler, method, evaluator, test_dataset)
    new_test_acc = evaluate_dataset(model, sampler, method, evaluator, new_test_dataset)

    return test_acc, new_test_acc, (test_acc + new_test_acc) / 2.0


@torch.no_grad()
def evaluate_dataset(model, sampler, method, evaluator, dataset):
    n = min(len(dataset), MAX_EVAL_SAMPLES)

    total_acc = 0.0
    total_batches = 0

    for start in range(0, n, EVAL_SAMPLE_BATCH_SIZE):
        labels = []

        for idx in range(start, min(start + EVAL_SAMPLE_BATCH_SIZE, n)):
            label, _, _ = dataset[idx]
            labels.append(label)

        labels = torch.stack(labels, dim=0).to(DEVICE)

        if method == "ddpm":
            images = sampler.sample(
                model=model,
                cond=labels,
                image_size=IMAGE_SIZE,
                img_channels=IMG_CHANNELS,
            )
        elif method == "fm":
            images = sampler.sample(
                model=model,
                cond=labels,
                image_size=IMAGE_SIZE,
                img_channels=IMG_CHANNELS,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        acc = evaluator.eval(images.to(DEVICE), labels.to(DEVICE))

        total_acc += acc
        total_batches += 1

    return total_acc / max(total_batches, 1)


def save_csv(rows):
    ensure_dir(PLOT_DIR)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "checkpoint",
                "epoch",
                "loss",
                "test_acc",
                "new_test_acc",
                "mean_acc",
                "eval_time_sec",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def plot_loss_curve(rows):
    ensure_dir(PLOT_DIR)

    plt.figure(figsize=(8, 5))

    for method in MODEL_GROUPS.keys():
        subset = [r for r in rows if r["method"] == method and r["loss"] is not None]
        subset = sorted(subset, key=lambda r: r["epoch"])

        if len(subset) == 0:
            continue

        epochs = [r["epoch"] for r in subset]
        losses = [r["loss"] for r in subset]

        plt.plot(epochs, losses, marker="o", label=method)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "training_loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved loss curve to: {save_path}")


def plot_acc_curve(rows):
    ensure_dir(PLOT_DIR)

    plt.figure(figsize=(8, 5))

    for method in MODEL_GROUPS.keys():
        subset = [r for r in rows if r["method"] == method]
        subset = sorted(subset, key=lambda r: r["epoch"])

        if len(subset) == 0:
            continue

        epochs = [r["epoch"] for r in subset]
        accs = [r["mean_acc"] for r in subset]

        plt.plot(epochs, accs, marker="o", label=method)

    plt.xlabel("Epoch")
    plt.ylabel("Evaluator Mean Accuracy")
    plt.title("Evaluator Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "evaluator_accuracy_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved accuracy curve to: {save_path}")


def main():
    ensure_dir(PLOT_DIR)

    print("=" * 70)
    print("Evaluate checkpoints and generate training curves")
    print("=" * 70)

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

    print("Loading TA evaluator...")
    evaluator = import_evaluator()
    print("Evaluator loaded.")

    rows = []

    for method_name, cfg in MODEL_GROUPS.items():
        ckpt_paths = find_checkpoints(cfg)

        print("-" * 70)
        print(f"Method: {method_name}")
        print(f"Found {len(ckpt_paths)} checkpoints")

        for ckpt_path in ckpt_paths:
            print(f"Evaluating: {ckpt_path}")

            start_time = time.time()

            model, epoch, loss = load_model_from_checkpoint(ckpt_path)

            test_acc, new_test_acc, mean_acc = evaluate_checkpoint(
                model=model,
                method=cfg["method"],
                evaluator=evaluator,
                test_dataset=test_dataset,
                new_test_dataset=new_test_dataset,
            )

            elapsed = time.time() - start_time

            row = {
                "method": method_name,
                "checkpoint": os.path.basename(ckpt_path),
                "epoch": epoch,
                "loss": loss,
                "test_acc": test_acc,
                "new_test_acc": new_test_acc,
                "mean_acc": mean_acc,
                "eval_time_sec": elapsed,
            }

            rows.append(row)

            print(
                f"Epoch={epoch} | "
                f"Loss={loss} | "
                f"test={test_acc:.4f} | "
                f"new_test={new_test_acc:.4f} | "
                f"mean={mean_acc:.4f} | "
                f"time={elapsed:.2f}s"
            )

            del model
            torch.cuda.empty_cache()

    save_csv(rows)
    plot_loss_curve(rows)
    plot_acc_curve(rows)

    print("=" * 70)
    print(f"Saved summary CSV to: {SUMMARY_CSV}")
    print("Finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()