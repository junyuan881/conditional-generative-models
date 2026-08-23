import os
import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "iclevr")

TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
NEW_TEST_JSON = os.path.join(DATA_DIR, "new_test.json")
OBJECT_JSON = os.path.join(DATA_DIR, "objects.json")

EVALUATOR_DIR = os.path.join(PROJECT_ROOT, "evaluator")
EVALUATOR_PY = os.path.join(EVALUATOR_DIR, "evaluator.py")
EVALUATOR_CKPT = os.path.join(EVALUATOR_DIR, "checkpoint.pth")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test")
NEW_TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "new_test")
GRID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "grids")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train_log.txt")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_result.txt")

BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "ddpm_cosine_best.pth")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "ddpm_last.pth")


# ============================================================
# Basic settings
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 64
IMG_CHANNELS = 3
NUM_CLASSES = 24

NUM_WORKERS = 4
PIN_MEMORY = True


# ============================================================
# Training settings
# ============================================================

BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-4
WEIGHT_DECAY = 0.0

SAVE_EVERY = 10
SAMPLE_EVERY = 10

GRAD_CLIP = 1.0


# ============================================================
# DDPM settings
# ============================================================

TIMESTEPS = 1000

# options: "linear", "cosine"
NOISE_SCHEDULE = "linear"

BETA_START = 1e-4
BETA_END = 0.02

PREDICTION_TYPE = "epsilon"

SAMPLING_METHOD = "ddpm"   # options: "ddpm", "ddim"

DDIM_STEPS = 50
DDIM_ETA = 0.0
# ============================================================
# Flow Matching settings
# ============================================================

FM_STEPS = 100
FM_TIME_EPS = 1e-3

FM_BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "fm_best.pth")
FM_LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "fm_last.pth")

FM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_fm")
FM_TEST_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "test")
FM_NEW_TEST_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "new_test")
FM_GRID_OUTPUT_DIR = os.path.join(FM_OUTPUT_DIR, "grids")

# ============================================================
# Model settings
# ============================================================

BASE_CHANNELS = 64
TIME_EMB_DIM = 256
COND_EMB_DIM = 256


# ============================================================
# Sampling settings
# ============================================================

SAMPLE_BATCH_SIZE = 32

TEST_GRID_PATH = os.path.join(GRID_OUTPUT_DIR, "test_grid.png")
NEW_TEST_GRID_PATH = os.path.join(GRID_OUTPUT_DIR, "new_test_grid.png")
DENOISING_PROCESS_PATH = os.path.join(GRID_OUTPUT_DIR, "denoising_process.png")

GRID_NROW = 8

DENOISING_LABEL_SET = [
    "red sphere",
    "cyan cylinder",
    "cyan cube",
]

DENOISING_SAVE_STEPS = [
    999,
    800,
    600,
    400,
    200,
    100,
    50,
    0,
]


# ============================================================
# Evaluator settings
# ============================================================

EVAL_IMAGE_SIZE = 64
EVAL_BATCH_SIZE = 32

# evaluator input should be normalized by:
# transforms.Normalize((0.5, 0.5, 0.5),
#                      (0.5, 0.5, 0.5))


# ============================================================
# Submission settings
# ============================================================

SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")
SUBMISSION_IMAGE_DIR = os.path.join(SUBMISSION_DIR, "images")

STUDENT_ID = "114024511"
STUDENT_NAME = "謝濬遠"

ZIP_NAME = f"DL_LAB6_{STUDENT_ID}_{STUDENT_NAME}.zip"


# ============================================================
# Helper function
# ============================================================

def print_config():
    print("=" * 60)
    print("Lab6 Conditional DDPM Config")
    print("=" * 60)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"TRAIN_JSON: {TRAIN_JSON}")
    print(f"TEST_JSON: {TEST_JSON}")
    print(f"NEW_TEST_JSON: {NEW_TEST_JSON}")
    print(f"OBJECT_JSON: {OBJECT_JSON}")

    print("-" * 60)
    print(f"DEVICE: {DEVICE}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"NUM_CLASSES: {NUM_CLASSES}")

    print("-" * 60)
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"LR: {LR}")

    print("-" * 60)
    print(f"TIMESTEPS: {TIMESTEPS}")
    print(f"NOISE_SCHEDULE: {NOISE_SCHEDULE}")
    print(f"BETA_START: {BETA_START}")
    print(f"BETA_END: {BETA_END}")

    print("-" * 60)
    print(f"BASE_CHANNELS: {BASE_CHANNELS}")
    print(f"TIME_EMB_DIM: {TIME_EMB_DIM}")
    print(f"COND_EMB_DIM: {COND_EMB_DIM}")

    print("-" * 60)
    print(f"BEST_CKPT_PATH: {BEST_CKPT_PATH}")
    print(f"LAST_CKPT_PATH: {LAST_CKPT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()