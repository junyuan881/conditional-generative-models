import os
import sys
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import *
from dataset import ICLEVRConditionDataset
from utils import (
    ensure_dir,
    write_log,
    clear_log,
)


def import_evaluator():
    """
    Import TA-provided evaluator.py without modifying it.

    evaluator.py internally loads:
        ./checkpoint.pth

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


class GeneratedImageDataset(Dataset):
    """
    Dataset for generated images.

    It reads generated images from:
        outputs/test/
        outputs/new_test/

    Image names should follow:
        0.png, 1.png, 2.png, ...

    Return:
        image: Tensor, shape (3, 64, 64), range [-1, 1]
        label: Tensor, shape (24,)
    """

    def __init__(
        self,
        image_dir,
        condition_json_path,
        object_json_path,
        image_size=64,
        num_classes=24,
    ):
        super().__init__()

        self.image_dir = image_dir

        self.condition_dataset = ICLEVRConditionDataset(
            json_path=condition_json_path,
            object_json_path=object_json_path,
            num_classes=num_classes,
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    def __len__(self):
        return len(self.condition_dataset)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f"{idx}.png")

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Generated image not found: {image_path}\n"
                f"Please run sample.py first."
            )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label, original_labels, index = self.condition_dataset[idx]

        return image, label


@torch.no_grad()
def evaluate_generated_images(
    evaluator,
    image_dir,
    condition_json_path,
    name,
    batch_size=32,
):
    """
    Evaluate generated images using TA evaluator.

    Args:
        evaluator: TA evaluation_model
        image_dir: generated image folder
        condition_json_path: test.json or new_test.json
        name: display name

    Return:
        mean accuracy
    """
    dataset = GeneratedImageDataset(
        image_dir=image_dir,
        condition_json_path=condition_json_path,
        object_json_path=OBJECT_JSON,
        image_size=EVAL_IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_acc = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        acc = evaluator.eval(images, labels)

        total_acc += acc
        total_batches += 1

    mean_acc = total_acc / max(total_batches, 1)

    return mean_acc


def evaluate():
    ensure_dir(LOG_DIR)

    # clear_log(EVAL_LOG_PATH)

    write_log("=" * 70, EVAL_LOG_PATH)
    write_log("Evaluate generated images by TA evaluator", EVAL_LOG_PATH)
    write_log("=" * 70, EVAL_LOG_PATH)

    write_log("Loading TA evaluator...", EVAL_LOG_PATH)
    evaluator = import_evaluator()
    write_log("TA evaluator loaded.", EVAL_LOG_PATH)

    write_log(f"Evaluating {TEST_OUTPUT_DIR} generated images...", EVAL_LOG_PATH)

    test_acc = evaluate_generated_images(
        evaluator=evaluator,
        image_dir=TEST_OUTPUT_DIR,
        condition_json_path=TEST_JSON,
        name="test.json",
        batch_size=EVAL_BATCH_SIZE,
    )

    write_log(f"Evaluating {NEW_TEST_OUTPUT_DIR} generated images...", EVAL_LOG_PATH)

    new_test_acc = evaluate_generated_images(
        evaluator=evaluator,
        image_dir=NEW_TEST_OUTPUT_DIR,
        condition_json_path=NEW_TEST_JSON,
        name="new_test.json",
        batch_size=EVAL_BATCH_SIZE,
    )

    mean_acc = (test_acc + new_test_acc) / 2.0

    write_log("-" * 70, EVAL_LOG_PATH)
    write_log(f"Accuracy on test.json:     {test_acc:.6f}", EVAL_LOG_PATH)
    write_log(f"Accuracy on new_test.json: {new_test_acc:.6f}", EVAL_LOG_PATH)
    write_log(f"Mean accuracy:             {mean_acc:.6f}", EVAL_LOG_PATH)
    write_log("-" * 70, EVAL_LOG_PATH)

    write_log(f"Result saved to: {EVAL_LOG_PATH}", EVAL_LOG_PATH)
    write_log("=" * 70, EVAL_LOG_PATH)


if __name__ == "__main__":
    evaluate()