import os
import json
import random
import numpy as np

import torch
from torchvision.utils import save_image, make_grid


def set_seed(seed=42):
    """
    Fix random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path):
    """
    Create folder if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def load_json(json_path):
    """
    Load json file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, json_path):
    """
    Save object as json file.
    """
    folder = os.path.dirname(json_path)
    if folder != "":
        ensure_dir(folder)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def denormalize(x):
    """
    Convert image tensor from [-1, 1] to [0, 1].

    Input:
        x: Tensor, shape (B, C, H, W) or (C, H, W)

    Output:
        Tensor clipped to [0, 1]
    """
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x


def normalize_to_minus_one_to_one(x):
    """
    Convert image tensor from [0, 1] to [-1, 1].
    """
    return x * 2.0 - 1.0


def save_tensor_image(x, path):
    """
    Save one tensor image.

    Input:
        x: shape (3, H, W) or (1, 3, H, W), range [-1, 1]
    """
    folder = os.path.dirname(path)
    if folder != "":
        ensure_dir(folder)

    if x.dim() == 4:
        x = x[0]

    x = denormalize(x.detach().cpu())
    save_image(x, path)


def save_image_batch(images, save_dir, start_index=0):
    """
    Save a batch of generated images as png.

    Input:
        images: shape (B, 3, H, W), range [-1, 1]
        save_dir: output folder
        start_index: image index offset
    """
    ensure_dir(save_dir)

    images = images.detach().cpu()

    for i in range(images.shape[0]):
        save_path = os.path.join(save_dir, f"{start_index + i}.png")
        save_tensor_image(images[i], save_path)


def make_grid_and_save(images, save_path, nrow=8):
    """
    Save image grid.

    Input:
        images: shape (B, 3, H, W), range [-1, 1]
    """
    folder = os.path.dirname(save_path)
    if folder != "":
        ensure_dir(folder)

    images = denormalize(images.detach().cpu())
    grid = make_grid(images, nrow=nrow)
    save_image(grid, save_path)


def save_process_grid(process_images, save_path, nrow=8):
    """
    Save denoising process grid.

    Input:
        process_images: list of tensors.
        Each tensor shape can be (1, 3, H, W) or (3, H, W).
        Range should be [-1, 1].
    """
    folder = os.path.dirname(save_path)
    if folder != "":
        ensure_dir(folder)

    imgs = []

    for img in process_images:
        if img.dim() == 4:
            img = img[0]
        imgs.append(img)

    imgs = torch.stack(imgs, dim=0)
    make_grid_and_save(imgs, save_path, nrow=nrow)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.
    """
    folder = os.path.dirname(path)
    if folder != "":
        ensure_dir(folder)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss": loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, path, device="cuda", optimizer=None):
    """
    Load model checkpoint.

    Return:
        model, optimizer, start_epoch, loss
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    return model, optimizer, start_epoch, loss


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_log(message, log_path):
    """
    Write message to log file and print it.
    """
    print(message)

    folder = os.path.dirname(log_path)
    if folder != "":
        ensure_dir(folder)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def clear_log(log_path):
    """
    Clear existing log file.
    """
    folder = os.path.dirname(log_path)
    if folder != "":
        ensure_dir(folder)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")


def get_device():
    """
    Get available device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def labels_to_multihot(labels, object_to_idx, num_classes=24):
    """
    Convert list of object labels to multi-hot vector.

    This is repeated here for convenience.
    dataset.py can also import and use this function.
    """
    multihot = torch.zeros(num_classes, dtype=torch.float32)

    for label in labels:
        if label not in object_to_idx:
            raise KeyError(f"Unknown label: {label}")
        multihot[object_to_idx[label]] = 1.0

    return multihot


def load_object_mapping(object_json_path):
    """
    Load object.json.

    Return:
        object_to_idx, idx_to_object
    """
    object_to_idx = load_json(object_json_path)
    idx_to_object = {v: k for k, v in object_to_idx.items()}

    return object_to_idx, idx_to_object


def prepare_submission_images(output_test_dir, output_new_test_dir, submission_image_dir):
    """
    Copy generated images into required submission structure.

    Required:
        images/
        ├── test/
        └── new_test/
    """
    import shutil

    test_target_dir = os.path.join(submission_image_dir, "test")
    new_test_target_dir = os.path.join(submission_image_dir, "new_test")

    ensure_dir(test_target_dir)
    ensure_dir(new_test_target_dir)

    for filename in sorted(os.listdir(output_test_dir), key=lambda x: int(os.path.splitext(x)[0])):
        if filename.endswith(".png"):
            shutil.copy(
                os.path.join(output_test_dir, filename),
                os.path.join(test_target_dir, filename),
            )

    for filename in sorted(os.listdir(output_new_test_dir), key=lambda x: int(os.path.splitext(x)[0])):
        if filename.endswith(".png"):
            shutil.copy(
                os.path.join(output_new_test_dir, filename),
                os.path.join(new_test_target_dir, filename),
            )


if __name__ == "__main__":
    set_seed(42)

    x = torch.randn(4, 3, 64, 64)
    x = torch.clamp(x, -1, 1)

    ensure_dir("./outputs/test_utils")
    make_grid_and_save(x, "./outputs/test_utils/grid.png", nrow=4)
    save_image_batch(x, "./outputs/test_utils/images", start_index=0)

    print("utils.py test finished.")