import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_object_mapping(object_json_path):
    """
    Load object.json and return:
    object_to_idx: object name -> index
    idx_to_object: index -> object name
    """
    object_dict = load_json(object_json_path)

    # object.json 通常格式為：
    # {
    #   "gray cube": 0,
    #   "red cube": 1,
    #   ...
    # }
    object_to_idx = object_dict
    idx_to_object = {v: k for k, v in object_to_idx.items()}

    return object_to_idx, idx_to_object


def labels_to_multihot(labels, object_to_idx, num_classes=24):
    """
    Convert label list to multi-hot vector.

    Example:
    labels = ["red sphere", "cyan cube"]
    output shape = (24,)
    """
    multihot = torch.zeros(num_classes, dtype=torch.float32)

    for label in labels:
        if label not in object_to_idx:
            raise KeyError(f"Unknown label: {label}")

        idx = object_to_idx[label]
        multihot[idx] = 1.0

    return multihot


class ICLEVRDataset(Dataset):
    """
    Dataset for i-CLEVR training data.

    train.json format is usually:
    {
        "image_name.png": ["red sphere", "cyan cube"],
        ...
    }

    Return:
        image: Tensor, shape (3, image_size, image_size), normalized to [-1, 1]
        label: Tensor, shape (24,), multi-hot vector
    """

    def __init__(
        self,
        json_path,
        image_dir,
        object_json_path,
        image_size=64,
        transform=None,
        num_classes=24,
    ):
        super().__init__()

        self.json_path = json_path
        self.image_dir = image_dir
        self.object_json_path = object_json_path
        self.image_size = image_size
        self.num_classes = num_classes

        self.data = load_json(json_path)
        self.object_to_idx, self.idx_to_object = load_object_mapping(object_json_path)

        self.image_names = list(self.data.keys())

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        labels = self.data[image_name]

        image_path = os.path.join(self.image_dir, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = labels_to_multihot(
            labels=labels,
            object_to_idx=self.object_to_idx,
            num_classes=self.num_classes,
        )

        return image, label


class ICLEVRConditionDataset(Dataset):
    """
    Dataset for test.json / new_test.json.

    test.json and new_test.json format is usually:
    [
        ["red sphere", "cyan cube"],
        ["gray cylinder"],
        ...
    ]

    Return:
        label: Tensor, shape (24,)
        labels: original label list
        index: data order index
    """

    def __init__(
        self,
        json_path,
        object_json_path,
        num_classes=24,
    ):
        super().__init__()

        self.json_path = json_path
        self.object_json_path = object_json_path
        self.num_classes = num_classes

        self.conditions = load_json(json_path)
        self.object_to_idx, self.idx_to_object = load_object_mapping(object_json_path)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        labels = self.conditions[idx]

        label = labels_to_multihot(
            labels=labels,
            object_to_idx=self.object_to_idx,
            num_classes=self.num_classes,
        )

        return label, labels, idx


if __name__ == "__main__":
    # Simple testing code.
    # 依照你的資料夾位置修改這些路徑。
    train_json = "./data/train.json"
    test_json = "./data/test.json"
    object_json = "./data/objects.json"
    image_dir = "./data/iclevr"

    train_dataset = ICLEVRDataset(
        json_path=train_json,
        image_dir=image_dir,
        object_json_path=object_json,
        image_size=64,
    )

    print("Number of training images:", len(train_dataset))

    image, label = train_dataset[0]
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Label vector:", label)

    test_dataset = ICLEVRConditionDataset(
        json_path=test_json,
        object_json_path=object_json,
    )

    print("Number of test conditions:", len(test_dataset))

    test_label, original_labels, idx = test_dataset[0]
    print("Test index:", idx)
    print("Original labels:", original_labels)
    print("Test label shape:", test_label.shape)
    print("Test label vector:", test_label)