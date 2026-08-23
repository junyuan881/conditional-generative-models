# NYCU 2026 Spring Deep Learning Lab6

- Name: 謝濬遠
- Student ID: 114024511

## Conditional Generative Models: DDPM and Flow Matching

This project implements:

1. Conditional Denoising Diffusion Probabilistic Model (DDPM)
2. Conditional Flow Matching (FM)

for the i-CLEVR multi-object image generation task in NYCU Deep Learning Lab6.

The models generate synthetic images according to multi-label conditions such as:

```python
["red sphere", "cyan cylinder", "cyan cube"]
```

The generated images are evaluated by the TA-provided pretrained evaluator based on ResNet18.

---

# Project Structure

```text
Deep-Learning-Lab6-114024511/
├── data/
│   ├── train.json
│   ├── test.json
│   ├── new_test.json
│   ├── objects.json
│   └── iclevr/
│
├── evaluator/
│   ├── evaluator.py
│   ├── checkpoint.pth.part01
│   ├── checkpoint.pth.part02
│   ├── checkpoint.pth.part03
│   └── reassemble_checkpoint.py
│
├── checkpoints/
│
├── outputs/
│   ├── test/
│   ├── new_test/
│   ├── grids/
│   └── logs/
│
├── outputs_fm/
│   ├── test/
│   ├── new_test/
│   ├── grids/
│   └── logs/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── diffusion.py
│   ├── flow_matching.py
│   ├── train.py
│   ├── train_fm.py
│   ├── sample.py
│   ├── sample_fm.py
│   ├── evaluate.py
│   └── utils.py
│
├── requirements.txt
├── LAB6_114024511_Report.pdf
└── README.md
```

---

# Environment

## Python

```text
Python 3.10+
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Put the dataset files into:

```text
data/
```

Required files:

```text
train.json
test.json
new_test.json
objects.json
iclevr/
```

Put the evaluator files into:

```text
evaluator/
```

Required files:

```text
evaluator.py
checkpoint.pth
```

The GitHub repository stores `checkpoint.pth` in three lossless parts because
the original file is larger than the browser upload limit. Reassemble it after
cloning:

```bash
python evaluator/reassemble_checkpoint.py
```

The script verifies the reconstructed file's SHA-256 checksum before replacing
`evaluator/checkpoint.pth`.

---

# Training

## Train Conditional DDPM

```bash
python src/train.py
```

Outputs:

```text
checkpoints/ddpm_best.pth
checkpoints/ddpm_last.pth
outputs/logs/train_log.txt
```

---

## Train Conditional Flow Matching

```bash
python src/train_fm.py
```

Outputs:

```text
checkpoints/fm_best.pth
checkpoints/fm_last.pth
outputs_fm/logs/train_fm_log.txt
```

---

# Sampling

## Generate images using DDPM

```bash
python src/sample.py
```

Outputs:

```text
outputs/
├── test/
├── new_test/
└── grids/
```

Generated grids:

```text
outputs/grids/test_grid.png
outputs/grids/new_test_grid.png
outputs/grids/denoising_process.png
```

---

## Generate images using Flow Matching

```bash
python src/sample_fm.py
```

Outputs:

```text
outputs_fm/
├── test/
├── new_test/
└── grids/
```

Generated grids:

```text
outputs_fm/grids/fm_test_grid.png
outputs_fm/grids/fm_new_test_grid.png
outputs_fm/grids/fm_generation_process.png
```

---

# Evaluation

Evaluate generated images using the TA-provided evaluator.

```bash
python src/evaluate.py
```

Outputs:

```text
outputs/logs/eval_result.txt
```

The evaluator computes:

1. Accuracy on `test.json`
2. Accuracy on `new_test.json`

---

# Model Details

## Conditional DDPM

The DDPM model predicts the added Gaussian noise:

```text
x_t -> epsilon_theta(x_t, t, y)
```

Main components:

* Conditional UNet
* Time embedding
* Multi-label condition embedding
* Linear / cosine noise schedule
* Reverse denoising sampling

---

## Conditional Flow Matching

The Flow Matching model predicts the velocity field:

```text
x_t -> v_theta(x_t, t, y)
```

Main components:

* Conditional UNet
* Continuous probability path
* ODE-based generation
* Faster sampling compared to DDPM

---

# Features

## Implemented Features

* Conditional DDPM
* Conditional Flow Matching
* Multi-label condition embedding
* Time embedding
* Cosine / linear noise schedule
* Automatic evaluator-based best model selection
* Image grid generation
* Denoising / generation process visualization
* Automatic checkpoint saving

---

# Important Notes

## Evaluator Input

The TA evaluator requires:

```text
image shape: (B, 3, 64, 64)
label shape: (B, 24)
```

Images must be normalized using:

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

The current implementation already follows this requirement.

---

# Suggested Improvements

Possible extensions:

* Self-attention blocks
* EMA model
* Classifier guidance
* DDIM sampling
* Better condition injection (FiLM)
* Candidate sampling + evaluator selection

---

# References

1. Denoising Diffusion Probabilistic Models
   Ho et al., 2020

2. Flow Matching for Generative Modeling
   Lipman et al., 2023

3. Hugging Face Diffusion Course
   [https://huggin](https://huggin)

---

## Repository Notes

- Student: 謝濬遠 (114024511)
- `requirements.txt` and `data/objects.json` now match the filenames used by the source code.
- Large per-epoch preview images and training logs were removed; final generated images, key grids, and evaluation results are retained.
- Training checkpoints are generated under `checkpoints/` and ignored by Git.
