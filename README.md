# MAI500-s4: Fashion-MNIST Image Classification

Weekly task for MAI500 (Machine Learning Algorithms).

**Activity:** Hands-On Machine Learning Workshop

**Objective:** Gain practical experience with machine learning algorithms by working through data preparation, model selection, training and evaluation on a straightforward dataset.

## What this project does

`train.py` trains two supervised learning models to sort images of clothing into 10 categories, then compares how accurate they are.

1. **Dataset:** [Fashion-MNIST](https://www.openml.org/d/40996), downloaded from OpenML with scikit-learn. It has 70,000 greyscale images of 28×28 pixels in 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.
2. **Data preparation:** each image becomes 784 pixel values, scaled from 0–255 to 0–1. The data is split into 60,000 training images and 10,000 test images. The split is stratified so every class is equally represented, and repeatable (`random_state=42`).
3. **Model selection:** a decision tree (`max_depth=10`) and logistic regression (`max_iter=200`).
4. **Training:** both models are trained on the CPU.
5. **Evaluation:** both models are scored on the 10,000 test images they did not see during training.

## Results

| Model | Test accuracy | Training time |
|---|---|---|
| Logistic regression | **85.5%** | ~75 s |
| Decision tree | 81.1% | ~29 s |

Random guessing would get about 10%. The full write-up is in [MAI500-report.md](MAI500-report.md).

## How to run

You need [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
uv sync
uv run train.py
```

On the first run the dataset is downloaded from OpenML, which can take a while. The script prints the test accuracy and a per-class report for each model.

> **Windows note:** if `uv sync` fails while installing packages, the project path may be longer than Windows allows. Move the project to a shorter path, or turn on long-path support.

## Files

| File | Contents |
|---|---|
| `train.py` | Loads the data, prepares it, and trains and evaluates both models |
| `pyproject.toml`, `uv.lock` | Project dependencies (scikit-learn, pandas) |
| `MAI500-report.md` | Submitted report, with screenshots in `report-media/` |
