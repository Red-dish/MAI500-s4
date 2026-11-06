# Fashion-MNIST Classification: Work Done and Results

## Summary

I trained two supervised learning models to classify images of clothing into 10 categories. Logistic regression performed best, with **85.5% test accuracy**, compared with **81.1%** for a decision tree. With 10 equally common classes, random guessing would score about 10%.

## Work done

The code is in `train.py` and follows the steps from the activity brief.

### 1. Dataset

I used [Fashion-MNIST](https://www.openml.org/d/40996), downloaded from OpenML with scikit-learn. It has 70,000 grayscale images of 28×28 pixels, split equally across 10 classes:

T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.

### 2. Data preparation

- **Features:** each image is stored as 784 numbers, one per pixel.
- **Scaling:** pixel values were divided by 255, changing their range from 0–255 to 0–1.
- **Split:** 60,000 images were used for training and 10,000 for testing. The split is random but repeatable (`random_state=42`), and it keeps the classes balanced (`stratify=y`).

### 3. Model selection

| Model | Settings | Why I included it |
|---|---|---|
| Decision tree | `max_depth=10` | Learns simple if/else rules, and limiting the depth helps reduce overfitting |
| Logistic regression | `max_iter=200` | A standard linear classifier for comparison |

These settings were picked by hand and not tuned.

### 4. Training

Each model was trained on the 60,000 training images using the CPU. The GPU was not needed.

### 5. Evaluation

Each model was scored on the 10,000 test images, which it did not see during training.

## Results

### Overall

| Model | Test accuracy | Macro F1 | Training time |
|---|---|---:|---:|
| **Logistic regression** | **85.5%** | 0.853 | 75 seconds |
| Decision tree | 81.1% | 0.810 | 29 seconds |

The whole script took about 2 minutes to run, including loading the data. Training times will vary between computers.

### By class

Each class has 1,000 test images.

| Class | Tree precision | Tree recall | Tree F1 | LogReg precision | LogReg recall | LogReg F1 |
|---|---:|---:|---:|---:|---:|---:|
| T-shirt/top | 0.763 | 0.794 | 0.778 | 0.787 | 0.830 | 0.808 |
| Trouser | 0.977 | 0.929 | 0.952 | 0.975 | 0.960 | 0.967 |
| Pullover | 0.702 | 0.700 | 0.701 | 0.771 | 0.739 | 0.754 |
| Dress | 0.810 | 0.824 | 0.817 | 0.838 | 0.851 | 0.844 |
| Coat | 0.632 | 0.791 | 0.703 | 0.740 | 0.817 | 0.777 |
| Sandal | 0.907 | 0.888 | 0.897 | 0.949 | 0.940 | 0.944 |
| Shirt | 0.600 | 0.459 | 0.520 | 0.641 | 0.565 | 0.601 |
| Sneaker | 0.871 | 0.902 | 0.886 | 0.926 | 0.937 | 0.931 |
| Bag | 0.936 | 0.917 | 0.926 | 0.959 | 0.953 | 0.956 |
| Ankle boot | 0.925 | 0.910 | 0.917 | 0.950 | 0.953 | 0.952 |

## Understanding the output

- **Accuracy:** the share of all test images labeled correctly.
- **Precision:** of the images predicted as a class, the share that really were that class.
- **Recall:** of the images that really were a class, the share the model found.
- **F1-score:** a single score that balances precision and recall, from 0 to 1.
- **Support:** the number of test images in that class (1,000 each).
- **Macro average:** the plain average across all classes. It is almost the same as the weighted average here because the classes are balanced.

For example, logistic regression has a recall of 0.565 for Shirt, so it correctly identified 565 of the 1,000 real shirts. Its precision of 0.641 means about 64% of the images it labeled "Shirt" really were shirts.

## Findings

- **Logistic regression was stronger overall:** it had a higher F1-score than the decision tree in all 10 classes.
- **The decision tree was faster:** it trained about 2.5 times faster, but was 4.4 percentage points less accurate.
- **Some classes were easiest:** Trouser, Bag, Ankle boot and Sandal scored highest with logistic regression (F1 of 0.94–0.97), probably because their shapes are distinctive.
- **Shirt was the hardest class:** it had the lowest score for both models (F1 of 0.60 and 0.52). Pullover, Coat and T-shirt/top also scored lower, so these similar upper-body garments may be getting confused with each other. A confusion matrix would be needed to confirm this.

## Limitations

- **Logistic regression did not fully converge:** training printed a `ConvergenceWarning` because it reached the 200-iteration limit. The 85.5% is a real test score, but more iterations could change it slightly.
- **The models were not tuned:** other settings, such as a different tree depth, could give different results.
- **Pixel order is ignored:** both models treat each image as a flat list of pixel values, so they do not directly use its spatial structure the way a CNN does.
- **The split is not the official one:** I used a random split, so these scores are not directly comparable with benchmarks that use the official Fashion-MNIST test set.

## How to reproduce

The project uses `uv`. From this folder, run:

```bash
uv run train.py
```

Results came from Python 3.14.2, scikit-learn 1.9.1 and pandas 2.3.3 on Windows 11. pandas is pinned below version 3 because scikit-learn's default dataset parser crashed on this dataset under pandas 3.1.1.
