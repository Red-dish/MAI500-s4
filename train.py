"""Classify Fashion-MNIST clothing images with two supervised learning models.

Run with:  uv run train.py
"""

from time import perf_counter

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# 1. Load the data: 70,000 images of 28x28 pixels, flattened to 784 numbers each.
print("Loading Fashion-MNIST (70,000 images of 28x28 pixels)...")
X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True)

# 2. Prepare the data: scale pixels from 0-255 to 0-1, and hold out a test set.
X = X / 255.0
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10_000, stratify=y, random_state=42
)
print(f"Training images: {len(X_train):,} | Test images: {len(X_test):,}")

# 3. Choose the models to compare.
models = {
    "Decision tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Logistic regression": LogisticRegression(max_iter=200, random_state=42),
}

for name, model in models.items():
    # 4. Train the model.
    print(f"\nTraining {name}...")
    start = perf_counter()
    model.fit(X_train, y_train)
    print(f"Finished in {perf_counter() - start:.0f} seconds")

    # 5. Evaluate it on images it has never seen.
    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=3))
