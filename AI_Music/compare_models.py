
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    ConfusionMatrixDisplay
)

data = pd.read_csv("AI_Music/DEAM_data/model_dataset.csv")

X = data.drop(columns=[
    "song_id",
    "valence_mean",
    "arousal_mean",
    "emotion"
])

y = data["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

baseline = joblib.load("AI_Music/music_emotion_model.pkl")
tuned = joblib.load("AI_Music/music_emotion_model_tuned.pkl")

baseline_predictions = baseline.predict(X_test)
tuned_predictions = tuned.predict(X_test)

classes = ["енергійна", "спокійна", "сумна", "щаслива"]

baseline_f1 = f1_score(
    y_test, baseline_predictions,
    labels=classes, average=None
)

tuned_f1 = f1_score(
    y_test, tuned_predictions,
    labels=classes, average=None
)

print("\n" + "=" * 60)
print("          ПОРІВНЯННЯ F1-SCORE")
print("=" * 60)

print(f"{'Емоція':<18}{'Початкова':>15}{'Налаштована':>17}")
print("-" * 60)

for i, emotion in enumerate(classes):
    print(
        f"{emotion:<18}"
        f"{baseline_f1[i]:>15.3f}"
        f"{tuned_f1[i]:>17.3f}"
    )

print("-" * 60)

# Графік порівняння F1-score

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    baseline_f1,
    width,
    label="Початкова модель"
)

plt.bar(
    x + width / 2,
    tuned_f1,
    width,
    label="Налаштована модель"
)

plt.xticks(x, classes)
plt.ylabel("F1-score")
plt.xlabel("Емоційні категорії")
plt.title("Порівняння F1-score моделей Random Forest")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("AI_Music/f1_comparison.png", dpi=300)
plt.show()

# Матриці помилок

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    baseline_predictions,
    labels=classes,
    display_labels=classes,
    cmap="Blues",
    xticks_rotation=25,
    ax=axes[0],
    colorbar=False
)

axes[0].set_title("Початкова модель")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tuned_predictions,
    labels=classes,
    display_labels=classes,
    cmap="Blues",
    xticks_rotation=25,
    ax=axes[1],
    colorbar=False
)

axes[1].set_title("Налаштована модель")

plt.tight_layout()
plt.savefig("AI_Music/confusion_comparison.png", dpi=300)
plt.show()

print("\nГрафіки збережено:")
print("AI_Music/f1_comparison.png")
print("AI_Music/confusion_comparison.png")
