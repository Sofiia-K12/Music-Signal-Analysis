
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

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

model = joblib.load("AI_Music/music_emotion_model.pkl")

importance = pd.DataFrame({
    "Характеристика": X.columns,
    "Важливість": model.feature_importances_
})

importance = importance.sort_values(
    by="Важливість",
    ascending=False
)

print("\n" + "=" * 65)
print("          НАЙВАЖЛИВІШІ АУДІОХАРАКТЕРИСТИКИ")
print("=" * 65)

for _, row in importance.head(15).iterrows():
    print(f"{row['Характеристика']:<48} {row['Важливість']:.5f}")

plt.figure(figsize=(12, 7))

top_features = importance.head(15).sort_values(
    by="Важливість"
)

plt.barh(
    top_features["Характеристика"],
    top_features["Важливість"]
)

plt.xlabel("Важливість")
plt.ylabel("Аудіохарактеристика")
plt.title("15 найважливіших аудіохарактеристик")
plt.tight_layout()

plt.savefig("AI_Music/feature_importance.png", dpi=300)
plt.show()

predictions = model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Енергійна", "Спокійна", "Сумна", "Щаслива"],
    cmap="Blues",
    xticks_rotation=20
)

plt.title("Матриця помилок Random Forest")
plt.tight_layout()

plt.savefig("AI_Music/confusion_matrix.png", dpi=300)
plt.show()

print("\nГрафіки збережено:")
print("AI_Music/feature_importance.png")
print("AI_Music/confusion_matrix.png")
