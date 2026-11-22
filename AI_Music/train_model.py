
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 55)
print("          РЕЗУЛЬТАТИ НАВЧАННЯ AI-МОДЕЛІ")
print("=" * 55)

print(f"\nКількість навчальних композицій: {len(X_train)}")
print(f"Кількість тестових композицій:   {len(X_test)}")
print(f"Кількість характеристик:         {X.shape[1]}")
print(f"Кількість дерев:                 {model.n_estimators}")

print(f"\nТочність моделі: {accuracy * 100:.2f}%")

print("\nОЦІНКА ДЛЯ КОЖНОЇ ЕМОЦІЇ")
print("-" * 55)
print(classification_report(y_test, predictions, zero_division=0))

print("МАТРИЦЯ ПОМИЛОК")
print("-" * 55)
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "AI_Music/music_emotion_model.pkl")

print("\nМодель збережено: AI_Music/music_emotion_model.pkl")
print("=" * 55)
