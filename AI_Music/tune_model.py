
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5],
    "class_weight": ["balanced", "balanced_subsample"]
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1_macro",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nПОЧАТОК ПІДБОРУ ПАРАМЕТРІВ")
print("=" * 55)

search.fit(X_train, y_train)

best_model = search.best_estimator_

baseline_predictions = baseline.predict(X_test)
tuned_predictions = best_model.predict(X_test)

print("\n" + "=" * 55)
print("          РЕЗУЛЬТАТИ НАЛАШТУВАННЯ")
print("=" * 55)

print("\nНайкращі параметри:")
for name, value in search.best_params_.items():
    print(f"{name:<22}: {value}")

print("\nРЕЗУЛЬТАТИ ПОЧАТКОВОЇ МОДЕЛІ")
print("-" * 55)
print(f"Accuracy:  {accuracy_score(y_test, baseline_predictions) * 100:.2f}%")
print(f"Macro F1:  {f1_score(y_test, baseline_predictions, average='macro'):.4f}")

print("\nРЕЗУЛЬТАТИ НАЛАШТОВАНОЇ МОДЕЛІ")
print("-" * 55)
print(f"Accuracy:  {accuracy_score(y_test, tuned_predictions) * 100:.2f}%")
print(f"Macro F1:  {f1_score(y_test, tuned_predictions, average='macro'):.4f}")

print("\nОЦІНКА НАЛАШТОВАНОЇ МОДЕЛІ")
print("-" * 55)
print(classification_report(y_test, tuned_predictions, zero_division=0))

joblib.dump(best_model, "AI_Music/music_emotion_model_tuned.pkl")

print("\nНалаштовану модель збережено:")
print("AI_Music/music_emotion_model_tuned.pkl")
print("=" * 55)
