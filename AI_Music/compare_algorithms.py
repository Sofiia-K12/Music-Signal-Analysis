
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Завантаження датасету
data = pd.read_csv("AI_Music/DEAM_data/model_dataset.csv")

X = data.drop(
    columns=["song_id", "valence_mean", "arousal_mean", "emotion"]
)

y = data["emotion"]

X = X.replace([np.inf, -np.inf], np.nan)

# Однаковий поділ даних для всіх моделей
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest (налаштована)": joblib.load(
        "AI_Music/music_emotion_model_tuned.pkl"
    ),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=500,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            C=3,
            kernel="rbf",
            class_weight="balanced"
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])
}

results = []
best_score = -1
best_name = None
best_model = None

print("\n" + "=" * 65)
print("          ПОРІВНЯННЯ АЛГОРИТМІВ AI-КЛАСИФІКАЦІЇ")
print("=" * 65)

for name, model in models.items():

    print(f"\nНавчання: {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Модель": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    if macro_f1 > best_score:
        best_score = macro_f1
        best_name = name
        best_model = model

    print("\nРезультати за категоріями:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# Загальна таблиця
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n" + "=" * 65)
print("                    ПІДСУМКОВЕ ПОРІВНЯННЯ")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format
        }
    )
)

# Збереження результатів
results_df.to_csv(
    "AI_Music/algorithm_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# Збереження найкращої моделі окремо
joblib.dump(
    best_model,
    "AI_Music/music_emotion_model_best.pkl"
)

print("\n" + "=" * 65)
print(f"НАЙКРАЩА МОДЕЛЬ: {best_name}")
print(f"MACRO F1: {best_score:.4f}")
print("Модель збережено: music_emotion_model_best.pkl")
print("=" * 65)
