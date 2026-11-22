
import librosa
import numpy as np
import pandas as pd
import opensmile
import joblib

# Шляхи до файлів
audio_path = "song.wav.wav"
model_path = "AI_Music/music_emotion_model_tuned.pkl"
dataset_path = "AI_Music/DEAM_data/model_dataset.csv"

# Параметри аудіофрагмента
start_time = 45
duration = 15
window_duration = 0.5

print("=" * 55)
print("       AI-АНАЛІЗ ЕМОЦІЙНОГО ЗАБАРВЛЕННЯ МУЗИКИ")
print("=" * 55)

# Завантаження аудіо
print("\nЗавантаження аудіофрагмента...")

y, sr = librosa.load(
    audio_path,
    sr=22050,
    offset=start_time,
    duration=duration
)

print(f"Фрагмент: {start_time}-{start_time + duration} секунд")
print(f"Частота дискретизації: {sr} Гц")

# Налаштування openSMILE
smile_lld = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS13,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)

smile_delta = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS13,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas
)

# Обробка фрагмента частинами по 0,5 секунди
all_features = []
window_samples = int(sr * window_duration)

print("\nОбчислення аудіохарактеристик...")

for start in range(0, len(y), window_samples):

    segment = y[start:start + window_samples]

    if len(segment) < window_samples:
        continue

    lld = smile_lld.process_signal(segment, sr)
    delta = smile_delta.process_signal(segment, sr)

    row = {}

    for data in [lld, delta]:

        for column in data.columns:

            name = str(column)

            # Узгодження назви MFCC з навчальним датасетом
            if name.startswith("mfcc_sma"):
                name = "pcm_fftMag_" + name

            row[name + "_amean"] = data[column].mean()
            row[name + "_stddev"] = data[column].std(ddof=0)

    all_features.append(row)

# Формування таблиці характеристик
features_df = pd.DataFrame(all_features)

song_features = features_df.mean().to_frame().T

print(f"Оброблено інтервалів: {len(all_features)}")
print(f"Отримано характеристик: {song_features.shape[1]}")

# Завантаження моделі
print("\nЗавантаження AI-моделі...")

model = joblib.load(model_path)

# Отримання правильного порядку характеристик
expected_features = list(model.feature_names_in_)

missing = [
    column for column in expected_features
    if column not in song_features.columns
]

if missing:
    print("\nПОМИЛКА: не вистачає характеристик:")
    for column in missing:
        print(column)
    raise ValueError("Набір характеристик не збігається з моделлю.")

# Впорядкування характеристик
X = song_features.reindex(columns=expected_features)

# Заповнення можливих пропусків середніми значеннями навчального датасету
training_data = pd.read_csv(dataset_path)
training_means = training_data[expected_features].mean()

X = X.fillna(training_means)

# Прогнозування
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print("\n" + "=" * 55)
print("              РЕЗУЛЬТАТ AI-АНАЛІЗУ")
print("=" * 55)

print(f"\nВизначена емоція: {prediction.upper()}")

print("\nОЦІНКИ МОДЕЛІ:")
print("-" * 55)

for emotion, probability in zip(model.classes_, probabilities):

    percent = probability * 100

    print(f"{emotion:<20} {percent:>7.2f}%")

print("-" * 55)
print("\nАналіз завершено.")


