import librosa
import matplotlib.pyplot as plt
import numpy as np

file_path = 'song.wav.wav'

start_time = 45
duration_time = 15

# Завантаження аудіо
print(f"Завантаження приспіву (з {start_time}-ї по {start_time + duration_time}-у секунду)...")

y, sr = librosa.load(
    file_path,
    offset=start_time,
    duration=duration_time
)

print(f"Успішно завантажено! Частота дискретизації: {sr} Гц")


# ==========================================
# 1. АНАЛІЗ ОРИГІНАЛЬНОГО СИГНАЛУ
# ==========================================

amplitude = np.max(np.abs(y))
rms = np.sqrt(np.mean(y ** 2))
energy = np.sum(y ** 2)

print("\n--- Аналіз оригінального сигналу ---")
print(f"Максимальна амплітуда: {amplitude:.6f}")
print(f"RMS: {rms:.6f}")
print(f"Енергія сигналу: {energy:.6f}")


# ==========================================
# 2. ГРАФІК ОРИГІНАЛЬНОГО СИГНАЛУ
# ==========================================

plt.figure(figsize=(11, 5))

plt.plot(
    np.linspace(0, duration_time, len(y)),
    y
)

plt.title(f'Оригінальний сигнал ({start_time}–{start_time + duration_time} с)')
plt.xlabel('Час (секунди)')
plt.ylabel('Амплітуда')
plt.grid()

plt.savefig(
    'original_signal.png',
    dpi=300
)

print("Графік оригінального сигналу збережено!")

plt.show()


# ==========================================
# 3. ШВИДКЕ ПЕРЕТВОРЕННЯ ФУР'Є (FFT)
# ==========================================

N = len(y)

# FFT
fft_result = np.fft.fft(y)

# Частоти
frequencies = np.fft.fftfreq(N, 1 / sr)

# Амплітуда спектра
fft_amplitude = np.abs(fft_result) / N

# Залишаємо тільки додатні частоти
positive_frequencies = frequencies[:N // 2]
positive_amplitude = fft_amplitude[:N // 2]

# Домінуюча частота
max_frequency_index = np.argmax(positive_amplitude)

max_frequency = positive_frequencies[max_frequency_index]
max_frequency_amplitude = positive_amplitude[max_frequency_index]

print("\n--- Аналіз частотного складу ---")
print(f"Домінуюча частота: {max_frequency:.2f} Гц")
print(f"Амплітуда на цій частоті: {max_frequency_amplitude:.6f}")


# ==========================================
# 4. ПОБУДОВА СПЕКТРА
# ==========================================

plt.figure(figsize=(11, 5))

plt.plot(
    positive_frequencies,
    positive_amplitude
)

plt.title('Спектр оригінального сигналу')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплітуда')
plt.xlim(0, 5000)
plt.grid()

plt.savefig(
    'spectrum_original.png',
    dpi=300
)

print("Спектр збережено як 'spectrum_original.png'!")

plt.show()
# ==========================================
# 6. НИЗЬКОЧАСТОТНА ФІЛЬТРАЦІЯ
# ==========================================

from scipy.signal import butter, filtfilt

# Гранична частота фільтра
cutoff = 1000

# Порядок фільтра
order = 4

# Створення Butterworth-фільтра
b, a = butter(
    order,
    cutoff / (sr / 2),
    btype='low'
)

# Фільтрація сигналу
y_filtered = filtfilt(b, a, y)

print("\n--- Низькочастотна фільтрація ---")
print(f"Гранична частота: {cutoff} Гц")
print(f"Порядок фільтра: {order}")


# Графік відфільтрованого сигналу
plt.figure(figsize=(11, 5))

plt.plot(
    np.linspace(0, duration_time, len(y_filtered)),
    y_filtered
)

plt.title(f'Сигнал після низькочастотної фільтрації (до {cutoff} Гц)')
plt.xlabel('Час (секунди)')
plt.ylabel('Амплітуда')
plt.grid()

plt.savefig(
    'filtered_signal.png',
    dpi=300
)

print("Графік відфільтрованого сигналу збережено!")

plt.show()
# ==========================================
# FFT ВІДФІЛЬТРОВАНОГО СИГНАЛУ
# ==========================================

N_filtered = len(y_filtered)

# FFT відфільтрованого сигналу
fft_filtered = np.fft.fft(y_filtered)

# Частоти
frequencies_filtered = np.fft.fftfreq(
    N_filtered,
    1 / sr
)

# Амплітуда спектра
amplitude_filtered = np.abs(fft_filtered) / N_filtered

# Тільки додатні частоти
positive_frequencies_filtered = frequencies_filtered[:N_filtered // 2]
positive_amplitude_filtered = amplitude_filtered[:N_filtered // 2]

# Домінуюча частота
max_index_filtered = np.argmax(
    positive_amplitude_filtered
)

dominant_frequency_filtered = positive_frequencies_filtered[
    max_index_filtered
]

dominant_amplitude_filtered = positive_amplitude_filtered[
    max_index_filtered
]

print("\n--- Аналіз відфільтрованого сигналу ---")
print(
    f"Домінуюча частота: "
    f"{dominant_frequency_filtered:.2f} Гц"
)

print(
    f"Амплітуда на цій частоті: "
    f"{dominant_amplitude_filtered:.6f}"
)


# ==========================================
# СПЕКТР ВІДФІЛЬТРОВАНОГО СИГНАЛУ
# ==========================================

plt.figure(figsize=(11, 5))

plt.plot(
    positive_frequencies_filtered,
    positive_amplitude_filtered
)

plt.title('Спектр відфільтрованого сигналу')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплітуда')
plt.xlim(0, 5000)
plt.grid()

plt.savefig(
    'spectrum_filtered.png',
    dpi=300
)

print("Спектр відфільтрованого сигналу збережено!")

plt.show()
# ==========================================
# ПОРІВНЯННЯ СПЕКТРІВ
# ==========================================

plt.figure(figsize=(11, 5))

plt.plot(
    positive_frequencies,
    positive_amplitude,
    label='Оригінальний сигнал'
)

plt.plot(
    positive_frequencies_filtered,
    positive_amplitude_filtered,
    label='Відфільтрований сигнал'
)

plt.title('Порівняння спектрів сигналу')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплітуда')
plt.xlim(0, 5000)
plt.legend()
plt.grid()

plt.savefig(
    'spectrum_comparison.png',
    dpi=300
)

print("Порівняльний спектр збережено!")

plt.show()
# ==========================================
# СПЕКТРОГРАМА ОРИГІНАЛЬНОГО СИГНАЛУ
# ==========================================

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# STFT
D = librosa.stft(y)

# Переведення амплітуди в децибели
S_db = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)

# Побудова спектрограми
plt.figure(figsize=(11, 5))

librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis='time',
    y_axis='hz'
)

plt.colorbar(
    format='%+2.0f дБ'
)

plt.title('Спектрограма оригінального сигналу')
plt.xlabel('Час (секунди)')
plt.ylabel('Частота (Гц)')

plt.ylim(0, 5000)

plt.savefig(
    'spectrogram_original.png',
    dpi=300
)

print("Спектрограму оригінального сигналу збережено!")

plt.show()
# ==========================================
# СПЕКТРОГРАМА ВІДФІЛЬТРОВАНОГО СИГНАЛУ
# ==========================================

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# STFT відфільтрованого сигналу
D_filtered = librosa.stft(y_filtered)

# Переведення амплітуди в децибели
S_db_filtered = librosa.amplitude_to_db(
    np.abs(D_filtered),
    ref=np.max
)

# Побудова спектрограми
plt.figure(figsize=(11, 5))

librosa.display.specshow(
    S_db_filtered,
    sr=sr,
    x_axis='time',
    y_axis='hz'
)

plt.colorbar(
    format='%+2.0f дБ'
)

plt.title('Спектрограма відфільтрованого сигналу')
plt.xlabel('Час (секунди)')
plt.ylabel('Частота (Гц)')

plt.ylim(0, 5000)

plt.savefig(
    'spectrogram_filtered.png',
    dpi=300
)

print("Спектрограму відфільтрованого сигналу збережено!")

plt.show()
# ==========================================
# ПОРІВНЯННЯ СПЕКТРОГРАМ
# ==========================================

plt.figure(figsize=(11, 5))

plt.subplot(2, 1, 1)

librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis='time',
    y_axis='hz'
)

plt.title('Оригінальний сигнал')
plt.ylabel('Частота (Гц)')
plt.ylim(0, 5000)
plt.colorbar(format='%+2.0f дБ')


plt.subplot(2, 1, 2)

librosa.display.specshow(
    S_db_filtered,
    sr=sr,
    x_axis='time',
    y_axis='hz'
)

plt.title('Відфільтрований сигнал')
plt.xlabel('Час (секунди)')
plt.ylabel('Частота (Гц)')
plt.ylim(0, 5000)
plt.colorbar(format='%+2.0f дБ')

plt.tight_layout()

plt.savefig(
    'spectrogram_comparison.png',
    dpi=300
)

print("Порівняння спектрограм збережено!")

plt.show()
# ==========================================
# АНАЛІЗ ВІДФІЛЬТРОВАНОГО СИГНАЛУ
# ==========================================

amplitude_filtered = np.max(np.abs(y_filtered))
rms_filtered = np.sqrt(np.mean(y_filtered ** 2))
energy_filtered = np.sum(y_filtered ** 2)

print("\n--- Аналіз відфільтрованого сигналу ---")
print(f"Максимальна амплітуда: {amplitude_filtered:.6f}")
print(f"RMS: {rms_filtered:.6f}")
print(f"Енергія сигналу: {energy_filtered:.6f}")
# ==========================================
# ПОРІВНЯННЯ МЕТРИК
# ==========================================

amplitude_change = (
    (amplitude_filtered - amplitude) / amplitude
) * 100

rms_change = (
    (rms_filtered - rms) / rms
) * 100

energy_change = (
    (energy_filtered - energy) / energy
) * 100

print("\n--- Порівняння оригінального та відфільтрованого сигналів ---")

print(f"Амплітуда: {amplitude:.6f} → {amplitude_filtered:.6f}")
print(f"Зміна амплітуди: {amplitude_change:.2f}%")

print(f"RMS: {rms:.6f} → {rms_filtered:.6f}")
print(f"Зміна RMS: {rms_change:.2f}%")

print(f"Енергія: {energy:.6f} → {energy_filtered:.6f}")
print(f"Зміна енергії: {energy_change:.2f}%")









# ==========================================
# AI-АНАЛІЗ ОРИГІНАЛЬНОГО ТА ВІДФІЛЬТРОВАНОГО СИГНАЛІВ
# ==========================================

import pandas as pd
import opensmile
import joblib

model_path = "AI_Music/music_emotion_model_tuned.pkl"
dataset_path = "AI_Music/DEAM_data/model_dataset.csv"

print("\n" + "=" * 65)
print("       ПОРІВНЯННЯ AI-АНАЛІЗУ ДВОХ СИГНАЛІВ")
print("=" * 65)

# Завантаження навченої моделі
model = joblib.load(model_path)
expected_features = list(model.feature_names_in_)

# Середні значення навчального датасету
training_data = pd.read_csv(dataset_path)
training_means = training_data[expected_features].mean()

# Налаштування openSMILE
smile_lld = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS13,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)

smile_delta = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS13,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas
)

def analyze_audio(signal, signal_name):

    all_features = []
    window_samples = int(sr * 0.5)

    for start in range(0, len(signal), window_samples):

        segment = signal[start:start + window_samples]

        if len(segment) < window_samples:
            continue

        lld = smile_lld.process_signal(segment, sr)
        delta = smile_delta.process_signal(segment, sr)

        row = {}

        for data in [lld, delta]:

            for column in data.columns:

                name = str(column)

                if name.startswith("mfcc_sma"):
                    name = "pcm_fftMag_" + name

                row[name + "_amean"] = data[column].mean()
                row[name + "_stddev"] = data[column].std(ddof=0)

        all_features.append(row)

    features_df = pd.DataFrame(all_features)
    song_features = features_df.mean().to_frame().T

    X = song_features.reindex(columns=expected_features)
    X = X.fillna(training_means)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    print(f"\n{signal_name}")
    print("-" * 65)
    print(f"Визначена емоція: {prediction.upper()}")

    results = {}

    for emotion, probability in zip(model.classes_, probabilities):
        results[emotion] = probability * 100

    return results


# Аналіз оригінального сигналу
original_results = analyze_audio(
    y,
    "ОРИГІНАЛЬНИЙ СИГНАЛ"
)

# Аналіз відфільтрованого сигналу
filtered_results = analyze_audio(
    y_filtered,
    "СИГНАЛ ПІСЛЯ ФІЛЬТРАЦІЇ ДО 1000 ГЦ"
)

# Порівняння результатів
print("\n" + "=" * 65)
print("                 ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
print("=" * 65)

print(
    f"{'Емоція':<20}"
    f"{'Оригінал':>15}"
    f"{'Фільтрований':>18}"
)

print("-" * 65)

for emotion in model.classes_:

    print(
        f"{emotion:<20}"
        f"{original_results[emotion]:>14.2f}%"
        f"{filtered_results[emotion]:>17.2f}%"
    )

print("-" * 65)

original_emotion = max(original_results, key=original_results.get)
filtered_emotion = max(filtered_results, key=filtered_results.get)

print(f"\nЕмоція оригінального сигналу: {original_emotion.upper()}")
print(f"Емоція після фільтрації: {filtered_emotion.upper()}")

if original_emotion == filtered_emotion:
    print("\nВисновок: фільтрація не змінила визначену емоційну категорію.")
else:
    print("\nВисновок: після фільтрації визначена емоційна категорія змінилася.")

print("\nAI-аналіз завершено.")
