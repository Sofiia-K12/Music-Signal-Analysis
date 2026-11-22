
import pandas as pd

data = pd.read_csv("AI_Music/DEAM_data/annotations_all.csv")

def classify_emotion(row):
    valence = row["valence_mean"]
    arousal = row["arousal_mean"]

    if valence >= 5 and arousal >= 5:
        return "щаслива"
    elif valence < 5 and arousal < 5:
        return "сумна"
    elif valence < 5 and arousal >= 5:
        return "енергійна"
    else:
        return "спокійна"


data["emotion"] = data.apply(classify_emotion, axis=1)

data.to_csv(
    "AI_Music/DEAM_data/annotations_labeled.csv",
    index=False,
    encoding="utf-8-sig"
)

counts = data["emotion"].value_counts()
emotions = ["сумна", "щаслива", "спокійна", "енергійна"]

print()
print("╔" + "═" * 48 + "╗")
print("║" + " РЕЗУЛЬТАТИ АНАЛІЗУ МУЗИЧНИХ КОМПОЗИЦІЙ ".center(48) + "║")
print("╠" + "═" * 48 + "╣")
print("║" + " Емоційна категорія".ljust(32) + "Кількість".rjust(16) + "║")
print("╠" + "─" * 48 + "╣")

for emotion in emotions:
    count = counts.get(emotion, 0)
    line = f" {emotion.capitalize():<31}{count:>16}"
    print("║" + line + "║")

print("╠" + "═" * 48 + "╣")
line = f" Усього композицій{'':<14}{len(data):>16}"
print("║" + line + "║")
print("╚" + "═" * 48 + "╝")

print("\nФайл annotations_labeled.csv успішно створено.")
