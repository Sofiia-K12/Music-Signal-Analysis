
import pandas as pd
from pathlib import Path

features_folder = Path(r"D:\DDownloads\archive\features\features")
annotations_file = Path("AI_Music/DEAM_data/annotations_labeled.csv")

annotations = pd.read_csv(annotations_file)

all_features = []
missing = []

print("ПОЧАТОК ОБРОБКИ АУДІОХАРАКТЕРИСТИК")
print("=" * 45)

for song_id in annotations["song_id"]:
    file_path = features_folder / f"{int(song_id)}.csv"

    if not file_path.exists():
        missing.append(song_id)
        continue

    try:
        data = pd.read_csv(file_path, sep=";", header=0)

        data.columns = data.columns.str.strip()

        data = data.drop(columns=["frameTime"], errors="ignore")

        data = data.apply(pd.to_numeric, errors="coerce")

        mean_features = data.mean()

        row = mean_features.to_dict()
        row["song_id"] = int(song_id)

        all_features.append(row)

    except Exception as error:
        print(f"Помилка у файлі {song_id}.csv: {error}")

features_df = pd.DataFrame(all_features)

result = annotations.merge(
    features_df,
    on="song_id",
    how="inner"
)

output_file = "AI_Music/DEAM_data/model_dataset.csv"

result.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 45)
print("РЕЗУЛЬТАТИ ОБРОБКИ")
print("=" * 45)
print(f"Композицій в анотаціях: {len(annotations)}")
print(f"Оброблено композицій: {len(features_df)}")
print(f"Відсутні аудіофайли: {len(missing)}")
print(f"Підсумкова кількість: {len(result)}")
print(f"Кількість характеристик: {len(features_df.columns) - 1}")
print("=" * 45)
print(f"\nФайл збережено: {output_file}")
