import os
import csv
from pydub import AudioSegment

folder = r"C:\Users\TologonovAB\Desktop\audio"
output_csv = os.path.join(folder, "audio_duration.csv")

rows = []

for filename in os.listdir(folder):
    if filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        filepath = os.path.join(folder, filename)

        try:
            audio = AudioSegment.from_file(filepath)
            duration = len(audio) / 1000

            rows.append([
                filename,
                round(duration, 2)
            ])

            print(f"{filename}: {duration:.2f} сек")

        except Exception as e:
            print(f"Ошибка: {filename} — {e}")

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Файл", "Длительность, сек"])
    writer.writerows(rows)

print(f"\nГотово: {output_csv}")
