import os
import csv
import wave

# Папка с аудиофайлами
folder = r"C:\Users\tolog\Desktop\audio"

# Куда сохранить CSV
output_csv = os.path.join(folder, "audio_duration.csv")

rows = []

for filename in os.listdir(folder):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(folder, filename)

        try:
            with wave.open(filepath, "rb") as audio:
                frames = audio.getnframes()
                framerate = audio.getframerate()
                duration = frames / framerate

            rows.append([filename, round(duration, 2)])

        except Exception as e:
            print(f"Ошибка: {filename} — {e}")

# Записываем CSV
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Файл", "Длительность, сек"])
    writer.writerows(rows)

print(f"Готово. CSV сохранен: {output_csv}")
