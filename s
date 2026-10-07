import os
import re

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_DIR = r"C:\Users\tolog\Desktop\gig"
WAV_FILE = r"C:\Users\tolog\Desktop\test.wav"

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "v3_rnnt_encoder.int8.onnx"
)

DECODER_FILE = os.path.join(
    MODEL_DIR,
    "v3_rnnt_decoder.int8.onnx"
)

JOINT_FILE = os.path.join(
    MODEL_DIR,
    "v3_rnnt_joint.int8.onnx"
)

VOCAB_FILE = os.path.join(
    MODEL_DIR,
    "v3_vocab.txt"
)


# ============================================================
# GigaAM v3 RNNT
# ============================================================

SAMPLE_RATE = 16000

N_MELS = 64
N_FFT = 320
WIN_LENGTH = 320
HOP_LENGTH = 160

ENCODER_DIM = 768
DECODER_DIM = 320

BLANK_ID = 33

MAX_SYMBOLS_PER_STEP = 20


# ============================================================
# VOCAB
# ============================================================

vocab = []

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\r\n")

        # Файл имеет формат:
        #
        # ▁ 0
        # а 1
        # б 2
        #
        # Берём только символ до последнего пробела + номера.
        #
        # Для обычных символов это просто первый символ.
        # Для ▁ тоже получаем ▁.
        #
        match = re.match(r"^(.*?)\s+\d+$", line)

        if match:
            token = match.group(1)
        else:
            token = line

        vocab.append(token)


# ============================================================
# ONNX
# ============================================================

providers = [
    "CPUExecutionProvider"
]

encoder = ort.InferenceSession(
    ENCODER_FILE,
    providers=providers
)

decoder = ort.InferenceSession(
    DECODER_FILE,
    providers=providers
)

joint = ort.InferenceSession(
    JOINT_FILE,
    providers=providers
)


# ============================================================
# WAV
# ============================================================

audio, sr = sf.read(
    WAV_FILE,
    dtype="float32",
    always_2d=True
)


# Только правый канал
right = audio[:, 1]


# ============================================================
# RESAMPLE 8 -> 16 kHz
# ============================================================

if sr != SAMPLE_RATE:

    right = librosa.resample(
        right,
        orig_sr=sr,
        target_sr=SAMPLE_RATE,
        res_type="kaiser_best"
    )

    sr = SAMPLE_RATE


right = right.astype(
    np.float32
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

mel = librosa.feature.melspectrogram(
    y=right,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window="hann",
    center=False,
    power=2.0,
    n_mels=N_MELS,
    fmin=0,
    fmax=8000,
    htk=True,
    norm=None
)


# Log Mel
mel = np.log(
    np.maximum(
        mel,
        1e-10
    )
).astype(np.float32)


# [1, 64, frames]
features = mel[np.newaxis, :, :]


feature_length = np.array(
    [features.shape[2]],
    dtype=np.int64
)


# ============================================================
# ENCODER
# ============================================================

encoded, encoded_len = encoder.run(
    None,
    {
        "audio_signal": features,
        "length": feature_length
    }
)


encoded = encoded.astype(
    np.float32
)

num_frames = int(
    np.asarray(encoded_len).reshape(-1)[0]
)


# ============================================================
# DECODER
# ============================================================

def decoder_run(token, h, c):

    x = np.array(
        [[token]],
        dtype=np.int64
    )

    dec, new_h, new_c = decoder.run(
        None,
        {
            "x": x,
            "h.1": h,
            "c.1": c
        }
    )

    return dec, new_h, new_c


# ============================================================
# JOINT
# ============================================================

def joint_run(enc_vector, dec_vector):

    enc_input = enc_vector.reshape(
        1,
        ENCODER_DIM,
        1
    ).astype(np.float32)

    dec_input = dec_vector.reshape(
        1,
        DECODER_DIM,
        1
    ).astype(np.float32)

    result = joint.run(
        None,
        {
            "enc": enc_input,
            "dec": dec_input
        }
    )

    return result[0].reshape(-1)


# ============================================================
# RNNT GREEDY DECODING
# ============================================================

h = np.zeros(
    (1, 1, DECODER_DIM),
    dtype=np.float32
)

c = np.zeros(
    (1, 1, DECODER_DIM),
    dtype=np.float32
)


# Начальное состояние prediction network
decoder_output, _, _ = decoder_run(
    BLANK_ID,
    h,
    c
)


tokens = []


for t in range(num_frames):

    enc_vector = encoded[
        0,
        :,
        t
    ]

    symbols = 0

    while symbols < MAX_SYMBOLS_PER_STEP:

        logits = joint_run(
            enc_vector,
            decoder_output
        )

        token_id = int(
            np.argmax(logits)
        )


        # BLANK
        if token_id == BLANK_ID:
            break


        # Некорректный token
        if token_id < 0 or token_id >= len(vocab):
            break


        # Сохраняем token
        tokens.append(token_id)


        # Обновляем decoder
        decoder_output, h, c = decoder_run(
            token_id,
            h,
            c
        )


        symbols += 1


# ============================================================
# TOKENS -> TEXT
# ============================================================

text = "".join(
    vocab[token_id]
    for token_id in tokens
)


# ▁ = начало нового слова / пробел
text = text.replace(
    "▁",
    " "
)


# Убираем лишние пробелы
text = re.sub(
    r"\s+",
    " ",
    text
).strip()


# ============================================================
# КРАСИВЫЙ ВЫВОД
# ============================================================

print()
print("=" * 70)
print("ТРАНСКРИПЦИЯ — ПРАВЫЙ КАНАЛ")
print("=" * 70)
print()

print(text)

print()
print("=" * 70)
