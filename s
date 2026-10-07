import os
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort


# ============================================================
# PATHS
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
# GigaAM v3 RNNT CONFIG
# Берем значения непосредственно из v3_rnnt.yaml
# ============================================================

SAMPLE_RATE = 16000

N_MELS = 64

WIN_LENGTH = 320
HOP_LENGTH = 160
N_FFT = 320

FMIN = 0
FMAX = SAMPLE_RATE // 2

MEL_SCALE = "htk"

CENTER = False

# Из YAML:
# mel_norm: null
#
# Поэтому НИКАКОЙ Slaney normalization
# здесь не используем.

# RNNT
ENCODER_DIM = 768
DECODER_DIM = 320

NUM_CLASSES = 34

# RNNT blank находится после vocabulary.
#
# Vocabulary имеет 33 символа:
#
# 0  пробел
# 1  а
# ...
# 32 ю
#
# 33 я
#
# Но RNNT имеет num_classes=34.
#
# Следовательно дополнительный класс = blank.
#
# В экспортированной модели это нужно проверить.
# Для стандартного RNNT blank обычно последний индекс.
BLANK_ID = 33


# Максимальное количество non-blank символов
# на один encoder frame.
MAX_SYMBOLS_PER_STEP = 20


# ============================================================
# VOCAB
# ============================================================

print("=" * 70)
print("ЗАГРУЗКА VOCAB")
print("=" * 70)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = [line.rstrip("\r\n") for line in f]


print("Количество строк в vocab:", len(vocab))

for i, token in enumerate(vocab):
    print(
        f"{i:2d}: {repr(token)}"
    )


# ============================================================
# ПРОВЕРКА VOCAB
# ============================================================

if len(vocab) != 33:
    print()
    print(
        "ВНИМАНИЕ: ожидалось 33 токена vocabulary, "
        f"получено {len(vocab)}"
    )


print()
print("RNNT NUM_CLASSES:", NUM_CLASSES)
print("BLANK_ID:", BLANK_ID)


# ============================================================
# ONNX
# ============================================================

print()
print("=" * 70)
print("ЗАГРУЗКА ONNX")
print("=" * 70)

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
# MODEL INFO
# ============================================================

def print_model_info(name, session):

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print("INPUTS:")

    for item in session.get_inputs():

        print(
            " ",
            item.name,
            "|",
            item.shape,
            "|",
            item.type
        )

    print("OUTPUTS:")

    for item in session.get_outputs():

        print(
            " ",
            item.name,
            "|",
            item.shape,
            "|",
            item.type
        )


print_model_info(
    "ENCODER",
    encoder
)

print_model_info(
    "DECODER",
    decoder
)

print_model_info(
    "JOINT",
    joint
)


# ============================================================
# READ WAV
# ============================================================

print()
print("=" * 70)
print("ЗАГРУЗКА WAV")
print("=" * 70)

audio, sr = sf.read(
    WAV_FILE,
    dtype="float32",
    always_2d=True
)

print("Sample rate:", sr)
print("Shape:", audio.shape)
print("Channels:", audio.shape[1])


# ============================================================
# RIGHT CHANNEL
# ============================================================

if audio.shape[1] < 2:

    raise RuntimeError(
        "WAV не стерео. "
        "Нужен WAV с двумя каналами."
    )


right = audio[:, 1]

print()
print("Выбран ПРАВЫЙ канал")

print(
    "Duration:",
    len(right) / sr,
    "sec"
)


# ============================================================
# RESAMPLE
# ============================================================

if sr != SAMPLE_RATE:

    print()
    print(
        f"Ресемплинг {sr} -> {SAMPLE_RATE}"
    )

    right = librosa.resample(
        right,
        orig_sr=sr,
        target_sr=SAMPLE_RATE,
        res_type="kaiser_best"
    )

    sr = SAMPLE_RATE

else:

    print()
    print("Ресемплинг не требуется.")


right = right.astype(
    np.float32
)


print(
    "После resample:",
    len(right),
    "samples"
)

print(
    "Duration:",
    len(right) / SAMPLE_RATE,
    "sec"
)


# ============================================================
# FEATURE EXTRACTION
#
# ТОЧНО ПО YAML
#
# features: 64
# win_length: 320
# hop_length: 160
# n_fft: 320
# mel_scale: htk
# mel_norm: null
# center: false
# ============================================================

print()
print("=" * 70)
print("FEATURE EXTRACTION")
print("=" * 70)

print("sample_rate :", SAMPLE_RATE)
print("n_fft       :", N_FFT)
print("win_length  :", WIN_LENGTH)
print("hop_length  :", HOP_LENGTH)
print("n_mels      :", N_MELS)
print("mel_scale   :", MEL_SCALE)
print("mel_norm    :", None)
print("center      :", CENTER)


# librosa:
#
# htk=True
# соответствует mel_scale="htk"
#
# norm=None
# соответствует mel_norm=null
#
# center=False
# соответствует YAML
#

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

    fmin=FMIN,
    fmax=FMAX,

    htk=True,
    norm=None
)


print()
print("Mel shape:", mel.shape)


# ============================================================
# LOG
# ============================================================

# GigaAM FeatureExtractor работает с логарифмическими
# mel features.
#
# Используем натуральный логарифм.

mel = np.log(
    np.maximum(
        mel,
        1e-10
    )
)


mel = mel.astype(
    np.float32
)


# ============================================================
# ENCODER INPUT
# ============================================================

# Требуется:
#
# [batch, 64, seq_len]
#

features = mel[
    np.newaxis,
    :,
    :
]


print()
print("Encoder input:")
print("shape :", features.shape)
print("dtype :", features.dtype)


# ============================================================
# LENGTH
# ============================================================

feature_length = np.array(
    [features.shape[2]],
    dtype=np.int64
)


print(
    "length:",
    feature_length
)


# ============================================================
# ENCODER
# ============================================================

print()
print("=" * 70)
print("ENCODER")
print("=" * 70)

encoder_outputs = encoder.run(
    None,
    {
        "audio_signal": features,
        "length": feature_length
    }
)


encoded = encoder_outputs[0]
encoded_len = encoder_outputs[1]


print(
    "encoded shape:",
    encoded.shape
)

print(
    "encoded_len:",
    encoded_len
)


# ============================================================
# INITIAL RNNT STATE
# ============================================================

print()
print("=" * 70)
print("INITIAL DECODER STATE")
print("=" * 70)


h = np.zeros(
    (
        1,
        1,
        DECODER_DIM
    ),
    dtype=np.float32
)

c = np.zeros(
    (
        1,
        1,
        DECODER_DIM
    ),
    dtype=np.float32
)


# ============================================================
# DECODER
# ============================================================

def decoder_run(
    token,
    h_state,
    c_state
):

    x = np.array(
        [[token]],
        dtype=np.int64
    )

    outputs = decoder.run(
        None,
        {
            "x": x,
            "h.1": h_state,
            "c.1": c_state
        }
    )

    dec = outputs[0]
    new_h = outputs[1]
    new_c = outputs[2]

    return (
        dec,
        new_h,
        new_c
    )


# ============================================================
# JOINT
# ============================================================

def joint_run(
    enc_vector,
    dec_vector
):

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

    output = joint.run(
        None,
        {
            "enc": enc_input,
            "dec": dec_input
        }
    )

    return output[0]


# ============================================================
# RNNT
# ============================================================

print()
print("=" * 70)
print("RNNT GREEDY DECODING")
print("=" * 70)


encoded = encoded.astype(
    np.float32
)


num_frames = int(
    np.asarray(
        encoded_len
    ).reshape(-1)[0]
)


print(
    "Encoder frames:",
    num_frames
)


# ============================================================
# RNNT START
# ============================================================

# Для RNNT нужен initial prediction network output.
#
# В decoder подаем blank.
#
# Важно:
# состояние после blank НЕ нужно использовать как обычный
# emitted token state.
#
# Поэтому отдельно получаем initial decoder output.

decoder_output, _, _ = decoder_run(
    BLANK_ID,
    h,
    c
)


tokens = []


# ============================================================
# DEBUG COUNTERS
# ============================================================

blank_count = 0
nonblank_count = 0


# ============================================================
# GREEDY LOOP
# ============================================================

for t in range(num_frames):

    enc_vector = encoded[
        0,
        :,
        t
    ]


    symbols_this_frame = 0


    while True:

        logits = joint_run(
            enc_vector,
            decoder_output
        )


        logits = np.asarray(
            logits
        ).reshape(-1)


        token_id = int(
            np.argmax(logits)
        )


        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        if t < 20:

            token_text = (
                vocab[token_id]
                if token_id < len(vocab)
                else "<BLANK>"
            )

            print(
                "frame:",
                t,
                "token:",
                token_id,
                "token_text:",
                repr(token_text),
                "max:",
                float(np.max(logits))
            )


        # ----------------------------------------------------
        # BLANK
        # ----------------------------------------------------

        if token_id == BLANK_ID:

            blank_count += 1

            break


        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        if token_id >= len(vocab):

            print(
                "INVALID TOKEN:",
                token_id
            )

            break


        # ----------------------------------------------------
        # NORMAL TOKEN
        # ----------------------------------------------------

        tokens.append(
            token_id
        )

        nonblank_count += 1


        # ----------------------------------------------------
        # UPDATE DECODER
        # ----------------------------------------------------

        decoder_output, h, c = decoder_run(
            token_id,
            h,
            c
        )


        symbols_this_frame += 1


        # ----------------------------------------------------
        # PROTECTION
        # ----------------------------------------------------

        if symbols_this_frame >= MAX_SYMBOLS_PER_STEP:

            print(
                "WARNING: MAX_SYMBOLS_PER_STEP "
                "reached at frame",
                t
            )

            break


# ============================================================
# TOKENS
# ============================================================

print()
print("=" * 70)
print("TOKENS")
print("=" * 70)

print(
    "Total tokens:",
    len(tokens)
)

print(
    "Blank count:",
    blank_count
)

print(
    "Non-blank count:",
    nonblank_count
)

print()
print(tokens)


# ============================================================
# TOKEN -> TEXT
# ============================================================

result = ""

for token_id in tokens:

    if 0 <= token_id < len(vocab):

        result += vocab[token_id]


# ============================================================
# CLEAN TEXT
# ============================================================

result = result.strip()


# Убираем повторные пробелы

result = " ".join(
    result.split()
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print()
print("RIGHT CHANNEL:")
print()

print(result)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
