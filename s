import os
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

TARGET_SAMPLE_RATE = 16000

N_MELS = 64
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160

BLANK_ID = 0
MAX_SYMBOLS_PER_STEP = 20


# ============================================================
# VOCAB
# ============================================================

def load_vocab(path):
    vocab = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vocab.append(line.rstrip("\r\n"))

    return vocab


# ============================================================
# WAV
# ============================================================

def load_right_channel(path):
    """
    Читаем стерео WAV.
    Используем только правый канал.
    При необходимости ресэмплируем в 16 kHz.
    """

    audio, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    print("\nWAV:")
    print("  sample rate:", sample_rate)
    print("  shape:", audio.shape)

    # --------------------------------------------------------
    # Проверяем стерео
    # --------------------------------------------------------

    if audio.ndim != 2:
        raise ValueError(
            "Ожидается стерео WAV. "
            "Получена форма: {}".format(
                audio.shape
            )
        )

    if audio.shape[1] != 2:
        raise ValueError(
            "Ожидается 2 канала. "
            "Получено каналов: {}".format(
                audio.shape[1]
            )
        )

    # --------------------------------------------------------
    # Только правый канал
    # --------------------------------------------------------

    right_channel = audio[:, 1]

    print("  используем: RIGHT channel")

    # --------------------------------------------------------
    # Ресэмплинг
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        print(
            "  resample: {} Hz -> {} Hz".format(
                sample_rate,
                TARGET_SAMPLE_RATE
            )
        )

        right_channel = librosa.resample(
            right_channel,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE
        )

    else:
        print(
            "  resample: не требуется"
        )

    right_channel = np.asarray(
        right_channel,
        dtype=np.float32
    )

    print(
        "  samples:",
        len(right_channel)
    )

    print(
        "  duration: {:.2f} sec".format(
            len(right_channel) / TARGET_SAMPLE_RATE
        )
    )

    return right_channel


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=True,
        window="hann",
        fmin=0,
        fmax=TARGET_SAMPLE_RATE // 2,
    )

    mel = np.log(
        np.maximum(
            mel,
            1e-10
        )
    )

    mel = mel.astype(
        np.float32
    )

    # [64, T] -> [1, 64, T]

    mel = np.expand_dims(
        mel,
        axis=0
    )

    return mel


# ============================================================
# ONNX MODELS
# ============================================================

def load_models():

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    encoder = ort.InferenceSession(
        ENCODER_FILE,
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    decoder = ort.InferenceSession(
        DECODER_FILE,
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    joint = ort.InferenceSession(
        JOINT_FILE,
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    return encoder, decoder, joint


# ============================================================
# ENCODER
# ============================================================

def run_encoder(
    encoder,
    features
):

    length = np.array(
        [
            features.shape[2]
        ],
        dtype=np.int64
    )

    outputs = encoder.run(
        None,
        {
            "audio_signal": features,
            "length": length,
        }
    )

    encoded = outputs[0]
    encoded_len = outputs[1]

    print("\nEncoder:")
    print(
        "  encoded shape:",
        encoded.shape
    )

    print(
        "  encoded_len:",
        encoded_len
    )

    return encoded, encoded_len


# ============================================================
# DECODER
# ============================================================

def decoder_step(
    decoder,
    token,
    h,
    c
):

    x = np.array(
        [[token]],
        dtype=np.int64
    )

    outputs = decoder.run(
        None,
        {
            "x": x,
            "h.1": h,
            "c.1": c,
        }
    )

    dec = outputs[0]
    new_h = outputs[1]
    new_c = outputs[2]

    return dec, new_h, new_c


# ============================================================
# JOINT
# ============================================================

def run_joint(
    joint,
    enc,
    dec
):

    enc = np.asarray(
        enc,
        dtype=np.float32
    )

    dec = np.asarray(
        dec,
        dtype=np.float32
    )

    enc = enc.reshape(
        1,
        768,
        1
    )

    dec = dec.reshape(
        1,
        320,
        1
    )

    output = joint.run(
        None,
        {
            "enc": enc,
            "dec": dec,
        }
    )[0]

    return output.reshape(-1)


# ============================================================
# RNNT GREEDY DECODER
# ============================================================

def greedy_decode(
    encoder_output,
    encoder_length,
    decoder,
    joint
):

    encoded = encoder_output[0]

    T = int(
        encoder_length
    )

    # --------------------------------------------------------
    # Initial decoder state
    # --------------------------------------------------------

    h = np.zeros(
        (1, 1, 320),
        dtype=np.float32
    )

    c = np.zeros(
        (1, 1, 320),
        dtype=np.float32
    )

    # Первый decoder input
    dec, h, c = decoder_step(
        decoder,
        BLANK_ID,
        h,
        c
    )

    tokens = []

    t = 0

    # --------------------------------------------------------
    # RNNT greedy decoding
    # --------------------------------------------------------

    while t < T:

        enc = encoded[:, t]

        emitted = 0

        while True:

            logits = run_joint(
                joint,
                enc,
                dec
            )

            token = int(
                np.argmax(logits)
            )

            # ------------------------------------------------
            # Blank
            # ------------------------------------------------

            if token == BLANK_ID:

                t += 1
                break

            # ------------------------------------------------
            # Token
            # ------------------------------------------------

            emitted += 1

            if emitted > MAX_SYMBOLS_PER_STEP:

                print(
                    "WARNING: слишком много токенов "
                    "на одном timestep"
                )

                t += 1
                break

            tokens.append(
                token
            )

            # ------------------------------------------------
            # Decoder
            # ------------------------------------------------

            dec, h, c = decoder_step(
                decoder,
                token,
                h,
                c
            )

    return tokens


# ============================================================
# TOKENS -> TEXT
# ============================================================

def tokens_to_text(
    tokens,
    vocab
):

    result = []

    for token_id in tokens:

        if token_id == BLANK_ID:
            continue

        if token_id < 0:
            continue

        if token_id >= len(vocab):

            print(
                "WARNING: token {} отсутствует "
                "в vocab".format(
                    token_id
                )
            )

            continue

        result.append(
            vocab[token_id]
        )

    return "".join(result)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GigaAM v3 RNNT")
    print("Python 3.8 / CPU / INT8")
    print("=" * 70)

    # --------------------------------------------------------
    # Проверяем файлы
    # --------------------------------------------------------

    required_files = [
        ENCODER_FILE,
        DECODER_FILE,
        JOINT_FILE,
        VOCAB_FILE,
        WAV_FILE
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                "Файл не найден: {}".format(
                    path
                )
            )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab = load_vocab(
        VOCAB_FILE
    )

    print(
        "\nVocabulary:",
        len(vocab),
        "tokens"
    )

    # --------------------------------------------------------
    # WAV
    # --------------------------------------------------------

    audio = load_right_channel(
        WAV_FILE
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    print(
        "\nExtracting features..."
    )

    features = extract_features(
        audio
    )

    print(
        "  features shape:",
        features.shape
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print(
        "\nLoading ONNX models..."
    )

    encoder, decoder, joint = load_models()

    print("  encoder loaded")
    print("  decoder loaded")
    print("  joint loaded")

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoded, encoded_len = run_encoder(
        encoder,
        features
    )

    # --------------------------------------------------------
    # RNNT
    # --------------------------------------------------------

    print(
        "\nRNNT decoding..."
    )

    tokens = greedy_decode(
        encoded,
        encoded_len[0],
        decoder,
        joint
    )

    print(
        "  token count:",
        len(tokens)
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = tokens_to_text(
        tokens,
        vocab
    )

    print("\n")
    print("=" * 70)
    print("RIGHT CHANNEL:")
    print("=" * 70)
    print(text)
    print("=" * 70)


if __name__ == "__main__":
    main()
