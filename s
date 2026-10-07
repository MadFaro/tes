import os
import re
import csv

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_DIR = r"C:\Users\tolog\Desktop\gig"

# Папка с WAV-файлами
INPUT_DIR = r"C:\Users\tolog\Desktop\audio"

# Куда сохранить результат
OUTPUT_CSV = r"C:\Users\tolog\Desktop\transcriptions.csv"


# ============================================================
# MODEL FILES
# ============================================================

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
# AUDIO SETTINGS
# ============================================================

TARGET_SR = 16000

# Минимальная продолжительность тишины,
# которая считается границей реплики.
MIN_SILENCE_DURATION = 0.45

# Максимальная длина одного куска.
MAX_SEGMENT_DURATION = 25.0

# Слишком короткие куски игнорируем.
MIN_SEGMENT_DURATION = 0.30

# Порог тишины.
SILENCE_DB = -38


# ============================================================
# GIGAAM SETTINGS
# ============================================================

N_MELS = 64
N_FFT = 320
WIN_LENGTH = 320
HOP_LENGTH = 160

ENCODER_DIM = 768
DECODER_DIM = 320

BLANK_ID = 33

MAX_SYMBOLS_PER_STEP = 20


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):

    seconds = max(
        0,
        seconds
    )

    minutes = int(
        seconds // 60
    )

    secs = int(
        seconds % 60
    )

    millis = int(
        (seconds - int(seconds))
        * 1000
    )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{millis:03d}"
    )


# ============================================================
# LOAD VOCAB
# ============================================================

vocab = []

with open(
    VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.rstrip(
            "\r\n"
        )

        # Формат:
        #
        # ▁ 0
        # а 1
        # б 2
        # ...
        # я 32
        # <blk> 33

        match = re.match(
            r"^(.*?)\s+\d+$",
            line
        )

        if match:

            token = match.group(1)

        else:

            token = line

        vocab.append(token)


# ============================================================
# LOAD ONNX
# ============================================================

print("Загрузка модели...")

encoder = ort.InferenceSession(
    ENCODER_FILE,
    providers=[
        "CPUExecutionProvider"
    ]
)

decoder = ort.InferenceSession(
    DECODER_FILE,
    providers=[
        "CPUExecutionProvider"
    ]
)

joint = ort.InferenceSession(
    JOINT_FILE,
    providers=[
        "CPUExecutionProvider"
    ]
)

print("Модель загружена.")
print()


# ============================================================
# DECODER
# ============================================================

def decoder_run(
    token,
    h,
    c
):

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

    result = joint.run(
        None,
        {
            "enc": enc_input,
            "dec": dec_input
        }
    )

    return result[0].reshape(-1)


# ============================================================
# TRANSCRIBE ONE AUDIO SEGMENT
# ============================================================

def transcribe_audio(
    audio,
    sr
):

    if len(audio) == 0:

        return ""


    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    if sr != TARGET_SR:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR,
            res_type="kaiser_best"
        )

        sr = TARGET_SR


    audio = audio.astype(
        np.float32
    )


    # --------------------------------------------------------
    # MEL
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,

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


    # --------------------------------------------------------
    # LOG MEL
    # --------------------------------------------------------

    mel = np.log(
        np.maximum(
            mel,
            1e-10
        )
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    features = mel[
        np.newaxis,
        :,
        :
    ]

    feature_length = np.array(
        [features.shape[2]],
        dtype=np.int64
    )


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
        np.asarray(
            encoded_len
        ).reshape(-1)[0]
    )


    # --------------------------------------------------------
    # INITIAL DECODER STATE
    # --------------------------------------------------------

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


    decoder_output, _, _ = decoder_run(
        BLANK_ID,
        h,
        c
    )


    # --------------------------------------------------------
    # RNNT
    # --------------------------------------------------------

    tokens = []


    for t in range(num_frames):

        enc_vector = encoded[
            0,
            :,
            t
        ]

        symbols = 0


        while (
            symbols
            < MAX_SYMBOLS_PER_STEP
        ):

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


            # INVALID
            if (
                token_id < 0
                or token_id >= len(vocab)
            ):

                break


            tokens.append(
                token_id
            )


            decoder_output, h, c = decoder_run(
                token_id,
                h,
                c
            )


            symbols += 1


    # --------------------------------------------------------
    # TOKENS -> TEXT
    # --------------------------------------------------------

    text = "".join(
        vocab[token_id]
        for token_id in tokens
    )


    # SentencePiece-style word marker
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


    return text


# ============================================================
# SPLIT CHANNEL BY SILENCE
# ============================================================

def split_channel(
    audio,
    sr
):

    frame_length = int(
        0.030 * sr
    )

    hop_length = int(
        0.010 * sr
    )


    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False
    )[0]


    # --------------------------------------------------------
    # DB
    # --------------------------------------------------------

    db = librosa.amplitude_to_db(
        rms,
        ref=np.max
    )


    speech = (
        db > SILENCE_DB
    )


    # --------------------------------------------------------
    # FIND SPEECH
    # --------------------------------------------------------

    segments = []

    start_frame = None

    silence_frames = 0

    max_silence_frames = int(
        MIN_SILENCE_DURATION
        / 0.010
    )


    for i, is_speech in enumerate(
        speech
    ):

        if is_speech:

            if start_frame is None:

                start_frame = i

            silence_frames = 0

        else:

            if start_frame is not None:

                silence_frames += 1


                if (
                    silence_frames
                    >= max_silence_frames
                ):

                    end_frame = (
                        i
                        - silence_frames
                        + 1
                    )


                    start_time = (
                        start_frame
                        * hop_length
                        / sr
                    )

                    end_time = (
                        end_frame
                        * hop_length
                        / sr
                    )


                    if (
                        end_time
                        - start_time
                        >= MIN_SEGMENT_DURATION
                    ):

                        segments.append(
                            (
                                start_time,
                                end_time
                            )
                        )


                    start_frame = None

                    silence_frames = 0


    # --------------------------------------------------------
    # LAST SEGMENT
    # --------------------------------------------------------

    if start_frame is not None:

        start_time = (
            start_frame
            * hop_length
            / sr
        )

        end_time = (
            len(audio)
            / sr
        )


        if (
            end_time
            - start_time
            >= MIN_SEGMENT_DURATION
        ):

            segments.append(
                (
                    start_time,
                    end_time
                )
            )


    # ========================================================
    # MAX 25 SECONDS
    # ========================================================

    final_segments = []


    for start, end in segments:

        duration = (
            end - start
        )


        if duration <= MAX_SEGMENT_DURATION:

            final_segments.append(
                (
                    start,
                    end
                )
            )

            continue


        current = start


        while (
            end - current
            > MAX_SEGMENT_DURATION
        ):

            final_segments.append(
                (
                    current,
                    current
                    + MAX_SEGMENT_DURATION
                )
            )

            current += (
                MAX_SEGMENT_DURATION
            )


        if (
            end - current
            >= MIN_SEGMENT_DURATION
        ):

            final_segments.append(
                (
                    current,
                    end
                )
            )


    return final_segments


# ============================================================
# PROCESS CHANNEL
# ============================================================

def process_channel(
    audio,
    sr,
    speaker
):

    segments = split_channel(
        audio,
        sr
    )


    result = []


    for start, end in segments:

        start_sample = int(
            start * sr
        )

        end_sample = int(
            end * sr
        )


        segment_audio = audio[
            start_sample:end_sample
        ]


        text = transcribe_audio(
            segment_audio,
            sr
        )


        if not text:

            continue


        result.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text
            }
        )


    return result


# ============================================================
# BUILD DIALOGUE
# ============================================================

def process_file(
    wav_file
):

    # --------------------------------------------------------
    # READ WAV
    # --------------------------------------------------------

    audio, sr = sf.read(
        wav_file,
        dtype="float32",
        always_2d=True
    )


    if audio.shape[1] < 2:

        raise RuntimeError(
            "WAV не является стерео"
        )


    # --------------------------------------------------------
    # CHANNELS
    # --------------------------------------------------------

    operator_channel = audio[
        :,
        0
    ]

    client_channel = audio[
        :,
        1
    ]


    # --------------------------------------------------------
    # OPERATOR
    # --------------------------------------------------------

    operator_segments = process_channel(
        operator_channel,
        sr,
        "Оператор"
    )


    # --------------------------------------------------------
    # CLIENT
    # --------------------------------------------------------

    client_segments = process_channel(
        client_channel,
        sr,
        "Клиент"
    )


    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    dialogue = (
        operator_segments
        + client_segments
    )


    dialogue.sort(
        key=lambda x: x["start"]
    )


    # --------------------------------------------------------
    # MERGE SAME SPEAKER
    # --------------------------------------------------------

    merged = []


    for item in dialogue:

        if not merged:

            merged.append(
                item.copy()
            )

            continue


        previous = merged[-1]


        if (
            previous["speaker"]
            == item["speaker"]
        ):

            previous["end"] = max(
                previous["end"],
                item["end"]
            )


            previous["text"] = (
                previous["text"].rstrip()
                + " "
                + item["text"].lstrip()
            )

        else:

            merged.append(
                item.copy()
            )


    # --------------------------------------------------------
    # FORMAT DIALOGUE
    # --------------------------------------------------------

    dialogue_lines = []


    for item in merged:

        start = format_time(
            item["start"]
        )

        end = format_time(
            item["end"]
        )

        dialogue_lines.append(
            f"[{start} - {end}] "
            f"{item['speaker']}: "
            f"{item['text']}"
        )


    return "\n".join(
        dialogue_lines
    )


# ============================================================
# FIND WAV FILES
# ============================================================

wav_files = [
    os.path.join(
        INPUT_DIR,
        filename
    )
    for filename in os.listdir(
        INPUT_DIR
    )
    if filename.lower().endswith(
        ".wav"
    )
]


wav_files.sort()


if not wav_files:

    raise RuntimeError(
        "В указанной папке нет WAV-файлов."
    )


print(
    f"Найдено файлов: {len(wav_files)}"
)

print()


# ============================================================
# PROCESS ALL FILES
# ============================================================

rows = []


for index, wav_file in enumerate(
    wav_files,
    start=1
):

    filename = os.path.basename(
        wav_file
    )


    print(
        f"[{index}/{len(wav_files)}] "
        f"{filename}"
    )


    try:

        dialogue = process_file(
            wav_file
        )


        rows.append(
            {
                "filename": filename,
                "dialogue": dialogue
            }
        )


        print(
            "    готово"
        )


    except Exception as e:

        print(
            f"    ОШИБКА: {e}"
        )


        rows.append(
            {
                "filename": filename,
                "dialogue": ""
            }
        )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "dialogue"
        ],
        delimiter=";"
    )


    writer.writeheader()


    writer.writerows(
        rows
    )


# ============================================================
# DONE
# ============================================================

print()
print(
    f"Готово. Результат: {OUTPUT_CSV}"
)
