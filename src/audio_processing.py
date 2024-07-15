import os
import numpy as np
import scipy.io.wavfile as wav
import ffmpeg
import io
from pydub import AudioSegment

def read_audio(file_path):
    fs, snd = wav.read(file_path)
    return fs, snd


def convert_to_aac_in_memory(snd, fs=44100, channels=1):

    audio_bytes = snd.tobytes()
    input_params = {
        'format': 's16le',
        'acodec': 'pcm_s16le',
        'ar': fs,
        'ac': channels
    }
    output_params = {
        'format': 'adts',
        'acodec': 'aac'
    }
    out, _ = (
        ffmpeg
        .input('pipe:', **input_params)
        .output('pipe:', **output_params)
        .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
    )
    aac_io = io.BytesIO(out)
    aac_io.seek(0)
    
    return aac_io

def decode_aac_from_memory(aac_io, snd):
    audio = AudioSegment.from_file(aac_io, format="aac")
    decoded_snd = np.array(audio.get_array_of_samples(), dtype=np.int16)
    if len(decoded_snd) > len(snd):
        decoded_snd = decoded_snd[:len(snd)]
    else:
        decoded_snd = np.pad(decoded_snd, (0, len(snd) - len(decoded_snd)), 'constant')
    return decoded_snd


def simulate_aac(signal, preservation_factor=0.9):
    noise = np.random.normal(0, 0.01, len(signal))
    return signal * preservation_factor + noise

def quantize(signal, levels):
    return np.round(signal * (levels-1)) / (levels-1)
