from scipy.io.wavfile import read, write
import numpy as np
import sounddevice as sd
from OChirpOldFunctions import add_wgn
from pathlib import Path
from os import path


def play_file(file: str, padding_duration_ms: float = 0.0, db: float = None, add_padding_to_end: bool = False,
              block: bool = True):
    """
        Play a sound file, optionally add padding of a certain length, dB and whether to also add it the end.

        Adding padding to the end might help with speakers trying to smooth the end of an audio segment.
    """

    fs, data = read(file)

    if padding_duration_ms > 0:
        padding = np.zeros(int((padding_duration_ms/1000.0) * fs)).astype(np.int16)

        # If we want the padding to contain some white noise
        if db is not None:
            noise = np.ones(padding.size)
            noise = add_wgn(noise, db)
            padding = noise

        data = np.append(padding, data)
        if add_padding_to_end:
            data = np.append(data, padding)

    sd.play(data, fs, blocking=block)


def get_sound_file_length(file: str) -> int:
    fs, data = read(file)
    return data.size / fs


def record_sound(file: str, duration_s: float, samplerate: int = 44100, channels: int = 1) -> np.ndarray:
    """
        Record a sound for a duration at a samplerate. Select the number of microphones by setting channels.

        Create the path towards file if it does not exists yet.

        If file is "", we ignore it
    """
    frames = int(0.5 + (duration_s * samplerate))
    data = sd.rec(frames=frames, channels=channels, dtype=np.int16, blocking=True)

    if file != "":
        directory = path.split(file)[0]
        Path(directory).mkdir(parents=True, exist_ok=True)
        write(file, samplerate, data)

    return data



