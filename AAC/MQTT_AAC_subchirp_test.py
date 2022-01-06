from tqdm.auto import tqdm
import numpy as np
import contextlib
import sys
from tqdm.contrib import DummyTqdmFile
from generic_audio_functions import play_file, get_sound_file_length
from OChirpEncode import OChirpEncode
import paho.mqtt.client as mqtt


configurations = ["fixed", "dynamic"]
offsets = [0, 2, 4, 6]
symbol_times = np.arange(1, 25, 1) / 1000
fstart = 9500
fend = 13500
repeats = 40


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def encode_message(config: str, filename: str, duration: float) -> str:
    return f"--music {config} " \
           f"--duration {np.ceil(duration)} " \
           f"--location {filename} "


def generate_settings(configurations: list, symbol_times: list, offsets: list, fstart: int, fend: int, repeats: int) -> list:
    settings = []
    for _ in range(repeats):
        for conf in configurations:
            for offset in offsets:
                for symbol_time in symbol_times:
                    settings.append({
                        "configuration": conf,
                        "symbol_time": symbol_time,
                        "offset": offset,
                        "fstart": fstart,
                        "fend": fend
                    })
    return settings


def get_cycles(setting: dict) -> float:
    if setting["configuration"] == "fixed":
        return (setting["symbol_time"] * ((2 * setting["fstart"]) + ((setting["fend"] - setting["fstart"]) / 8))) / 16
    else:
        sum = 0
        for i in range(1, 9):
            sum += 2 / ((2 * setting["fstart"]) + (((2 * i) - 1) * ((setting["fend"] - setting["fstart"]) / 8)))
        return setting["symbol_time"] / sum


def rec_done_callback(client, userdata, message):
    global rec_done
    rec_done = True
    print("rec_done")


settings = generate_settings(configurations, symbol_times, fstart, fend, repeats)

rec_done = False

client = mqtt.Client()

client.connect("192.168.1.196")

client.subscribe("rec_done")
client.subscribe("play_done")

client.message_callback_add("rec_done", rec_done_callback)

# loop
client.loop_start()

with std_out_err_redirect_tqdm() as orig_stdout:
    for setting in tqdm(settings, file=orig_stdout, dynamic_ncols=True):
        cycles = get_cycles(setting)
        encoder = OChirpEncode(T=None, fs=setting["fstart"], fe=setting["fend"], T_preamble=0,
                               minimize_sub_chirp_duration=setting["configuration"] == "dynamic",
                               required_number_of_cycles=cycles)

        file, data = encoder.convert_data_to_sound("UUUU")

        msg = encode_message(setting["configuration"], str(setting["offset"] + "_" + str(setting["symbol_time"]) + "_" + str(cycles), get_sound_file_length(file) + 0.75)
        client.publish("playrec", msg)

        play_file(file, padding_duration_ms=200, add_padding_to_end=True)

        while not rec_done:
            pass

        rec_done = False

        print("")

print("DONE!")
