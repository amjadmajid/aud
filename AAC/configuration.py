from enum import Enum as enum
from OChirpEncode import OChirpEncode


class Configuration(enum):
    baseline = 0,
    optimized = 1,
    baseline48 = 2,


def get_configuration_encoder(config: Configuration) -> OChirpEncode:
    """
        Get the encoder associated with the relevant Configuration.
        The first two configurations (baseline and baseline_fast) are set in stone and should not change.
        For the other two configurations we separate the localization and communication, so we are free to design
        everything to optimize the bitrate.
    """
    if config == Configuration.baseline:
        return OChirpEncode(T=None, T_preamble=0, required_number_of_cycles=17.25, minimize_sub_chirp_duration=False)
    elif config == Configuration.baseline48:
        return OChirpEncode(T=None, T_preamble=0, required_number_of_cycles=17.25 * 2, minimize_sub_chirp_duration=False)
    elif config == Configuration.optimized:
        return OChirpEncode(T=None, T_preamble=0, required_number_of_cycles=17.25, minimize_sub_chirp_duration=True)


if __name__ == '__main__':
    for config in Configuration:
        encoder = get_configuration_encoder(config)
        print(f"config {config} has a bit rate {1/encoder.T:.1f} bps")
