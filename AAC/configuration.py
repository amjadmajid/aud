from enum import Enum as enum
from OChirpEncode import OChirpEncode


class Configuration(enum):
    baseline = 0,
    baseline_fast = 1,
    balanced = 2,
    fast = 3


def get_configuration_encoder(config: Configuration) -> OChirpEncode:
    """
        Get the encoder associated with the relevant Configuration.
        The first two configurations (baseline and baseline_fast) are set in stone and should not change.
        For the other two configurations we separate the localization and communication, so we are free to design
        everything to optimize the bitrate.
    """
    if config == Configuration.baseline:
        return OChirpEncode(T=0.048, T_preamble=0, blank_space_time=0)
    elif config == Configuration.baseline_fast:
        return OChirpEncode(T=0.024, T_preamble=0)
    elif config == Configuration.balanced:
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=22,
                            minimize_sub_chirp_duration=True)
    elif config == Configuration.fast:
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=12,
                            minimize_sub_chirp_duration=True, fs=12500, fe=20000, blank_space_time=0.002)


if __name__ == '__main__':
    for config in Configuration:
        encoder = get_configuration_encoder(config)
        print(f"config {config} has a bit rate {1/encoder.T:.0f} bps")
