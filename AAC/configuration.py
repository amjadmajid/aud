from enum import Enum as enum
from OChirpEncode import OChirpEncode


class Configuration(enum):
    baseline = 0,
    baseline_fast = 1,
    balanced = 2,
    fast = 3,
    baseline_optimized = 4,
    baseline_fast_optimized = 5


def get_configuration_encoder(config: Configuration) -> OChirpEncode:
    """
        Get the encoder associated with the relevant Configuration.
        The first two configurations (baseline and baseline_fast) are set in stone and should not change.
        For the other two configurations we separate the localization and communication, so we are free to design
        everything to optimize the bitrate.
    """
    if config == Configuration.baseline:
        return OChirpEncode(T=None, T_preamble=0.096, orthogonal_preamble=True, required_number_of_cycles=34.5, minimize_sub_chirp_duration=False)
    elif config == Configuration.baseline_fast:
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=17.25, minimize_sub_chirp_duration=False)
    elif config == Configuration.optimized:
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=34.5, minimize_sub_chirp_duration=True)
    elif config == Configuration.optimized_fast:
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=17.25, minimize_sub_chirp_duration=True)

    elif config == Configuration.balanced:
        print(f"Warning using old config! {Configuration}")
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=22,
                            minimize_sub_chirp_duration=True)
    elif config == Configuration.fast:
        print(f"Warning using old config! {Configuration}")
        return OChirpEncode(T=None, T_preamble=0.048, orthogonal_preamble=True, required_number_of_cycles=12,
                            minimize_sub_chirp_duration=True, fs=12500, fe=20000, blank_space_time=0.002)


if __name__ == '__main__':
    for config in Configuration:
        encoder = get_configuration_encoder(config)
        print(f"config {config} has a bit rate {1/encoder.T:.0f} bps")
