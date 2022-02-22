from Modules.tuning_systems import compare
from run_frequency_detection import run_frequency_detection3


def preprocess(audio_file, step_size=5, viterbi=True, save_freq_and_conf=False,
               save_time_freq=False, time_freq_output_file='', freq_output_file='',
               conf_thresh=.75, freq_A4=440):
    """
    Runs entire preprocessing pipeline.
    1. Identify notes and timestamps (run_frequency_detection3)
    2. Compute intonation statistics for just, edo, pythagorean stencils.
    """
    time_freq_dict = run_frequency_detection3(audio_file, step_size, viterbi,
                                              save_freq_and_conf, save_time_freq,
                                              time_freq_output_file, freq_output_file,
                                              conf_thresh)
    # NOTE: 'key' information will have to be generalized. For now, just feed it what we know
    # the key is.
    just_deltas = compare(time_freq_dict, 'just', 'C', freq_A4)
    edo_deltas = compare(time_freq_dict, 'edo', 'C', freq_A4)
    pythagorean_deltas = compare(time_freq_dict, 'pythagorean', 'C', freq_A4)

    return just_deltas, edo_deltas, pythagorean_deltas, time_freq_dict


if __name__ == "__main__":
    audio_file = "audio/test/maisky_scale.wav"
    audio_file = "audio/test/test.wav"

    just_deltas, edo_deltas, pythagorean_deltas, time_freq_dict = preprocess(audio_file, step_size=5,
                                                                             viterbi=True, save_freq_and_conf=False,
                                                                             save_time_freq=False,
                                                                             time_freq_output_file='',
                                                                             freq_output_file='',
                                                                             conf_thresh=.9, freq_A4=440)

    print(f"just deltas: {just_deltas}")
    print(f"edo deltas: {edo_deltas}")
    print(f"pythagorean deltas: {pythagorean_deltas}")
    print(time_freq_dict)
