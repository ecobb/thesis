from run_frequency_detection import run_frequency_detection3
from run_tuning_system_detection import run_tuning_system_detection2


def preprocess(audio_file, step_size=5, viterbi=True, save_freq_and_conf=False,
               save_time_freq=False, time_freq_output_file='', freq_output_file='',
               conf_thresh=.75, freq_A4=440, time_thresh=.55,
               num_notes_filter=3, min_len_seq=2):
    """
    Runs entire preprocessing pipeline.
    1. Identify notes and timestamps (run_frequency_detection3)
    2. Compute intonation statistics for just, edo, pythagorean stencils.
    """
    time_freq_dict = run_frequency_detection3(audio_file, step_size, viterbi,
                                              save_freq_and_conf, save_time_freq,
                                              time_freq_output_file, freq_output_file,
                                              conf_thresh)

    # print(time_freq_dict)
    result = run_tuning_system_detection2(time_freq_dict, num_notes_filter, min_len_seq,
                                          time_thresh, freq_A4)

    return result


if __name__ == "__main__":
    # audio_file = "/Users/ethancobb/Documents/Thesis Data/Audio/test/maisky_scale.wav"
    audio_file = "/Users/ethancobb/Documents/Thesis Data/Audio/test/test_just.wav"
    test = '/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/audio/carr_scale.wav'

    output = preprocess(test, step_size=5, viterbi=True, save_freq_and_conf=False,
               save_time_freq=False, time_freq_output_file='', freq_output_file='',
               conf_thresh=.75, freq_A4=440, time_thresh=.55,
               num_notes_filter=3, min_len_seq=2)
    print(output)
