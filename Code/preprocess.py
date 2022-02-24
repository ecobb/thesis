from Modules.tuning_systems import run_tuning_system_detection
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

    result = run_tuning_system_detection(time_freq_dict, freq_A4)

    return result, time_freq_dict


if __name__ == "__main__":
    audio_file = "/Users/ethancobb/Documents/Thesis Data/Audio/test/maisky_scale.wav"
    # audio_file = "/Users/ethancobb/Documents/Thesis Data/Audio/test/test.wav"

    result, time_freq_dict = preprocess(audio_file, step_size=5,
                                                     viterbi=True, save_freq_and_conf=False,
                                                     save_time_freq=False,
                                                     time_freq_output_file='',
                                                     freq_output_file='',
                                                     conf_thresh=.9, freq_A4=440)
    print(result)
    print(time_freq_dict)
