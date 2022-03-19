import fmplib as fmp
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython.display as ipd
import crepe
import tensorflow
from collections import OrderedDict

import sys
import csv

sys.path.append('/Users/ethancobb/Documents/Thesis/Code/Modules')
import midi_tools
import tuning_systems
import freq_detection


def run_frequency_detection(audio_file, mid_file, step_size=10, viterbi=True,
                            freq_A4=440, save_freq=False, output_file=''):
    """
    Calculate the fundamental frequency of every note in the audio with crepe
    This implementation converts the frequency readings to midi pitch numbers
    and throws away any pitches that aren't in the actual pitches in the piece
    """
    actual_notes = freq_detection.load_actual_notes(mid_file)
    time, frequency, conf, act = freq_detection.run_crepe(audio_file, step_size, viterbi)
    if save_freq:
        freq_detection.save_freq_to_csv(output_file, frequency)

    freq_note_dict = freq_detection.calc_fundamental_freq_with_actual_notes(frequency,
                                                                            actual_notes, freq_A4)
    f0_dict = freq_detection.calc_f0_with_simple_mean(freq_note_dict)
    sorted_note_dict = freq_detection.sort_note_dict(f0_dict, actual_notes)
    sorted_note_dict_items = sorted_note_dict.items()

    return sorted_note_dict_items


def run_frequency_detection2(audio_file, step_size=5, viterbi=True, save_freq_and_conf=False,
                             save_time_freq=False, time_freq_output_file='', freq_output_file='',
                             conf_thresh=.75):
    """
    This implementation of note detection employs crepe but identifies individual notes
    and their timestamps by percent differences between consecutive frequency readings
    """
    time, frequency, conf, act = freq_detection.run_crepe(audio_file, step_size, viterbi)
    if save_freq_and_conf:
        freq_detection.save_freq_and_conf_to_csv(freq_output_file, frequency, conf)

    frequency, conf_perc = freq_detection.clean_frequency(frequency, conf, conf_thresh)
    time_freq_dict = freq_detection.calc_fundamental_freq_with_diff(time, frequency, step_size)

    if save_time_freq:
        freq_detection.save_time_freq_dict_to_csv(time_freq_output_file, time_freq_dict)


def run_frequency_detection3(audio_file, step_size=5, viterbi=True, save_freq_and_conf=False,
                             save_time_freq=False, time_freq_output_file='', freq_output_file='',
                             conf_thresh=.8):
    """
    This implementation of note detection employs crepe but identifies individual notes
    and their timestamps by using a confidence thresholding technique.
    """
    time, frequency, conf, act = freq_detection.run_crepe(audio_file, step_size, viterbi)
    if save_freq_and_conf:
        freq_detection.save_freq_and_conf_to_csv(freq_output_file, frequency, conf)

    frequency_clean, conf_perc = freq_detection.clean_frequency(frequency, conf, conf_thresh)
    print(f"{conf_perc} percent of data retained")
    time_freq_dict = freq_detection.calc_fundamental_freq_with_diff3(time, frequency_clean, step_size)

    if save_time_freq:
        freq_detection.save_time_freq_dict_to_csv(time_freq_output_file, time_freq_dict)

    return time_freq_dict


if __name__ == "__main__":
    bylsma_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/test/bylsma_scale.wav"
    carr_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/carr_scale.wav"
    casals_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/casals_scale.wav"
    cobb_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/audio/cobb_scale.wav"
    kirshbaum_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/kirshbaum_scale.wav"
    ma_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/ma_scale.wav"
    maisky_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/maisky_scale.wav"
    queyras_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/queyras_scale.wav"
    rostropovich_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/rostropovich_scale.wav"
    schiff_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/schiff_scale.wav"
    vogler_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/vogler_scale.wav"
    wispelwey_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/wispelwey_scale.wav"

    c_major_scale = "/Users/ethancobb/Documents/Thesis Data/Audio/test/test_just.wav"

    # wav_file_full = "/Users/ethancobb/Documents/Thesis Data/Audio/wav_full/maisky.wav"
    # mid_file = "/Users/ethancobb/Documents/Thesis Data/Audio/midi/suite_3/cs3-1pre.mid"
    # freq_data = 'maisky_scale_frequency.csv'

    # note_freq_list = run_frequency_detection(wav_file_scale, mid_file, step_size=5, viterbi=True)
    # run_frequency_detection3(wav_file_scale, step_size=5, save_freq_and_conf=False,
    #                          time_freq_output_file='time_freq_test_scale.csv', freq_output_file=
    #                          'maisky_scale_freq.csv')
    res = run_frequency_detection3(cobb_scale, step_size=5, save_freq_and_conf=False,
                                                               time_freq_output_file='', freq_output_file=
                                                               '', conf_thresh=.8)
    freq = list(res.values())
    print(freq)
    print([midi_tools.freq2str(f) for f in freq])

