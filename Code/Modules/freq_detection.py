#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.io import wavfile
import midi_tools
import matplotlib.pyplot as plt
import IPython.display as ipd
import crepe
from collections import OrderedDict
from music21 import *
import pandas as pd
import csv
import sys


sys.path.append("..")
import fmplib as fmp


def load_freq_from_csv(filename):
    frequency = np.genfromtxt(filename, delimiter=',')
    return frequency


def save_freq_and_conf_to_csv(filename, frequency, confidence):
    np.savetxt(filename, np.c_[frequency, confidence], delimiter=",", fmt='%1.5f')


def save_time_freq_dict_to_csv(filename, time_freq_dict):
    keys, values = [], []
    for key, value in time_freq_dict.items():
        keys.append(key)
        values.append(value)
    with open(filename, "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        csvwriter.writerow(values)


# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)


def load_actual_notes(mid_file):
    # TODO: load midi file and return notes in a numpy array
    """
        Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
            0-127 - note on at specified pitch
            128   - note off
            129   - no event
            from: https://colab.research.google.com/github/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb#scrollTo=x1OfN14AwDP_
        """
    stream = converter.parse(mid_file)

    # Part one, extract from stream
    total_length = int(np.round(stream.flat.highestTime / 0.25))  # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append(
                [np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25),
                                element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=int)
    df = pd.DataFrame({'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    df = df.sort_values(['pos', 'pitch'], ascending=[True, False])  # sort the dataframe properly
    df = df.drop_duplicates(subset=['pos'])  # drop duplicate values
    # part 2, convert into a sequence of note events
    output = np.zeros(total_length + 1, dtype=np.int16) + np.int16(
        MELODY_NO_EVENT)  # set array full of no events by default.
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos == i].empty:
            n = df[df.pos == i].iloc[0]  # pick the highest pitch at each semiquaver
            output[i] = n.pitch  # set note on
            output[i + n.dur] = MELODY_NOTE_OFF

    # remove the global variables from the output array
    new_arr = np.delete(output, np.where((output == MELODY_NO_EVENT) | (output == MELODY_NOTE_OFF)))

    # convert midi pitch values to SPN note names
    actual_notes = []
    for i in range(len(new_arr)):
        actual_notes.append(midi_tools.midi2str(new_arr[i]))

    return actual_notes


def run_crepe(filepath, step_size, viterbi=True):
    sr, audio = wavfile.read(filepath)
    time, frequency, confidence, activation = crepe.predict(audio, sr,
                                                            viterbi=viterbi, step_size=step_size)
    return time, frequency, confidence, activation


def make_note_dictionary(pitches):
    note_dict = {}
    for pitch in pitches:
        if pitch not in note_dict:
            note_dict[pitch] = 1
        else:
            note_dict[pitch] += 1

    return note_dict


def clean_note_dictionary(note_dict, actual_notes):
    # delete any notes that aren't in the piece from the dictionary
    for note in list(note_dict):
        if note not in actual_notes:
            note_dict.pop(note, None)

    return note_dict


def make_note_dict_with_freq(pitches, frequency):
    note_dict = {}
    for i in range(len(pitches)):
        pitch = pitches[i]
        if pitch not in note_dict:
            note_dict[pitch] = calc_fundamental_freq_with_actual_notes
        else:
            note_dict[pitch].append(frequency[i])

    return note_dict


def calc_fundamental_freq_with_actual_notes(frequency, actual_notes, freq_A4=440):
    '''
    Given crepe fundamental frequency estimates for audio data,
    calculate f0 for each note in the audio. 
    
    Returns: a dictionary where each key is the SPN of each note
    and the values are lists of the corresponding f0 estimates
    '''
    midi_pitches = midi_tools.freq2midi(frequency, freq_A4)

    pitches = []
    for i in range(len(midi_pitches)):
        pitches.append(midi_tools.midi2str(midi_pitches[i]))

    freq_note_dict = make_note_dict_with_freq(pitches, frequency)
    freq_note_dict_clean = clean_note_dictionary(freq_note_dict, actual_notes)

    return freq_note_dict_clean


def calc_fundamental_freq_with_diff(time, frequency, step_size=10):
    """
    This implementation of note detection employs crepe but identifies individual notes
    and their timestamps by percent differences between consecutive frequency readings
    """
    time_freq_dict = OrderedDict()
    freq_threshold = 2 ** (1 / 16)
    epsilon = 1e-5
    # new array storing the moving average estimates
    current_frequencies = [frequency[0]]
    for i in range(1, len(frequency)):
        current = frequency[i]
        previous = frequency[i - 1]

        if max(current, previous) / min(current, previous) > freq_threshold:
            print('greater than threshold')
            current_frequencies_arr = np.array(current_frequencies)
            time_freq_dict[time[i] - epsilon] = np.nanmean(current_frequencies_arr)
            current_frequencies = [current]
        else:
            current_frequencies.append(current)
    # After we've gone through, add the final key to the dictionary
    time_freq_dict[time[-1] + step_size * .001 - epsilon] = sum(current_frequencies) / len(current_frequencies)

    return time_freq_dict


def calc_fundamental_freq_with_diff2(time, frequency, step_size=10):
    """
    This implementation of note detection employs crepe but identifies individual notes
    and their timestamps by percent differences between consecutive frequency readings
    and handles the NaN's returned by the clean_frequency function.
    """
    time_freq_dict = OrderedDict()
    freq_threshold = 2 ** (1 / 16)
    epsilon = 1e-5
    # array storing the moving average estimates
    current_frequencies = []
    # Iterate through the frequencies until you hit the first consecutive pair of valid
    # frequency estimates. This defines the start of the algorithm.
    start = 0
    i = 1
    while np.isnan(frequency[i]) or np.isnan(frequency[i - 1]):
        i += 1

    start = i
    current = start
    previous = i - 1
    t_initial = time[previous]
    current_frequencies.append(frequency[previous])

    for i in range(start, len(frequency)):
        # todo: want to store the t_final estimate as the first nan after the last previous
        # valid frequency estimate
        if np.isnan(frequency[current]):
            t_final = time[current] - epsilon
            print('im going to continue')
            continue
        else:
            # if the ratio between two valid frequency estimates is greater than
            # the frequency threshold, this defines the onset of a new note
            if max(frequency[current], frequency[previous]) / \
                    min(frequency[current], frequency[previous]) > freq_threshold:
                print('greater than threshold')
                current_frequencies_arr = np.array(current_frequencies)
                time_freq_dict[time[i] - epsilon] = (np.mean(current_frequencies_arr))
                current_frequencies = [current]
            else:
                print('I wasnt greater than the threshold')
                current_frequencies.append(current)
    # After we've gone through, add the final key to the dictionary
    time_freq_dict[time[-1] + step_size * .001 - epsilon] = sum(current_frequencies) / len(current_frequencies)

    return time_freq_dict


def calc_fundamental_freq_with_diff3(time, frequency, step_size=10):
    """
    This implementation of note detection employs crepe but identifies individual notes
    and their timestamps by percent differences between consecutive frequency readings
    and handles the NaN's returned by the clean_frequency function.
    """
    time_freq_dict = OrderedDict()
    freq_threshold = 2 ** (1 / 26)
    epsilon = 1e-5
    # extract the index locations of the non-valid frequency estimates
    nan_arr = np.isnan(frequency)
    nan_loc = np.where(nan_arr)[0]
    nan_time = time[nan_loc]
    valid_freq_loc = np.where(~nan_arr)[0]
    valid_freq_time = time[valid_freq_loc]
    valid_freq = frequency[~nan_arr]
    # array storing the moving average estimates
    current_freq = [valid_freq[0]]
    # calculate the initial start time of the first note
    t_initial = valid_freq_time[0]
    # iterate through the valid frequencies and identify new notes when the time difference
    # between consecutive frequency readings exceed the step_size in seconds
    for i in range(1, len(valid_freq)):
        if valid_freq_time[i] - valid_freq_time[i - 1] > 4 * .001 * step_size + epsilon:
            # further check to see if it is actually a continuation of the same note
            freq_ratio = max(valid_freq[i], valid_freq[i - 1]) / min(valid_freq[i], valid_freq[i - 1])
            if freq_ratio < freq_threshold:
                continue
            # store the final time in the dictionary and begin the next new note
            # onset time
            t_final = valid_freq_time[i - 1] + .001 * step_size - epsilon
            time_freq_dict[(t_initial, t_final)] = np.mean(current_freq)
            t_initial = valid_freq_time[i]
            current_freq = [valid_freq[i]]
        else:
            current_freq.append(valid_freq[i])

    # After we've gone through, add the final key to the dictionary
    t_final = valid_freq_time[-1] + .001 * step_size - epsilon
    time_freq_dict[(t_initial, t_final)] = np.mean(current_freq)

    return time_freq_dict


def clean_frequency(frequency, confidence, conf_thresh=.8):
    """
    This helper function takes in the time and frequency estimates from crepe
    and replaces the frequency values below a certain confidence threshold with
    NaN. Also returns the fraction of frequency estimates above confidence threshold.
    """
    frequency_copy = frequency.copy()
    frequency_copy[confidence < conf_thresh] = np.nan
    #
    conf_perc = (np.count_nonzero(~np.isnan(frequency_copy)) / len(frequency_copy)) * 100

    return frequency_copy, conf_perc


def calc_f0_with_simple_mean(freq_note_dict):
    f0_dict = {}
    for note in freq_note_dict:
        f0_estimates_arr = np.array(freq_note_dict[note])
        f0_dict[note] = np.mean(f0_estimates_arr)

    return f0_dict


def compare_audio(file, freq_note_dict, fs=44100):
    # We convert the f0 estimates to midi pitches
    f0_estimates = np.array(list(freq_note_dict.values()))
    midi_pitches = midi_tools.freq2midi(f0_estimates, 440)
    midi_pitches = sorted(midi_pitches)[::-1]

    notes = []
    for i in range(len(midi_pitches)):
        notes.append((midi_pitches[i], i / 2, .5))

    # Original Audio:
    snd = fmp.load_wav(file)
    ipd.display(ipd.Audio(snd, rate=fs))
    # Synthesized midi:
    notes_snd = fmp.synthesize_sequence(notes, fs)
    ipd.display(ipd.Audio(notes_snd, rate=fs))


def notes_to_et(notes, A4=440):
    """
    Given a list of notes in SPN, convert to the corresponding ET
    frequency values.
    """
    et_freq = []

    for pitch in notes:
        et_freq.append(midi_tools.str2freq(pitch, A4))

    et_freq = np.array(et_freq)

    return et_freq


def freq_to_notes(frequency, A4=440):
    """
    Given an array of frequencies, convert to the approximate equal tempered pitch names
    E.g., [440.1, 220.5] -> ['A4', 'A3']
    """
    notes = []
    for freq in frequency:
        notes.append(midi_tools.freq2str(freq, A4))

    notes_arr = np.array(notes)

    return notes_arr


def calculate_and_plot_deviation_et(f0_estimates, actual_notes_sorted):
    et_freq = notes_to_et(actual_notes_sorted, 440)
    delta = f0_estimates - et_freq
    delta_dict = dict(zip(actual_notes_sorted, delta))

    names = list(delta_dict.keys())
    values = list(delta_dict.values())

    plt.bar(range(len(delta_dict)), values, tick_label=names)
    plt.show()


def identify_interval_spn(note1, note2):
    """
    Identifies musical interval between two notes given in SPN using
    music21. Reduces interval to less than or equal to an octave.
    Ex: 'C2', 'C3' -> 'P8'; 'G3', 'C2' -> 'P5'
    """
    n1 = note.Note(note1)
    n2 = note.Note(note2)

    i = interval.Interval(n1, n2)

    return i.semiSimpleName


def identify_interval_freq1(freq1, freq2, freq_A4=440):
    """
    Identify interval between two frequencies.
    This method first converts the frequencies to notes in SPN
    to make the determination.
    """
    note1 = midi_tools.freq2str(freq1, freq_A4)
    note2 = midi_tools.freq2str(freq2, freq_A4)

    interval = identify_interval_spn(note1, note2)

    return interval


def match_ratio_to_inverval(ratio):
    """
    Takes in a ratio between frequencies and spits back the closest
    musical interval. Assumes the ratio is already in the range [1, 2]
    """
    ratio_vals = np.array([2 ** (i / 12) for i in range(13)])
    intervals = ['unison', 'minor second', 'major second', 'minor third',
                 'major third', 'perfect fourth', 'tritone',
                 'perfect fifth', 'minor sixth', 'major sixth',
                 'minor seventh', 'major seventh', 'octave']
    interval_dict = dict(zip(ratio_vals, intervals))

    # the interval is the value of the closest key in the dictionary
    interval = interval_dict.get(ratio) or interval_dict[
        min(interval_dict.keys(), key=lambda key: abs(key - ratio))]

    return interval


def identify_interval_freq2(freq1, freq2, freq_A4=440):
    """
    Identify interval between two frequencies.
    This method first converts the frequencies to notes in SPN
    to make the determination.
    """
    # determine frequency ratio and scale in range [1.0, 2.0]
    if freq1 >= freq2:
        ratio = freq1 / freq2
    else:
        ratio = freq2 / freq1

    octave_dif = 0

    # edge case: if ratio is equal to an octave, count the octave
    # difference as 1:
    if ratio == 2.0:
        octave_dif = 1

    else:
        while ratio > 2.0:
            ratio /= 2
            octave_dif += 1
            if ratio == 2.0:
                octave_dif += 1
                break

    # match ratio to interval
    interval = match_ratio_to_inverval(ratio)

    return interval, octave_dif


def sort_note_dict(note_dict, actual_notes):
    """
    Reorders the note dictionary according to the ordering of the actual
    notes for a given sample. Assumes there is a one-to-one correspondence
    between the actual notes and notes in the dictionary.
    """
    sorted_note_dict = OrderedDict()
    for note in actual_notes:
        sorted_note_dict[note] = note_dict[note]

    return sorted_note_dict


def compute_pair_wise_intervals(note_dict):
    """
    Takes in a note dictionary where keys are notes in SPN in order and
    values are fundamental frequency values. Returns a list of each
    pair wise musical interval based on SPN.
    """
    notes = list(note_dict.keys())
    intervals = []
    for i in range(1, len(notes)):
        interval = identify_interval_spn(notes[i], notes[i - 1])
        intervals.append(interval)

    return intervals


def identify_tuning_system(ref_pitch, scale_freq):
    tuning_systems = ['just', 'edo', 'pythagorean']
    tuning_systems_costs = []
    for tuning_system in tuning_systems:
        cost = calculate_cost_scale(ref_pitch, tuning_system, scale_freq)
        tuning_systems_costs.append(cost)

    # the tuning system is that which minimizes the cost
    min_idx = tuning_systems_costs.index(min(tuning_systems_costs))
    result = tuning_systems[min_idx]

    return result


if __name__ == "__main__":
    # mid_file = "/Users/ethancobb/Documents/Thesis Data/Audio/midi/suite_3/cs3-1pre.mid"
    # actual_notes = load_actual_notes(mid_file)
    step_size = 5
    # time, frequency, confidence, act = run_crepe("/Users/ethancobb/Documents/Thesis
    # Data/Audio/test/maisky_scale.wav", step_size)
    time, frequency, confidence, act = run_crepe("/Users/ethancobb/Documents/Thesis Data/Audio/test/test.wav", step_size)

    frequency_clean, conf_perc = clean_frequency(frequency, confidence, conf_thresh=.9)
    print(conf_perc)

    time_freq_dict = calc_fundamental_freq_with_diff3(time, frequency_clean, step_size)
    print(time_freq_dict)
    time_freq_conf = np.array([time, frequency, confidence])
    time_freq_conf_clean = np.array([time, frequency_clean, confidence])
    print(freq_to_notes(time_freq_dict.values()))
    answer = {'C': 261.63, 'D': 293.66974569918125, 'E': 329.63314428399565,
              'F': 349.2341510465061, 'G': 392.0020805232462, 'A': 440.00745824565865,
              'B': 493.8916728538229}
