#!/usr/bin/env python
# coding: utf-8

# In[94]:
from collections import OrderedDict

from pytuning.scales import *
import sympy as sp
import numpy as np


# Equal Temperament

def create_edo_template(freq_A4=440.0):
    notes = ['Ab', 'A', 'A#', 'Bb', 'B', 'B#', 'Cb', 'C', 'C#', 'Db', 'D',
             'D#', 'Eb', 'E', 'E#', 'Fb', 'F', 'F#', 'Gb', 'G', 'G#']
    cello_c_thresh = 65.0

    a_flat_val = freq_A4 / (2 ** (1 / 12))
    a_natural_val = freq_A4
    a_sharp_val = freq_A4 * 2 ** (1 / 12)
    b_flat_val = a_sharp_val
    b_natural_val = freq_A4 * 2 ** (2 / 12)
    b_sharp_val = b_natural_val * 2 ** (1 / 12)
    c_flat_val = b_natural_val
    c_natural_val = b_sharp_val
    c_sharp_val = c_natural_val * 2 ** (1 / 12)
    d_flat_val = c_sharp_val
    d_natural_val = c_sharp_val * 2 ** (1 / 12)
    d_sharp_val = d_natural_val * 2 ** (1 / 12)
    e_flat_val = d_sharp_val
    e_natural_val = e_flat_val * 2 ** (1 / 12)
    e_sharp_val = e_natural_val * 2 ** (1 / 12)
    f_flat_val = e_natural_val
    f_natural_val = e_sharp_val
    f_sharp_val = f_natural_val * 2 ** (1 / 12)
    g_flat_val = f_sharp_val
    g_natural_val = g_flat_val * 2 ** (1 / 12)
    g_sharp_val = g_natural_val * 2 ** (1 / 12)

    initial_frequencies = np.array([a_flat_val, a_natural_val, a_sharp_val, b_flat_val, b_natural_val,
                                    b_sharp_val, c_flat_val, c_natural_val, c_sharp_val, d_flat_val, d_natural_val,
                                    d_sharp_val,
                                    e_flat_val, e_natural_val, e_sharp_val, f_flat_val, f_natural_val, f_sharp_val,
                                    g_flat_val,
                                    g_natural_val, g_sharp_val])
    # add +- 2 octaves for each note
    frequencies = []
    for freq in initial_frequencies:
        # don't include any frequencies that fall below the frequency for open C
        if freq / 2 < cello_c_thresh:
            frequencies.append((freq / (2 ** 3), freq / (2 ** 2), freq / 2, freq))
        else:
            if freq / (2 ** 3) < cello_c_thresh:
                frequencies.append((freq / (2 ** 2), freq / 2, freq, freq * 2))
            else:
                frequencies.append((freq / (2 ** 3), freq / (2 ** 2), freq / 2, freq))
            # frequencies.append((freq / (2 ** 2), freq / 2, freq, freq * 2))

    edo_template = dict(zip(notes, frequencies))

    return edo_template


def create_c_just_template(freq_A4=440.0):
    c_just_template = {}
    c_freq = freq_A4 * (2 / 3) ** 3 / 2
    notes = ['C', 'Cb', 'C#', 'D', 'Db', 'D#', 'E', 'Eb', 'E#', 'F', 'Fb', 'F#',
             'G', 'Gb', 'G#', 'A', 'Ab', 'A#', 'B', 'Bb', 'B#']

    ratios = np.array([1, 24 / 25, 25 / 24, 9 / 8, 24 / 25, 25 / 24, 5 / 4, 24 / 25, 25 / 24,
                       4 / 3, 24 / 25, 25 / 24, 3 / 2, 24 / 25, 25 / 24, 5 / 3, 24 / 25, 25 / 24,
                       15 / 8, 24 / 25, 25 / 24])

    # note_multipliers = [notes.index('C'), notes.index('C'), notes.index('C'), notes.index('C'),
    #                     notes.index('D'), notes.index('D'), notes.index('C'), notes.index('E'),
    #                     notes.index('E'), notes.index('C'), notes.index('F'), notes.index('F'),
    #                     notes.index('C'), notes.index('G'), notes.index('G'), notes.index('C'),
    #                     notes.index('A'), notes.index('A'), notes.index('C'), notes.index('B'),
    #                     notes.index('B')]
    note_multipliers = [0, 0, 0, 0, 3, 3, 0, 6, 6, 0, 9, 9, 0, 12, 12, 0, 15, 15, 0, 18, 18]

    c_just_template['C'] = c_freq
    for i in range(len(notes)):
        freq = ratios[i] * c_just_template[notes[note_multipliers[i]]]
        c_just_template[notes[i]] = freq

    initial_frequencies = c_just_template.values()
    frequencies = [(freq, freq * 2, freq * (2 ** 2), freq * (2 ** 3)) for freq in initial_frequencies]

    result = dict(zip(notes, frequencies))
    return result


def calculate_reference_freq(key, freq_A4=440.0):
    if key == 'C':
        return freq_A4 * (2 / 3) ** 3 / 2
    if key == 'Cb':
        return "haven't done yet"
    if key == 'C#':
        return "haven't done yet"
    if key == 'D':
        return freq_A4 * (2 / 3) / (2 ** 2)
    if key == 'Db':
        return "haven't done yet"
    if key == 'D#':
        return "haven't done yet"
    if key == 'E':
        return freq_A4 * (3 / 2) / (2 ** 3)
    if key == 'Eb':
        return "haven't done yet"
    if key == 'E#':
        return "haven't done yet"
    if key == 'F':
        return freq_A4 * (2 / 3) ** 4
    if key == 'Fb':
        return "haven't done yet"
    if key == 'F#':
        return freq_A4 * (3 / 2) ** 3 / (2 ** 4)
    if key == 'G':
        return freq_A4 * (2 / 3) ** 2 / 2
    if key == 'Gb':
        return "haven't done yet"
    if key == 'G#':
        return "haven't done yet"
    if key == 'A':
        return freq_A4 / (2 ** 2)
    if key == 'Ab':
        return "haven't done yet"
    if key == 'A#':
        return "haven't done yet"
    if key == 'B':
        return freq_A4 * (3 / 2) ** 2 / (2 ** 3)
    if key == 'Bb':
        return "haven't done yet"
    if key == 'B#':
        return "haven't done yet"


def create_note_ordering(key):
    # key_idx = notes.index(key)
    # result = notes[key_idx:] + notes[:key_idx]
    if key == 'C':
        return ['C', 'Cb', 'C#', 'D', 'Db', 'D#', 'E', 'Eb', 'E#', 'F', 'Fb', 'F#',
                'G', 'Gb', 'G#', 'A', 'Ab', 'A#', 'B', 'Bb', 'B#']
    if key == 'Cb':
        return "haven't done yet"
    if key == 'C#':
        return "haven't done yet"
    if key == 'D':
        return ['D', 'Db', 'D#', 'E', 'Eb', 'E#', 'F#', 'F', 'F##',
                'G', 'Gb', 'G#', 'A', 'Ab', 'A#', 'B', 'Bb', 'B#', 'C#', 'C', 'C##']
    if key == 'Db':
        return "haven't done yet"
    if key == 'D#':
        return "haven't done yet"
    if key == 'E':
        return ['E', 'Eb', 'E#', 'F#', 'F', 'F##', 'G#', 'G', 'G##',
                'A', 'Ab', 'A#', 'B', 'Bb', 'B#', 'C#', 'C', 'C##', 'D#', 'D', 'D##']
    if key == 'Eb':
        return "haven't done yet"
    if key == 'E#':
        return "haven't done yet"
    if key == 'F':
        return ['F', 'Fb', 'F#', 'G', 'Gb', 'G#', 'A', 'Ab', 'A#',
                'Bb', 'Bbb', 'B', 'C', 'Cb', 'C#', 'D', 'Db', 'D#', 'E', 'Eb', 'E#']
    if key == 'Fb':
        return "haven't done yet"
    if key == 'F#':
        return "haven't done yet"
    if key == 'G':
        return ['G', 'Gb', 'G#', 'A', 'Ab', 'A#', 'B', 'Bb', 'B#', 'C', 'Cb', 'C#',
                'D', 'Db', 'D#', 'E', 'Eb', 'E#', 'F#', 'F', 'F##']
    if key == 'Gb':
        return "haven't done yet"
    if key == 'G#':
        return "haven't done yet"
    if key == 'A':
        return ['A', 'Ab', 'A#', 'B', 'Bb', 'B#', 'C#', 'C', 'C##',
                'D', 'Db', 'D#', 'E', 'Eb', 'E#', 'F#', 'F', 'F##', 'G#', 'G', 'G##']
    if key == 'Ab':
        return "haven't done yet"
    if key == 'A#':
        return "haven't done yet"
    if key == 'B':
        return "haven't done yet"
    if key == 'Bb':
        return "haven't done yet"
    if key == 'B#':
        return "haven't done yet"


def create_just_template(key, freq_A4=440.0):
    just_template = {}
    cello_c_thresh = 65.0

    notes = create_note_ordering(key)
    ref_freq = calculate_reference_freq(key, freq_A4)

    ratios = np.array([1, 24 / 25, 25 / 24, 9 / 8, 24 / 25, 25 / 24, 5 / 4, 24 / 25, 25 / 24,
                       4 / 3, 24 / 25, 25 / 24, 3 / 2, 24 / 25, 25 / 24, 5 / 3, 24 / 25, 25 / 24,
                       15 / 8, 24 / 25, 25 / 24])

    note_multipliers = [0, 0, 0, 0, 3, 3, 0, 6, 6, 0, 9, 9, 0, 12, 12, 0, 15, 15, 0, 18, 18]

    just_template[key] = ref_freq
    for i in range(len(notes)):
        freq = ratios[i] * just_template[notes[note_multipliers[i]]]
        just_template[notes[i]] = freq

    initial_frequencies = just_template.values()
    frequencies = [(freq, freq * 2, freq * (2 ** 2), freq * (2 ** 3)) if freq / 2 < cello_c_thresh
                   else (freq / 2, freq, freq * 2, freq * (2 ** 2)) for freq in initial_frequencies]

    result = dict(zip(notes, frequencies))
    return result


# Pythagorean

# def create_pythagorean_template(ref_pitch):
#     pythagorean_scale = create_pythagorean_scale()
#     pythagorean_scale_float = [i.evalf() for i in pythagorean_scale]
#
#     return ref_pitch * np.array(pythagorean_scale_float)
def create_pythagorean_template(freq_A4=440.0):
    notes = ['Ab', 'A', 'A#', 'Bb', 'B', 'B#', 'Cb', 'C', 'C#', 'Db', 'D',
             'D#', 'Eb', 'E', 'E#', 'Fb', 'F', 'F#', 'Gb', 'G', 'G#']
    cello_c_thresh = 65.0

    a_flat_val = freq_A4 * (2 / 3) ** 7 * (2 ** 5)
    a_natural_val = freq_A4
    a_sharp_val = freq_A4 * (3 / 2) ** 7 / (2 ** 4)
    b_flat_val = freq_A4 * (2 / 3) ** 5 * (2 ** 3)
    b_natural_val = freq_A4 * (3 / 2) ** 2 / 2
    b_sharp_val = freq_A4 * (3 / 2) ** 9 / (2 ** 5)
    c_flat_val = freq_A4 * (2 / 3) ** 10 * (2 ** 6)
    c_natural_val = freq_A4 * (2 / 3) ** 3 * (2 ** 2)
    c_sharp_val = freq_A4 * (3 / 2) ** 4 / (2 ** 2)
    d_flat_val = freq_A4 * (2 / 3) ** 8 * (2 ** 5)
    d_natural_val = freq_A4 * (2 / 3) * 2
    d_sharp_val = freq_A4 * (3 / 2) ** 6 / (2 ** 3)
    e_flat_val = freq_A4 * (2 / 3) ** 6 * (2 ** 4)
    e_natural_val = freq_A4 * (3 / 2)
    e_sharp_val = freq_A4 * (3 / 2) ** 8 / (2 ** 4)
    f_flat_val = freq_A4 * (2 / 3) ** 11 * (2 ** 7)
    f_natural_val = freq_A4 * (2 / 3) ** 4 * (2 ** 3)
    f_sharp_val = freq_A4 * (3 / 2) ** 3 / 2
    g_flat_val = freq_A4 * (2 / 3) ** 9 * (2 ** 6)
    g_natural_val = freq_A4 * (2 / 3) ** 2 * (2 ** 2)
    g_sharp_val = freq_A4 * (3 / 2) ** 5 / (2 ** 2)

    initial_frequencies = np.array([a_flat_val, a_natural_val, a_sharp_val, b_flat_val, b_natural_val,
                                    b_sharp_val, c_flat_val, c_natural_val, c_sharp_val, d_flat_val, d_natural_val,
                                    d_sharp_val,
                                    e_flat_val, e_natural_val, e_sharp_val, f_flat_val, f_natural_val, f_sharp_val,
                                    g_flat_val,
                                    g_natural_val, g_sharp_val])
    # add +- 2 octaves for each note
    frequencies = []
    for freq in initial_frequencies:
        # don't include any frequencies that fall below the frequency for open C
        if freq == b_sharp_val:
            frequencies.append((freq / (2 ** 3), freq / (2 ** 2), freq / 2, freq))
        elif freq == a_sharp_val or freq == a_natural_val \
                or freq == b_natural_val or freq == b_flat_val:
            frequencies.append((freq / (2 ** 2), freq / 2, freq, freq * 2))
        else:
            if freq / (2 ** 3) < cello_c_thresh:
                frequencies.append((freq / (2 ** 2), freq / 2, freq, freq * 2))
            else:
                frequencies.append((freq / (2 ** 3), freq / (2 ** 2), freq / 2, freq))

    pythagorean_template = dict(zip(notes, frequencies))

    return pythagorean_template


def get_change(current, previous):
    """
    Helper function to compute the percent difference between two values.
    """
    if current == previous:
        return 100.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0


def calculate_delta(time_freq_dict, stencil):
    """
    This is a helper function to go through note by note in the time-frequency dictionary
    and find the closest frequency in the tuning system stencil. It will then calculate
    the estimated frequency's percent deviation from whatever tuning system's approximation
    """
    # for each note in the time_freq_dict, match its value to the closest value in the
    # stencil dictionary and store it
    delta_dict = OrderedDict()
    for key in time_freq_dict.keys():
        freq_estimate = time_freq_dict[key]
        # we search each octave band for the closest frequency match
        candidate_frequencies = [(min(stencil.items(), key=lambda x: abs(freq_estimate - x[1][i])))[1][i]
                                 for i in range(4)]
        candidate_deltas = [get_change(candidate_frequencies[i], freq_estimate) for i in range(4)]
        # the calculated delta is the lowest of these possible percent deviations
        delta = min(candidate_deltas)
        delta_dict[key] = delta

    return delta_dict


def compare(time_freq_dict, tuning_system, key, freq_A4=440):
    """
    Goes through entire time-frequency dictionary and calculates the delta
    with a particular tuning system stencil in a particular key on a
    note-by-note basis. Returns a dictionary with the same keys as time_freq_dict
    but with deltas for that particular tuning system as values.

    time_freq_dict: dictionary
    tuning_system: 'just', 'edo', 'pythagorean'
    key: 'C', 'D', 'E', 'F', 'G', 'A'
    """
    if tuning_system == 'just':
        just_template = create_just_template(key, freq_A4)
        just_delta_dict = calculate_delta(time_freq_dict, just_template)
        return just_delta_dict
    if tuning_system == 'edo':
        edo_template = create_edo_template(freq_A4)
        edo_delta_dict = calculate_delta(time_freq_dict, edo_template)
        return edo_delta_dict
    if tuning_system == 'pythagorean':
        pythagorean_template = create_pythagorean_template(freq_A4)
        pythagorean_delta_dict = calculate_delta(time_freq_dict, pythagorean_template)
        return pythagorean_delta_dict
    else:
        return 'unrecognized scale'


if __name__ == "__main__":
    c_just_template = create_c_just_template(441.0)
    # print(c_just_template)
    pythagorean_template = create_pythagorean_template(441.0)
    # print(pythagorean_template)

    c_just_set = set(c_just_template)
    pythagorean_set = set(pythagorean_template)
    # for note in just_set.intersection(pythagorean_set):
    #     print(f"{note} just: {c_just_template[note]}, pythagorean: {pythagorean_template[note]}")

    d_just_template = create_just_template('D', 441.0)
    d_just_set = set(d_just_template)
    # for note in c_just_set.intersection(d_just_set):
    #     print(f"{note} c major: {c_just_template[note]}, d major: {d_just_template[note]}")

    f_just_template = create_just_template('F', 441.0)
    f_just_set = set(f_just_template)
    # for note in f_just_set.intersection(pythagorean_set):
    #     print(f"{note} just: {f_just_template[note]}, pythagorean: {pythagorean_template[note]}")
    # print(d_just_template)
    # print(c_just_template)

    edo_template = create_edo_template(441.0)
    edo_set = set(edo_template)
    # for note in edo_set.intersection(pythagorean_set):
    #     print(f"{note} edo: {edo_template[note]}, pythagorean: {pythagorean_template[note]}")
    sample_time_freq_dict = OrderedDict()
    sample_time_freq_dict = {(0, .49): 260.5, (.5, 1.0): 120}
    test_c = compare(sample_time_freq_dict, 'just', 'C', 441)
    test_g = compare(sample_time_freq_dict, 'just', 'D', 441)
    # print(test_c)
    # print(test_g)
