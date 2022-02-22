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
from tuning_systems import create_scale_template

import sys
sys.path.append("..")
import fmplib as fmp


def load_file(filepath, step):
    sr, audio = wavfile.read(filepath)
    time, frequency, confidence, activation = crepe.predict(audio, sr, 
                                                            viterbi=True, step_size=step)
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
            note_dict[pitch] = [frequency[i]]
        else:
            note_dict[pitch].append(frequency[i])
    
    return note_dict


def calc_fundamental_freq(frequency, actual_notes, freq_A4=440):
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
    '''
    Given a list of notes in SPN, convert to the corresponding ET
    frequency values. 
    '''
    et_freq = []

    for pitch in notes:
        et_freq.append(midi_tools.str2freq(pitch, A4))

    et_freq = np.array(et_freq)
    
    return et_freq


def calculate_and_plot_deviation_et(f0_estimates, actual_notes_sorted):
    et_freq = notes_to_et(actual_notes_sorted, 440)
    delta = f0_estimates - et_freq
    delta_dict = dict(zip(actual_notes_sorted, delta))

    names = list(delta_dict.keys())
    values = list(delta_dict.values())

    plt.bar(range(len(delta_dict)), values, tick_label=names)
    plt.show()


def identify_interval_spn(note1, note2):
    '''
    Identifies musical interval between two notes given in SPN using 
    music21. Reduces interval to less than or equal to an octave. 
    Ex: 'C2', 'C3' -> 'P8'; 'G3', 'C2' -> 'P5'
    '''
    n1 = note.Note(note1)
    n2 = note.Note(note2)

    i = interval.Interval(n1, n2)
    
    return i.semiSimpleName


def identify_interval_freq1(freq1, freq2, freq_A4=440):
    '''
    Identify interval between two frequencies. 
    This method first converts the frequencies to notes in SPN
    to make the determination. 
    '''
    note1 = midi_tools.freq2str(freq1, freq_A4)
    note2 = midi_tools.freq2str(freq2, freq_A4)
    
    interval = identify_interval_spn(note1, note2)
    
    return interval

def match_ratio_to_inverval(ratio):
    '''
    Takes in a ratio between frequencies and spits back the closest
    musical interval. Assumes the ratio is already in the range [1, 2]
    '''
    ratio_vals = np.array([2**(i/12) for i in range(13)])
    intervals = ['unison', 'minor second', 'major second', 'minor third', 
                'major third', 'perfect fourth', 'tritone', 
                'perfect fifth', 'minor sixth', 'major sixth', 
                 'minor seventh','major seventh', 'octave']
    interval_dict = dict(zip(ratio_vals, intervals))
    
    # the interval is the value of the closest key in the dictionary
    interval = interval_dict.get(ratio) or interval_dict[
      min(interval_dict.keys(), key = lambda key: abs(key-ratio))]
    
    return interval


def identify_interval_freq2(freq1, freq2, freq_A4=440):
    '''
    Identify interval between two frequencies. 
    This method first converts the frequencies to notes in SPN
    to make the determination. 
    '''
    # determine frequency ratio and scale in range [1.0, 2.0]
    if freq1 >= freq2:
        ratio = freq1/freq2
    else:
        ratio = freq2/freq1
    
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
    '''
    Reorders the note dictionary according to the ordering of the actual 
    notes for a given sample. Assumes there is a one-to-one correspondence
    between the actual notes and notes in the dictionary. 
    '''
    sorted_note_dict = OrderedDict()
    for note in actual_notes:
        sorted_note_dict[note] = note_dict[note]
        
    return sorted_note_dict


def compute_pair_wise_intervals(note_dict):
    '''
    Takes in a note dictionary where keys are notes in SPN in order and
    values are fundamental frequency values. Returns a list of each
    pair wise musical interval based on SPN. 
    '''
    notes = list(note_dict.keys())
    intervals = []
    for i in range(1, len(notes)):
        interval = identify_interval_spn(notes[i], notes[i-1])
        intervals.append(interval)
        
    return intervals


def calculate_cost_scale(ref_pitch, tuning_system, scale):

    template = create_scale_template(ref_pitch, tuning_system)
    # adjusting to get rid of non-diatonic notes just for this example
    scale_freq = np.array([scale[i][1] for i in range(len(scale))][::-1])

    scale_delta = (np.abs(template - scale_freq)).astype(float)
    scale_delta_norm = np.linalg.norm(scale_delta)

    return scale_delta_norm





