#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import itertools as it

# Useful constants
MIDI_A4 = 69   # MIDI Pitch number


def midi2freq(midi_number, freq_A4=440):
    """
    Given a MIDI pitch number, returns its frequency in Hz.
    """
    return freq_A4 * 2 ** ((midi_number - MIDI_A4) * (1./12.))


# In[17]:


def str2midi(note_string):
    """
    Given a note string name (e.g. "Bb4"), returns its MIDI pitch number.
    """
    if note_string == "?":
        return np.nan
    data = note_string.strip().lower()
    name2delta = {"c": -9, "d": -7, "e": -5, "f": -4, "g": -2, "a": 0, "b": 2}
    accident2delta = {"b": -1, "#": 1, "x": 2}
    accidents = list(it.takewhile(lambda el: el in accident2delta, data[1:]))
    octave_delta = int(data[len(accidents) + 1:]) - 4
    return (MIDI_A4 +
          name2delta[data[0]] + # Name
          sum(accident2delta[ac] for ac in accidents) + # Accident
          12 * octave_delta # Octave
         )


# In[18]:


def str2freq(note_string, freq_A4=440):
    """
    Given a note string name (e.g. "F#2"), returns its frequency in Hz.
    """
    return midi2freq(str2midi(note_string), freq_A4)


# In[19]:


def freq2midi(freq, freq_A4=440):
    """
    Given a frequency in Hz, returns its MIDI pitch number.
    """
    result = 12 * (np.log2(freq) - np.log2(freq_A4)) + MIDI_A4
    return np.nan if isinstance(result, complex) else result


# In[20]:


def midi2str(midi_number, sharp=True):
    """
    Given a MIDI pitch number, returns its note string name (e.g. "C3").
    """
    MIDI_A4 = 69
    if np.any(np.isinf(midi_number)) or np.any(np.isnan(midi_number)):
        return "?"
    num = midi_number - (MIDI_A4 - 4 * 12 - 9)
    note = (num + .5) % 12 - .5
    rnote = int(round(note))

    error = note - rnote
    octave = str(int(round((num - note) / 12.)))

    if sharp:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    else:
        names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    names = names[rnote] + octave
    if abs(error) < 1e-4:
        return names
    else:
        err_sig = "+" if error > 0 else "-"
        err_str = err_sig + str(round(100 * abs(error), 2)) + "%"
        return names #+ err_str


# In[21]:


def freq2str(freq, freq_A4=440):
    """
    Given a frequency in Hz, returns its note string name (e.g. "D7").
    """
    return midi2str(freq2midi(freq, freq_A4))


# In[ ]:




