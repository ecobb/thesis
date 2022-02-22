# util.py

import wave
import sys
import os
import webbrowser

# check that python version is correct:
REQ_VER = (3,9)
if sys.version_info[:2] != REQ_VER:
    cur = f'{sys.version_info[0]}.{sys.version_info[1]}'
    req = f'{REQ_VER[0]}.{REQ_VER[1]}'
    raise Exception(f"You must run python version {req}, but you are running {cur}.")

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import matplotlib.patches as patches

import IPython.display as ipd
from ipywidgets import IntSlider, FloatSlider

######### Wave Files ##########

def load_wav(filepath, t_start = 0, t_end = sys.maxsize) :
    """Load a wave file, which must be 16bit and must be either mono or stereo. 
    :param filepath: audio file
    :param t_start: start time when loading a portion of the file (in seconds)
    :param t_end: end time when loading a portion of the file (in seconds)
    :return: a numpy floating-point array with a range of [-1, 1]
    """
    
    wf = wave.open(filepath)
    num_channels, sampwidth, fs, end, comptype, compname = wf.getparams()
    
    # for now, we will only accept 16 bit files
    assert(sampwidth == 2)

    # start frame, end frame, and duration in frames
    f_start = int(t_start * fs)
    f_end = min(int(t_end * fs), end)
    frames = f_end - f_start

    wf.setpos(f_start)
    raw_bytes = wf.readframes(frames)

    # convert raw data to numpy array, assuming int16 arrangement
    samples = np.fromstring(raw_bytes, dtype = np.int16)

    # convert from integer type to floating point, and scale to [-1, 1]
    samples = samples.astype(float)
    samples *= (1 / 32768.0)

    if num_channels == 1:
        return samples

    elif num_channels == 2:
        return 0.5 * (samples[0::2] + samples[1::2])

    else:
        raise('Can only handle mono or stereo wave files')

def save_wav(channels, fs, filepath) :
    """Interleave channels and write out wave file as 16bit audio.
    Inputs:
        channels: a tuple or list of np.arrays. Or can be a single np.array in which case this will be a mono file.
                  format of np.array is floating [-1, 1]
        fs: sampling rate
        filepath: output filepath
    """

    if type(channels) == tuple or type(channels) == list:
        num_channels = len(channels)
    else:
        num_channels = 1
        channels = [channels]

    length = min ([len(c) for c in channels])
    data = np.empty(length*num_channels, float)

    # interleave channels:
    for n in range(num_channels):
        data[n::num_channels] = channels[n][:length]

    data *= 32768.0
    data = data.astype(np.int16)
    data = data.tostring()

    wf = wave.open(filepath, 'w')
    wf.setnchannels(num_channels)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(data)


######### ipywidgets helper function for sliders ##########

def slider(min, max, value=None):
    """Create a FloatSlider or IntSlider, depending on type of input args.
    Turn off continuous_update
    """
    if type(min) is float or type(max) is float:
        return FloatSlider(min=float(min), max=float(max), continuous_update=False, value=value)
    else:
        return IntSlider(min=int(min), max=int(max), continuous_update=False, value=value)


######### Display Hacks ##########

def open_documentation():
    """Cause the web browser to open the documentation as a file:/// URL"""
    url = 'file://' + os.path.abspath('../doc/index.html')
    webbrowser.open(url)

def documentation_button():
    """Create an HTML button that will open up documentation by calling ``open_documentation()``
    """
    js_cmd = f"Jupyter.notebook.kernel.execute('fmp.open_documentation()')"
    js_button = f'<button onclick="{js_cmd}">Open fmplib Documentation</button>'
    ipd.display(ipd.HTML(js_button))

######### Annotations ##########

# create a mapping from a pitch name (eg 'f#') to its numerical value (eg 6)
# encodes all enharmonics - so all variants of shaprs (#) or flats (b)
g_pitch_names = { 'n':-1 }
def make_pitch_names() :
    global g_pitch_names
    letters = 'cdefgab'
    values  = (0,2,4,5,7,9,11) 
    for i in range(7):
        g_pitch_names[ letters[i] ] = values[i]
        g_pitch_names[ letters[i] + 'b' ] = (values[i] + 11) % 12
        g_pitch_names[ letters[i] + '#' ] = (values[i] + 1) % 12
make_pitch_names()

def pitch_name_to_num(txt):
    return g_pitch_names[ txt.lower() ]

def chord_name_to_number(sym) :
    """convert a chord name like A:min or C# or B:maj7 into an index.
    output is [0:23] representing the major and minor triads, and -1 for N
    """
    sym = sym.split('/')
    chord = sym[0].split(':')
    num = pitch_name_to_num(chord[0])
    # if it is a minor chord, add 12:
    if len(chord) > 1 and 'min' in chord[1]:
        num += 12
    return num

def load_chord_annotations(filepath, ff) :
    """load chord annotation from filepath.
    filepath: the name of the file
    ff: the desired feature rate for the output array.

    Returns: an array, sampled at ff, where each sample is a numeric
    chord label for that time step. 
    The labels 0-23 map to the major and minor chords.
    -1 is mapped to "no chord" (like silence or noise)
    """

    # read entire file and create a chords list, each item being:
    # (time-step, chord-num).
    lines = open(filepath).readlines()
    chords = []
    for l in lines:
        start, end, sym = l.strip().split()
        end = float(end)
        time_step = int(round(end * ff))
        chord_num = chord_name_to_number(sym)
        chords.append( (time_step, chord_num) )

    # final value of chords gives us the proper length:
    num_steps = chords[-1][0]
    out = np.empty(num_steps, dtype = int)
    s_beg = 0
    for s_end, n in chords:
        out[s_beg:s_end] = n
        s_beg = s_end

    return out

def load_boundary_annotations(filepath):
    """Read filepath, assumed to be a lab-style text file containing boundary annotations.
    Return the locations of the boundaries, as an np.array, in seconds.
    Do not include the beginning or ending of the song or silence.
    """
    lines = [l.strip().split('\t') for l in open(filepath).readlines()]
    bounds = np.array([float(l[0]) for l in lines if l[-1] != 'silence'])
    
    # don't include first boundary (which is either 0 or start of song)
    return bounds[1:]

def write_boundary_annotations(filepath, points):
    f = open(filepath, 'w')
    points = np.concatenate(([0], points))
    for p in range(len(points) - 1):
        f.write('%f\t%f\t%d\n' % (points[p], points[p+1], p))
        

######### Plotting ##########

def plot_and_listen(filepath, len_t = 0) :
    """Plot the audio waveform and create an audio listening widget.
    Inputs:
        filepath: audio file
        len_t: (optional) only load the first len_t seconds of audio.
    Returns:
        IPython.display.Audio object for listening
    """
    if len_t != 0:
        x = load_wav(filepath, 0, len_t)
    else:
        x = load_wav(filepath)
    fs = 22050
    t = np.arange(len(x)) / float(fs)
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (secs)")
    plt.show()
    return ipd.Audio(x, rate=fs)

def plot_fft_and_listen(filepath, raw_axis = False) :
    """Plot the audio waveform and create an audio listening widget.
    Inputs:
        filepath: audio file
        raw_axis: (optional)
    Returns:
        IPython.display.Audio object for listening
    """
    fs = 22050
    x = load_wav(filepath)
    x_ft = np.abs(np.fft.fft(x))

    time = np.arange(len(x),dtype=float) / fs
    freq = np.arange(len(x_ft), dtype=float) / len(x_ft) * fs

    if raw_axis:
        print('sample rate:', fs)
        print('N: ', len(x))

    plt.figure()
    plt.subplot(2,1,1)
    if raw_axis:
        plt.plot(x)
        plt.xlabel('n')
        plt.ylabel('$x(n)$')
    else:
        plt.plot(time, x)
        plt.xlabel('time')

    plt.subplot(2,1,2)
    if raw_axis:
        plt.plot(x_ft)
        plt.xlabel('k')
        plt.ylabel('$|X(k)|$')
        plt.xlim(0, 3000*len(x) / fs)
    else:
        plt.plot(freq, x_ft)
        plt.xlim(0, 3000)
        plt.xlabel('Frequency (Hz)')

    return ipd.Audio(x, rate=fs)


def plot_spectrogram(spec, cmap=None, colorbar=True, fs = None) :
    '''plot a spectrogram using a log scale for amplitudes (ie, color brightness)
    Inputs:
        spec: the spectrogram, |STFT|^2
        cmap: (optional), provide a cmap
        colorbar: (optional, default True), plot the colobar
        fs: the original sampling frequency - will show frequency labels for y-axis
    '''

    extent = None 
    if fs:
        extent = (0, spec.shape[1], 0, fs / 2)

    maxval = np.max(spec)
    minval = .1
    plt.imshow(spec, origin='lower', interpolation='nearest', aspect='auto', 
        norm=LogNorm(vmin=minval, vmax=maxval), cmap=cmap, extent=extent)
    if colorbar:
        plt.colorbar()


def plot_two_chromas(c1, c2, cmap = 'Greys'):
    '''plot two chromagrams with subplots(2,1,1) and (2,1,2). Ensure that vmin and vmax are the same
    for both chromagrams'''

    plt.subplot(2,1,1)
    _min = 0.5 * ( np.min(c1) + np.min(c2) )
    _max = 0.5 * ( np.max(c1) + np.max(c2) )
    plt.imshow(c1, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap, vmin=_min, vmax=_max)
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(c2, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap, vmin=_min, vmax=_max)
    plt.colorbar()


def plot_matrix_and_points(matrix, values):
    """Plot a matrix and a set of values that highlight a particular
    row for each column of the matrix
    Inputs:
        matrix: an NxM matrix (like a chromagram)
        values: 1xM array of values to highlight in the matrix"""

    plt.imshow(matrix, origin='lower', interpolation='nearest', aspect='auto', cmap='Greys');
    plt.plot(values, 'ro');
    plt.xlim(0, matrix.shape[1])
    plt.ylim(-.5, matrix.shape[0]+.5)


# helpful for labeling chords names along an axis
def chord_template_labels(axis=0) :
    class ChordFormatter(ticker.Formatter):
        def __call__(self, x, pos=None):
            pc = x % 12
            name = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'][pc]
            quality = ['','m'][int(x / 12)]
            return name + quality
    ax = plt.gca().yaxis if axis==0 else plt.gca().xaxis
    ax.set_major_locator( ticker.FixedLocator(np.arange(24)) )
    ax.set_major_formatter( ChordFormatter() )


def plot_ssm_paths(ssm_len, paths):
    "show paths as red lines on a plot"
    plt.xlim(-0.5,ssm_len-0.5)
    plt.ylim(-0.5,ssm_len-0.5)
    for p in paths:
        x,y = list(zip(*p))
        plt.plot(x, y, 'r', linewidth=4)

def plot_ssm_blocks(ssm_len, blocks):
    "show blocks as alpha'd blue rectangles"
    ax = plt.gca().axes
    for b in blocks:
        ax.add_patch(patches.Rectangle( (b[0], b[1]), b[2]-b[0], b[3]-b[1], alpha=.5))

def plot_ssm_corners(ssm_len, corners, d = 35):
    plt.xlim(-0.5,ssm_len-0.5)
    plt.ylim(-0.5,ssm_len-0.5)
    for p in corners:
        x,y = p
        plt.plot((x-d, x+d), (y, y), 'y', linewidth=4)
        plt.plot((x, x), (y-d, y+d), 'y', linewidth=4)

