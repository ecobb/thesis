from scipy.io.wavfile import write
import numpy as np

samplerate = 44100


def get_piano_notes():
    '''
    Returns a dict object for all the piano 
    note's frequencies
    '''
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 261.63  # Frequency of Note C4 in 440 Hz equal temperament

    note_freqs = {octave[i]: base_freq * pow(2, (i / 12)) for i in range(len(octave))}
    note_freqs[''] = 0.0

    return note_freqs


def get_wave(freq, duration=0.5):
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t)

    return wave


def get_song_data(music_notes):
    note_freqs = get_piano_notes()
    song = [get_wave(note_freqs[note]) for note in music_notes.split('-')]
    song = np.concatenate(song)
    return song.astype(np.int16)


def get_chord_data(chords):
    chords = chords.split('-')
    note_freqs = get_piano_notes()

    chord_data = []
    for chord in chords:
        data = sum([get_wave(note_freqs[note]) for note in list(chord)])
        chord_data.append(data)

    chord_data = np.concatenate(chord_data, axis=0)
    return chord_data.astype(np.int16)


def make_audio(freqs, duration=.5, filename=''):
    song = [get_wave(freqs[i], duration) for i in range(len(freqs))]
    data = np.concatenate(song).astype(np.int16)
    data = data * (16300 / np.max(data))
    write(filename, samplerate, data.astype(np.int16))


def make_edo_scale(duration=.5, filename='', chromatic=False):
    if chromatic:
        freqs = [220*2**(i/12) for i in range(13)]
        make_audio(freqs, duration, filename)

    else:
        ratios = np.array([1, 2**(2/12), 2**(4/12), 2**(5/12), 2**(7/12), 2**(9/12), 2**(11/12), 2])
        freqs = 220*ratios
        make_audio(freqs, duration, filename)


def make_just_scale(ref_freq, duration=.5, filename=''):
    just_ratios = np.array([1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2])
    freqs = ref_freq*just_ratios
    make_audio(freqs, duration, filename)


def make_random_just_audio(duration=.5, filename=''):
    just_ratios = np.array([1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2])
    initial_freq = np.random.uniform(220, 440)
    i = 1
    freqs = [initial_freq]
    print(freqs)
    while len(freqs) <= 8:
        freq = freqs[-1]*np.random.choice(just_ratios[1:])
        print(freq)
        print(freqs)
        while freq > 600:
            freq /= 2

        freqs.append(freq)

    print(freqs)
    make_audio(freqs, duration, filename)


def make_random_pythag_audio(duration=.5, filename=''):
    pythag_ratios = np.array([1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128, 2])
    initial_freq = np.random.uniform(220, 440)
    i = 1
    freqs = [initial_freq]
    print(freqs)
    while len(freqs) <= 8:
        freq = freqs[-1]*np.random.choice(pythag_ratios[1:])
        print(freq)
        print(freqs)
        while freq > 600:
            freq /= 2

        freqs.append(freq)

    print(freqs)
    make_audio(freqs, duration, filename)


def make_pythag_scale(ref_freq, duration=.5, filename=''):
    pythag_ratios = np.array([1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128, 2])
    freqs = ref_freq*pythag_ratios
    make_audio(freqs, duration, filename)

# def main():
#     # Notes
#     music_notes = 'C-D-E-F-G-A-B'
#     data = get_song_data(music_notes)
#     data = data * (16300 / np.max(data))
#     write('/Users/ethancobb/Documents/Thesis Data/Audio/test/c_major_test.wav', samplerate, data.astype(np.int16))


if __name__ == '__main__':
    # main()
    # just_file_path = '/Users/ethancobb/Documents/Thesis Data/Audio/test/test_just.wav'
    # make_just_scale(220.0, duration=.5, filename=just_file_path)
    # edo_file_path = '/Users/ethancobb/Documents/Thesis Data/Audio/test/test_edo.wav'
    # make_edo_scale(duration=.5, filename=edo_file_path, chromatic=False)
    # pythag_file_path = '/Users/ethancobb/Documents/Thesis Data/Audio/test/test_pythag.wav'
    # make_pythag_scale(220.0, duration=.5, filename=pythag_file_path)
    random_just_file_path = '/Users/ethancobb/Documents/Thesis Data/Audio/test/test_random_just.wav'
    make_random_just_audio(duration=.5, filename=random_just_file_path)
