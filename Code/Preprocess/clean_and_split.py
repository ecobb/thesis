import librosa, librosa.display
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
import wavio
from librosa.core import to_mono
import matplotlib.pyplot as plt
from scipy.io import wavfile

    
def envelope(y, rate, threshold, divisor):
    '''
    This function computes the envelope of a signal over a fixed window of samples. 
    '''
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/divisor), # 15
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

def test_threshold(src_root, fn, threshold=.01, divisor=19, sr=44100):
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(fn))
        return
    wav, rate = read_audio(wav_path[0], sr)
    mask, env = envelope(wav, rate, threshold, divisor)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()
    
def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)
    
def read_audio(path, sr=44100):
    
    audio, rate = librosa.load(path, sr=None, mono=False)
    audio = to_mono(audio)
    
    return audio, rate

def split_wavs(src_root, dst_root, dt=1, threshold=.01, divisor=19, sr=44100):

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    wav_files_full = os.listdir(src_root)
    check_dir(dst_root)
    for fn in tqdm(wav_files_full):
        src_dir = os.path.join(src_root, fn)
        wav, rate = read_audio(src_dir, sr)
        mask, y_mean = envelope(wav, rate, threshold, divisor)
        wav = wav[mask]
        delta_sample = int(dt*rate)

        # step through audio and save every delta_sample
        # discard the ending audio if it is too short
        trunc = wav.shape[0] % delta_sample
        for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
            start = int(i)
            stop = int(i + delta_sample)
            sample = wav[start:stop]
            save_sample(sample, rate, dst_root, fn, cnt)


if __name__ == '__main__':
    
    SRC_ROOT = "Audio/wav_full/"
    DST_ROOT = "Audio/wav_test/"
    
    split_wavs(SRC_ROOT, DST_ROOT)
    # test_threshold(SRC_ROOT, 'wispelwey')
    
    # count total number of samples created
    path, dirs, files = next(os.walk(DST_ROOT))
    file_count = len(files)
    print(f"\nTotal number of Samples: {file_count}")