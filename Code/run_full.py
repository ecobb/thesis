import csv
import os

import numpy as np
import pandas as pd

from preprocess import preprocess


def _generate_save_path(file_path, save_dir):
    file_name = os.path.split(file_path)[1]
    save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + ".csv")
    return save_path


def save_output(output, file_path, save_dir):
    save_path = _generate_save_path(file_path, save_dir)
    csvfile = open(save_path, 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(['t_start', 't_final', 'tuning system', 'total'])
    obj.writerows(output)
    csvfile.close()


def _get_params(file_path, csv_file_path):
    df = pd.read_csv(csv_file_path)
    name = os.path.splitext(os.path.split(file_path)[1])[0].split('_')[0]
    # freq_A4 = df['cellist'].loc[df['cellist'] == name]
    df.set_index("cellist", inplace=True)
    freq_A4 = df.loc[name][0]
    conf_thresh = df.loc[name][1]

    return freq_A4, conf_thresh


def process(audio_files_dir, save_dir, csv_file_path, step_size=5, viterbi=True,
            save_freq_and_conf=False, save_time_freq=False, time_freq_output_file='', freq_output_file='',
            time_thresh=.55, num_notes_filter=3, min_len_seq=2):
    """
    Runs batch preprocessing of audio files
    """
    for root, _, files in os.walk(audio_files_dir):
        for file in files:
            # ignore hidden files
            if not file[0] == '.':
                file_path = os.path.join(root, file)
                if file_path == '/Users/ethancobb/Documents/Thesis Data/Audio/scales/audio/pythag_scale.wav':
                    print(f"Processed file {file_path}")
                    # freq_A4, conf_thresh = _get_params(file_path, csv_file_path)
                    freq_A4, conf_thresh = 440.0, .8
                    print(freq_A4, conf_thresh)
                    output = preprocess(file_path, step_size, viterbi, save_freq_and_conf,
                                        save_time_freq, time_freq_output_file, freq_output_file,
                                        conf_thresh, freq_A4, time_thresh,
                                        num_notes_filter, min_len_seq)
                    save_output(output, file_path, save_dir)


# def load_output(output_path):
#     output_full = []
#     for root, _, files in os.walk(output_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             output = np.load(file_path)
#             print(output)
#             output_full.append(output)
#     result = np.array(output_full)
#
#     return result


# def run_full(audio_files_dir, save_dir, step_size=5, viterbi=True,
#              save_freq_and_conf=False, save_time_freq=False, time_freq_output_file='', freq_output_file='',
#              conf_thresh=.75, freq_A4=440, num_notes=3):
#     process(audio_files_dir, save_dir, step_size, viterbi,
#             save_freq_and_conf, save_time_freq, time_freq_output_file, freq_output_file,
#             conf_thresh, freq_A4, num_notes)


if __name__ == "__main__":
    FILES_DIR = "/Users/ethancobb/Documents/Thesis Data/Audio/scales/audio/"
    SAVE_DIR = "/Users/ethancobb/Documents/Thesis Data/Audio/scales/output/"
    CSV_FILE_PATH = "/Users/ethancobb/Documents/Thesis Data/Audio/freq_A4_data.csv"

    process(FILES_DIR, SAVE_DIR, CSV_FILE_PATH, step_size=5, viterbi=True,
            save_freq_and_conf=False, save_time_freq=False, time_freq_output_file='', freq_output_file='',
            time_thresh=.55, num_notes_filter=3, min_len_seq=2)
    # # test = load_output(SAVE_DIR)
    # file_path = '/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/output/carr_scale.wav.csv'
    # marks = [(1.195, 1.58499, 'edo', 3), (1.645, 4.154990000000001, 'just', 5)]
    # test_file = "/Users/ethancobb/Documents/Thesis Data/Audio/suite3/prelude/scale/audio/carr_scale.wav"

    # print(test)
