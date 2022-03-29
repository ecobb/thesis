import csv
import os
from operator import itemgetter

import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
import pandas as pd
from preprocess import preprocess


def generate_output_csv_save_path(file_path, save_dir):
    # file_name = os.path.split(file_path)[1]
    # save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + ".csv")
    file_name = os.path.split(file_path)[1]
    save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + ".csv")
    return save_path


def generate_plot_save_path(file_path, save_dir):
    file_name = os.path.split(file_path)[1]
    save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + ".png")
    return save_path


def generate_audio_save_path(file_path, save_dir, idx, label):
    file_name = os.path.split(file_path)[1]
    save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + '_' + str(idx) + '_' + label + ".wav")

    return save_path


def generate_statistics_save_path(file_path, save_dir):
    file_name = os.path.split(file_path)[1]
    save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + "_statistics.csv")
    return save_path


def generate_output_statistics(output):
    """
    Calculates the fraction of each tuning system in the output
    Returns: a list of tuples (tuning_system, total, perc) where total is the total
    number of samples under that intonation system and perc is total/N where N is the
    total number of detected notes in the audio.
    """
    # iterate through array and count total number of instances of each class
    # treat each just intonation key differently on this iteration; later, we'll
    # pool together all the keys into one just estimate
    num_counts = {}
    for sample in output:
        tuning_system = sample[3]
        count = sample[4]
        if tuning_system in num_counts:
            num_counts[tuning_system] += count
        else:
            num_counts[tuning_system] = count

    counts = list(num_counts.items())
    total = sum(tup[1] for tup in counts)

    edo_total = num_counts['edo'] if 'edo' in num_counts else 0
    pythag_total = num_counts['pythag'] if 'pythag' in num_counts else 0
    just_perc = round(100 * ((total - edo_total - pythag_total) / total), 2)
    # add the percentage to each tuple for the final result
    output_statistics = [tup + (tup[1] / total,) for tup in counts]

    return output_statistics, just_perc


def output_to_csv(output, file_path, save_path):
    # save rounded output to csv
    output_rounded = [(sample[0], sample[1], round(sample[2], 5), sample[3], sample[4]) for sample in output]
    csv_save_path = generate_output_csv_save_path(file_path, save_path)
    csvfile = open(csv_save_path, 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(['index', 't_start', 't_final', 'tuning system', 'total'])
    obj.writerows(output_rounded)
    csvfile.close()

    # save intonation statistics to csv
    output_statistics, just_perc = generate_output_statistics(output)
    output_statistics_rounded = [(sample[0], sample[1], round(100 * sample[2], 2)) for sample in output_statistics]
    statistics_save_path = generate_statistics_save_path(file_path, save_path)
    # print(f"{output_statistics_rounded}")
    # print(f"statistics_save_path: {statistics_save_path}")
    csvfile = open(statistics_save_path, 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(['tuning system', 'total', 'percentage'])
    obj.writerows(output_statistics_rounded)

    # obj.writerow(['', '', ''])
    next_row = ['Percent Just: ', '', str(just_perc)]
    obj.writerow(next_row)

    # obj.writerow(['', '', ''])
    most_probable_system = max(output_statistics_rounded, key=itemgetter(1))[0]
    # print(most_probable_system)
    last_row = ['Most Probable Intonation System: ', str(most_probable_system)]
    obj.writerow(last_row)
    # obj.writerow(['Most Probable Intonation System: ', ])

    csvfile.close()


def output_to_wav(output, file_path, save_dir):
    """
    Goes through output and converts each sample to an individual wav file
    by indexing the original audio.
    """
    for i in range(len(output)):
        t1 = output[i][1] * 1000  # Works in milliseconds
        t2 = output[i][2] * 1000
        new_audio = AudioSegment.from_wav(file_path)
        # print(f'save_dir = {save_dir}')
        new_dir = os.path.join(save_dir, 'audio_output')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        save_path = generate_audio_save_path(file_path, new_dir, i, output[i][3])
        # print(f'save_path = {save_path}')
        new_audio = new_audio[t1:t2]
        new_audio.export(save_path, format="wav")  # Exports to a wav file in the current path.


def plot_output(output, file_path, save_dir):
    # time = [sample[2] for sample in output]
    # time_initial = [sample[1] for sample in output]
    # # print(f"time_initial: {time_initial}")
    # time_final = [sample[2] for sample in output]
    # # print(f"time_final: {time_final}")
    # labels = [sample[3] for sample in output]
    # print(labels)
    test_dict = {}
    for sample in output:
        tuning_system = sample[3]
        if tuning_system not in test_dict:
            test_dict[tuning_system] = [[sample[1]], [sample[2]]]
        else:
            test_dict[tuning_system][0].append(sample[1])
            test_dict[tuning_system][1].append(sample[2])
    # print(test_dict)
    plt.figure(figsize=(9, 3))
    colors = ['r', 'g', 'b', 'tab:orange', 'c', 'k', 'm', 'tab:brown']
    for idx, key in enumerate(test_dict):
        labels = np.full(len(test_dict[key][0]), key)
        time_initial = test_dict[key][0]
        time_final = test_dict[key][1]
        plt.hlines(labels, time_initial, time_final, colors=colors[idx], linestyles=['dashed'])

    title = (str(os.path.splitext(os.path.split(file_path)[1])[0].split('_')[0])).capitalize()
    plt.title(title)
    plt.xlabel('time (s)')

    save_path = generate_plot_save_path(file_path, save_dir)
    plt.savefig(save_path, bbox_inches="tight")


def output_to_latex(save_dir):
    os.chdir(save_dir)
    cmd = "python3 -m csv2latex"
    os.system(cmd)


def save_output(output, file_path, save_dir):
    """
    1. Create directory for each cellist to save csv and audio
    2. Save the output and intonation statistics to csv files
    3. Convert the output to individual audio clips
    4. Save a plot of the output
    """
    # create directory for cellist
    cellist_name = os.path.splitext(os.path.split(file_path)[1])[0].split('_')[0]
    save_path = os.path.join(save_dir, cellist_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_to_csv(output, file_path, save_path)
    output_to_csv(output, file_path, save_dir)

    output_to_wav(output, file_path, save_path)
    plot_output(output, file_path, save_path)


def get_params(file_path, csv_file_path):
    df = pd.read_csv(csv_file_path)
    name = os.path.splitext(os.path.split(file_path)[1])[0].split('_')[0]
    # freq_A4 = df['cellist'].loc[df['cellist'] == name]
    df.set_index("cellist", inplace=True)
    freq_A4 = df.loc[name][0]
    conf_thresh = df.loc[name][1]

    return freq_A4, conf_thresh


def process(audio_files_dir, save_dir, csv_file_path, key, with_key=False, step_size=5, viterbi=True,
            save_freq_and_conf=False, save_time_freq=False, time_freq_output_file='', freq_output_file='',
            time_thresh=.55, num_notes_filter=3, min_len_seq=2, latex=False):
    """
    Runs batch preprocessing of audio files
    """
    i = 0
    num_files = len([name for name in os.listdir('.') if os.path.isfile(name)])
    for root, _, files in os.walk(audio_files_dir):
        for file in files:
            # ignore hidden files
            if not file[0] == '.':
                file_path = os.path.join(root, file)
                # if file_path == '/Users/ethancobb/Documents/Thesis_Data/Audio/suite3/prelude/full/audio/bylsma_full.wav':
                print(f"Processing file {file_path}")
                freq_A4, conf_thresh = get_params(file_path, csv_file_path)
                # freq_A4, conf_thresh = 440.0, .9
                # print(freq_A4, conf_thresh)
                output = preprocess(file_path, key, with_key, step_size, viterbi, save_freq_and_conf,
                                    save_time_freq, time_freq_output_file, freq_output_file,
                                    conf_thresh, freq_A4, time_thresh,
                                    num_notes_filter, min_len_seq)
                # print(output)
                save_output(output, file_path, save_dir)
                i += 1
                print(f"Processed {i} files so far")
    if latex:
        output_to_latex(save_dir)
    print(f"Processed {i} total files")


if __name__ == "__main__":
    AUDIO_DIR = "/Users/ethancobb/Documents/Thesis_Data/Audio/suite3/prelude/full/audio/"
    SAVE_DIR = "/Users/ethancobb/Documents/Thesis_Data/Audio/suite3/prelude/full/output/all_stencils"
    # AUDIO_DIR = "/Users/ethancobb/Documents/Thesis_Data/Audio/scales/audio"
    # SAVE_DIR = "/Users/ethancobb/Documents/Thesis_Data/Audio/scales/output"
    CSV_FILE_PATH = "/Users/ethancobb/Documents/Thesis_Data/Audio/freq_A4_data.csv"

    process(AUDIO_DIR, SAVE_DIR, CSV_FILE_PATH, key='C', with_key=False, step_size=5, viterbi=True,
            save_freq_and_conf=False, save_time_freq=False, time_freq_output_file='', freq_output_file='',
            time_thresh=.55, num_notes_filter=3, min_len_seq=2, latex=True)

    # test_output = [(0, 0.7, 1.7899899999999997, 'edo', 5), (1, 2.15, 3.9299899999999997, 'c just', 3)]
    # test_output = [(0, 0.7, 1.78, 'edo', 5), (1, 2.15, 3.929, 'c just', 3), (2, 4, 5, 'd just', 2),
    #                (3, 5, 6, 'pythag', 3), (4, 6, 7, 'c just', 5), (5, 7, 8, 'edo', 10), (6, 8, 9, 'pythag', 2)]

    # file_path = '/Users/ethancobb/Documents/Thesis_Data/Audio/suite3/prelude/full/audio/bylsma_scale.wav'
    # save_dir = "/Users/ethancobb/Documents/Thesis_Data/Audio/suite3/prelude/full/output/one_stencil/c/bylsma"
    # plot_output(test_output, file_path, save_dir)

    # # test = load_output(SAVE_DIR)
    # marks = [(1.195, 1.58499, 'edo', 3), (1.645, 4.154990000000001, 'just', 5)]
    # output_to_csv(test_output, file_path, save_dir)

    # print(test)
