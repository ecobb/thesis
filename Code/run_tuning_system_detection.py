from collections import OrderedDict
from itertools import islice

import numpy as np
import pandas as pd

from Modules import tuning_systems
from Modules.tuning_probability import calc_probability_cluster


def calculate_num_switches(x):
    """
    Helper function to calculate the number of times a label switches
    E.g., 'edo', 'edo', 'just', 'just', 'edo', 'just' -> 3
    """
    if isinstance(x, OrderedDict):
        vals = x.values()
        labels = [a_tuple[0] for a_tuple in vals]

    elif isinstance(x, list):
        labels = [a_tuple[0] for a_tuple in x]

    num_switches = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            num_switches += 1

    return num_switches


def calculate_deviation_cost(cluster):
    """
    This function computes the deviation cost of a cluster.

    cluster: OrderedDict
    """
    deviation_cost = 0
    deviation_thresh = .05

    if isinstance(cluster, OrderedDict):
        vals = cluster.values()
        delta = [a_tuple[2] for a_tuple in vals]

    if isinstance(cluster, list):
        delta = [a_tuple[2] for a_tuple in cluster]
    # the total deviation cost is the sum of the deviation costs (above the threshold)
    for val in delta:
        if val > deviation_thresh:
            deviation_cost += val

    return deviation_cost


def compute_cost(cluster):
    """
    This function computes the cost of a cluster. Switches between tuning systems
    are penalized as well as deviation cost.

    cluster: dict
    """
    num_switches = calculate_num_switches(cluster)
    deviation_cost = calculate_deviation_cost(cluster)

    return num_switches + deviation_cost


def compute_total_cluster_cost(cluster_list):
    """
    This function computes the cost of a list of clusters. Switches between tuning systems

    cluster: dict
    """
    total_cost = 0

    for cluster in cluster_list:
        num_switches = calculate_num_switches(cluster)
        deviation_cost = calculate_deviation_cost(cluster)
        cost = num_switches + deviation_cost
        total_cost += cost

    return total_cost


def span(lst):
    """
    Helper function to generate all possible sublists of a list, preserving the order
    of the elements.
    Taken from: https://stackoverflow.com/questions/9185119/algorithm-to-generate-not-quite-spanning-set-in-python/9185520#9185520
    """
    yield [lst]
    for i in range(1, len(lst)):
        for x in span(lst[i:]):
            yield [lst[:i]] + x


def run_tuning_system_clustering_binary(tuning_system_dict):
    """
    This function takes in a dictionary of note timestamps and their tuning system labelings
    and organizes them into clusters as part of the last step of the preprocessing pipeline.

    The final result is a list of OrderedDicts; each entry in the dictionary is a cluster of notes.
    """
    result = OrderedDict()
    # make initial split in the data
    inc = iter(tuning_system_dict.items())
    left = dict(islice(inc, len(tuning_system_dict) // 2))
    right = dict(inc)
    # explore the branch with the higher cost first
    left_cost = compute_cost(left)
    right_cost = compute_cost(right)
    # print(left_cost, right_cost)
    if left_cost > right_cost:
        result = right
        # split cluster in half
        inc = iter(left.items())
        left = dict(islice(inc, len(left) // 2))
        right = dict(inc)
        left_cost = compute_cost(left)
        right_cost = compute_cost(right)
    else:
        current_clusters = [left]
        # split cluster in half
        inc = iter(left.items())
        left = dict(islice(inc, len(left) // 2))
        right = dict(inc)

    return result


def run_tuning_system_clustering_brute_force(tuning_system_dict):
    """
    This function takes in a dictionary of note timestamps and their tuning system labelings
    and organizes them into clusters as part of the last step of the preprocessing pipeline.

    The final result is a list of OrderedDicts; each entry in the dictionary is a cluster of notes.
    """
    # extract labels and generate all possible clusters
    vals = tuning_system_dict.values()
    # the best possible clustering minimizes the total cost
    clusters = list(span(list(vals)))
    # print(len(clusters[1]))
    idx = 0
    min_cost = compute_total_cluster_cost(clusters[0])
    print(min_cost)

    for i in range(1, len(clusters)):
        cost = compute_total_cluster_cost(clusters[i])
        print(cost)
        if cost < min_cost:
            min_cost = cost
            idx = i
    print('idx = ', idx)
    result = clusters[idx]

    return result, cost


def run_tuning_system_change_detection(tuning_system_dict, num_notes=3, freq_A4=440.0):
    """
    This function takes in a dictionary of note timestamps and their fundamental frequencies
    and runs a sliding window of a certain number of notes through the time series to locate
    changes in the tuning system as measured by the most likely system at a point in time.

    Returns:
    """
    frequencies = np.array(list(tuning_system_dict.values()))
    # frequencies_new = np.concatenate([frequencies[0]], frequencies, [frequencies[-1]])
    pad_num = num_notes // 2
    frequencies_new = np.pad(frequencies, (pad_num,), 'constant',
                             constant_values=(frequencies[0], frequencies[-1]))
    # print(frequencies)
    # print(frequencies_new)

    predictions = []
    for i in range(len(frequencies)):
        label = calc_probability_cluster(frequencies_new[i:i + num_notes], freq_A4)
        # print(frequencies_new[i:i+num_notes])
        predictions.append(label)

    return predictions


def run_tuning_system_detection1(time_freq_dict, freq_A4=440):
    """
    This function goes through the note dictionary and computes the deviation note by note from
    the closest match frequency estimate for all tuning system stencils.

    Returns a dictionary with the same keys as time_freq_dict but where the values are tuning system
    labelings (e.g., 'just', edo', 'pythag'), the fundamental frequency of the note, and the
    deviation.
    """
    result = OrderedDict()
    # go through note by note and assign the tuning system labeling as the dict value
    # with the least deviation for that particular frequency, keep track of the deviation value
    for key in time_freq_dict:
        freq = time_freq_dict[key]
        tuning_system, deviation = identify_intonation_system(freq, freq_A4)
        result[key] = tuning_system, freq, deviation

    return result


def run_tuning_system_detection2(tuning_system_dict, num_notes_filter=3, min_len_seq=2,
                                 time_thresh=.55, freq_A4=440.0):
    """
    Runs entire change detection algorithm.
    1. Moving average filter.
    2. Identify sequences of consecutive labels.

    Returns a list of tuples (t_start, t_final, tuning_system)
    """
    predictions = run_tuning_system_change_detection(tuning_system_dict, num_notes_filter, freq_A4)
    print(f"predictions: {predictions}")
    frequencies = list(tuning_system_dict.values())
    # iterate through string of labels and identify sequences of consecutive labels.
    initial_sequences = []
    start_idx = 0
    # change_indices = [0]
    # # edge case: one reading in the beginning of the array
    # if predictions[1] != predictions[0]:
    #     start_idx = 1
    t_start = tuning_systems.get_dict_key(frequencies[start_idx], tuning_system_dict)[0]
    for i in range(1, len(predictions)):
        # print('i = ', i, sequences)
        if predictions[i] != predictions[i - 1]:
            # print('change occurred at i = ', i)
            # the shortest possible sequence is two consecutive notes
            # if i == change_indices[-1] + 1:
                # start_idx = i + 1
                # end_idx = i + 1
                # print(start_idx, end_idx)
                # continue
            # change_indices.append(i)
            end_idx = i - 1
            # if end_idx - start_idx == 0:
            #     start_idx = i
            #     t_start = tuning_systems.get_dict_key(frequencies[start_idx], tuning_system_dict)[0]
            #     continue
            last_freq = frequencies[end_idx]
            _, t_final = tuning_systems.get_dict_key(last_freq, tuning_system_dict)
            initial_sequences.append((t_start, t_final, predictions[end_idx], end_idx - start_idx + 1))
            start_idx = i
            freq_start = frequencies[start_idx]
            t_start, _ = tuning_systems.get_dict_key(freq_start, tuning_system_dict)
            continue
    # edge case: all the predictions are the same. In this case, the sequence is the entire
    # sequence.
    end_idx = len(frequencies) - 1
    last_freq = frequencies[end_idx]
    _, t_final = tuning_systems.get_dict_key(last_freq, tuning_system_dict)
    initial_sequences.append((t_start, t_final, predictions[i - 1], end_idx - start_idx + 1))
    # print(f"initial sequences: {initial_sequences}")
    # once we've gone through the entire array, do one final sweep
    # a) removing any cases of one note or fewer
    sequences = []
    for j in range(len(initial_sequences)):
        if initial_sequences[j][3] >= min_len_seq:
            sequences.append(initial_sequences[j])
    print(f"sequences: {sequences}")
    # sequences = [(0.04, 2.62499, 'just', 8), (4.04, 4.7449900000000005, 'just', 3), (4.975, 5.21499, 'edo', 2), (5.3, 9.06999, 'just', 13), (9.575, 11.11499, 'just', 7), (11.445, 12.214990000000002, 'just', 3), (12.405, 13.73999, 'edo', 4), (14.055, 16.08999, 'just', 7), (17.06, 18.10999, 'pythag', 3), (18.23, 18.584989999999998, 'edo', 2), (18.605, 19.29999, 'just', 4), (19.385, 20.244989999999998, 'edo', 3), (20.315, 21.46999, 'just', 4), (22.13, 23.00999, 'edo', 3), (23.035, 27.814989999999998, 'just', 14), (28.12, 28.46499, 'edo', 2), (28.515, 30.064989999999998, 'just', 8), (30.11, 30.744989999999998, 'edo', 2), (30.815, 32.31499, 'just', 4), (32.945, 33.859989999999996, 'edo', 3), (33.905, 36.35499, 'just', 9), (36.375, 37.04499, 'pythag', 2), (37.065, 37.44999, 'edo', 2), (37.48, 42.42499, 'just', 21), (42.67, 43.039989999999996, 'pythag', 2), (43.17, 43.88499, 'just', 3), (44.01, 44.40499, 'edo', 3), (44.51, 45.10499, 'just', 3), (45.15, 46.04999, 'edo', 3), (46.145, 51.83499, 'just', 24), (52.04, 52.40499, 'pythag', 3), (52.455, 53.234989999999996, 'just', 3), (53.475, 61.484989999999996, 'just', 22), (61.555, 62.62499, 'edo', 4), (62.82, 63.27499, 'edo', 2), (65.095, 69.17499, 'just', 15), (69.235, 70.11998999999999, 'edo', 3), (70.17, 71.19498999999999, 'just', 3), (71.26, 71.78499, 'pythag', 3), (71.83, 73.46498999999999, 'just', 5), (73.9, 75.50998999999999, 'edo', 5), (75.74, 76.74498999999999, 'just', 2), (76.91, 78.21999, 'edo', 3), (78.345, 80.78499, 'just', 4), (81.015, 81.60498999999999, 'edo', 3), (81.68, 82.57498999999999, 'just', 3), (82.715, 83.93499, 'edo', 2), (84.14, 85.03499, 'just', 3), (85.655, 86.95998999999999, 'just', 4), (87.325, 88.37998999999999, 'just', 4), (88.505, 89.21498999999999, 'edo', 2), (89.6, 91.03998999999999, 'just', 3), (91.1, 91.76998999999999, 'pythag', 3), (91.835, 98.13498999999999, 'just', 6), (98.665, 103.38498999999999, 'just', 10), (103.56, 107.48998999999999, 'edo', 6), (107.6, 110.75998999999999, 'just', 4), (111.21, 111.73998999999999, 'edo', 3), (111.89, 113.56998999999999, 'just', 6), (113.75, 113.94498999999999, 'edo', 2), (114.055, 116.81499, 'just', 4), (119.235, 122.61998999999999, 'just', 7), (122.875, 124.15499, 'edo', 3), (124.755, 128.18999, 'just', 5), (128.28, 128.60499, 'edo', 2), (128.88, 129.01998999999998, 'just', 2), (129.52, 133.23498999999998, 'just', 8), (133.365, 133.53499, 'edo', 2), (134.72, 135.98999, 'just', 4), (136.615, 141.95999, 'just', 14), (142.405, 148.94998999999999, 'just', 19), (149.085, 149.74999, 'edo', 3), (149.77, 151.12999, 'just', 8), (151.61, 152.99499, 'just', 9), (153.56, 157.96999, 'just', 16), (158.075, 158.65999, 'edo', 3), (158.915, 163.84499, 'just', 14)]

    # edge case: only one sequence found:
    if len(sequences) == 1:
        return sequences
    # b) "sandwiching any tuning system sequences based on a time threshold between
    # adjacent sequences
    final_sequences = []
    if sequences[1][2] == sequences[0][2]:
        if sequences[1][0] - sequences[0][1] < time_thresh:
            final_sequences.append((sequences[0][0], sequences[1][1],
                                    sequences[1][2], sequences[0][3] + sequences[1][3]))
            start = 2
        else:
            final_sequences.append(sequences[0])
            final_sequences.append(sequences[1])
            start = 2
    else:
        final_sequences.append(sequences[0])
        start = 1

    for k in range(start, len(sequences)):
        # if two adjacent sequences have the same label, check the time difference
        # to see if they can be sandwiched into one sequence
        if sequences[k][2] == sequences[k - 1][2]:
            if sequences[k][0] - sequences[k - 1][1] < time_thresh:
                final_sequences.append((sequences[k - 1][0], sequences[k][1],
                                        sequences[k][2], sequences[k - 1][3] + sequences[k][3]))
        else:
            final_sequences.append(sequences[k])

    return final_sequences


if __name__ == "__main__":
    test_dict_equal = OrderedDict(
        [((0.02, 0.47998999999999997), 261.7371000393325), ((0.515, 0.97999), 294.0568108488898),
         ((1.015, 1.4849899999999998), 329.6611441951898), ((1.515, 1.97999), 348.9253329596941),
         ((2.025, 2.48499), 392.29345157784684), ((2.52, 2.9899899999999997), 440.34798132037827),
         ((3.015, 3.49999), 494.15330780293453)])
    maisky_dict = OrderedDict([((0.095, 0.6699900000000001), 261.75677807329214),
                               ((0.75, 0.97999), 249.6003072074146),
                               ((1.22, 1.2799899999999997), 220.06420508906893),
                               ((1.43, 1.5799899999999998), 200.4594594153993),
                               ((1.71, 1.8199899999999998), 178.50852844566944),
                               ((1.94, 1.9649899999999998), 167.60637875085843),
                               ((2.045, 2.1949899999999998), 147.4740781152811),
                               ((3.11, 3.1149899999999997), 98.31306887815394),
                               ((3.475, 4.67999), 65.49040596310684)])
    ma_dict = OrderedDict([((0.03, 0.69499), 258.2666641837428),
                           ((0.76, 0.8999900000000001), 247.66575734798363),
                           ((0.94, 1.12499), 219.9819701942868),
                           ((1.215, 1.34499), 195.04853285988216),
                           ((1.435, 1.5299899999999997), 173.5182386513929),
                           ((1.565, 1.69999), 165.51187723503796),
                           ((1.985, 2.11999), 129.27158723249786),
                           ((2.215, 2.9899899999999997), 96.96714630796792),
                           ((3.875, 4.67999), 64.86510210610976)])

    carr_dict = OrderedDict([((0.015, 1.1199899999999998), 256.340879880009),
                             ((1.235, 1.2449899999999998), 220.99128268079824),
                             ((1.345, 1.4249899999999998), 196.2114708338092),
                             ((1.51, 1.5799899999999998), 174.99393772834318),
                             ((1.71, 1.7149899999999998), 170.84333172514795),
                             ((2.075, 2.16499), 390.6462783662873),
                             ((2.335, 2.33999), 247.7263744442558),
                             ((3.115, 4.14999), 65.21665721168117)])
    # res = run_tuning_system_clustering_binary(test_dict)
    # print(res)

    # answer = [(OrderedDict([((0.285, 0.85499), ('edo', 261.74283575788337, 0.04480369326829134)),
    #                         ((0.935, 1.19999), ('just', 249.6296668386235, 0.27298208627105863)),
    #                         ((1.44, 1.44999), ('just', 219.84608403287461, 0.0700107840457917)),
    #                         ((1.61, 1.7649899999999998), ('just', 200.33746599562357, 1.1667642814622787)),
    #                         ((1.895, 2.03999), ('just', 178.2984941930864, 0.05524118054511475)),
    #                         ((2.13, 2.14499), ('just', 167.85730084034174, 0.5857515647787508)),
    #                         ((2.225, 2.3849899999999997), ('edo', 147.40229951754537, 0.38663953052762284)),
    #                         ((2.98, 2.98499), ('just', 165.2123600288509, 0.12853761595912744))]), 'just'),
    #
    #           (OrderedDict([((3.29, 3.29499), ('edo', 98.25154971244494, 0.25718751281495034)),
    #                         ((3.66, 4.859990000000001), ('edo', 65.48787887023394, 0.12443149249915678)),
    #                         ((5.0, 5.5), ('edo', 66.4, .11))]), 'edo')]

    # res2, cost = run_tuning_system_clustering_brute_force(test_dict)
    # print(res2)
    # print(cost)
    # test_cluster = list(span(list(test_dict.values())))[2]
    # calculate_num_switches(test_cluster)
    test_dict_just = OrderedDict([((0.025, 0.48499), 219.96339256405656), ((0.52, 0.97499), 247.48998670049488),
                                  ((1.015, 1.47999), 275.2313611696483), ((1.525, 1.97999), 293.75538990304966),
                                  ((2.015, 2.4899899999999997), 329.94518315782875),
                                  ((2.52, 2.98499), 367.929463534712),
                                  ((3.025, 3.4899899999999997), 412.26946267442247),
                                  ((3.525, 3.9899899999999997), 440.3086410278321)])
    test_dict_carr = OrderedDict([((0.0, 1.12499), 256.2628782194918), ((1.195, 1.24999), 221.07984704923862), ((1.325, 1.4299899999999999), 196.01504029628853), ((1.5, 1.58499), 175.02832646753316), ((1.645, 1.7649899999999998), 170.19658655209338), ((2.055, 2.1799899999999997), 390.5190945595405), ((2.33, 2.37499), 247.7637372924045), ((2.65, 2.7399899999999997), 195.48739004806725), ((3.015, 4.154990000000001), 65.21613636040962)])

    sequences_test = run_tuning_system_detection2(ma_dict, num_notes_filter=3, min_len_seq=2,
                                 time_thresh=.55, freq_A4=440.0)
    print(sequences_test)


