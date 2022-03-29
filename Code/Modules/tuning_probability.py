import collections

import numpy as np
import scipy.stats

from Modules.performance_statistics import calculate_cents
from Modules.tuning_systems import create_just_template, calculate_delta_one_note, find_closest_freq, \
    create_edo_template, create_pythag_template


def find_just_ref_freq(freq, freq_A4=440.0):
    """
    This helper function finds the closest match frequency out of all the just
    intonation stencils.
    """
    # find the reference pitch to approximate for the mean
    # generate all possible stencils and closest ref frequencies
    c_just_template = create_just_template('C', freq_A4)
    c_just_delta = calculate_delta_one_note(freq, c_just_template)
    c_pitch, c_ref_freq = find_closest_freq(freq, c_just_template)
    # print(f"pitch in C stencil: {c_pitch}, freq in C stencil: {c_ref_freq}, delta: {c_just_delta}")

    d_just_template = create_just_template('D', freq_A4)
    d_just_delta = calculate_delta_one_note(freq, d_just_template)
    d_pitch, d_ref_freq = find_closest_freq(freq, d_just_template)
    # print(f"pitch in D stencil: {d_pitch}, freq in D stencil: {d_ref_freq}, delta: {d_just_delta}")

    e_just_template = create_just_template('E', freq_A4)
    e_just_delta = calculate_delta_one_note(freq, e_just_template)
    e_pitch, e_ref_freq = find_closest_freq(freq, e_just_template)
    # print(f"pitch in E stencil: {e_pitch}, freq in E stencil: {e_ref_freq}, delta: {e_just_delta}")

    f_just_template = create_just_template('F', freq_A4)
    f_just_delta = calculate_delta_one_note(freq, f_just_template)
    f_pitch, f_ref_freq = find_closest_freq(freq, f_just_template)
    # print(f"pitch in F stencil: {f_pitch}, freq in F stencil: {f_ref_freq}, delta: {f_just_delta}")

    g_just_template = create_just_template('G', freq_A4)
    g_just_delta = calculate_delta_one_note(freq, g_just_template)
    g_pitch, g_ref_freq = find_closest_freq(freq, g_just_template)
    # print(f"pitch in G stencil: {g_pitch}, freq in G stencil: {g_ref_freq}, delta: {g_just_delta}")

    a_just_template = create_just_template('A', freq_A4)
    a_just_delta = calculate_delta_one_note(freq, a_just_template)
    a_pitch, a_ref_freq = find_closest_freq(freq, a_just_template)
    # print(f"pitch in A stencil: {a_pitch}, freq in A stencil: {a_ref_freq}, delta: {a_just_delta}")

    ref_freq = np.array([c_ref_freq, d_ref_freq, e_ref_freq, f_ref_freq, g_ref_freq, a_ref_freq])
    deltas = np.array([c_just_delta, d_just_delta, e_just_delta, f_just_delta, g_just_delta,
                       a_just_delta])
    keys = np.array(['C', 'D', 'E', 'F', 'G', 'A'])

    # min_idx = np.argmin(deltas)
    # if not return_keys:
    #     # the final result is the frequency that corresponds to the least delta
    #     return ref_freq[min_idx]
    # else:
    #     # extract the index locations of all values matching the minimum
    #     min_indices = np.where(deltas == deltas.min())
    #     return ref_freq[min_indices[0]], keys[min_indices]
    min_indices = np.where(deltas == deltas.min())
    # print(f"min_indices = {min_indices}")
    return ref_freq[min_indices[0][0]], keys[min_indices]


def find_just_ref_freq_with_key(freq, key, freq_A4=440.0):
    template = create_just_template(key, freq_A4)
    pitch, ref_freq = find_closest_freq(freq, template)

    return ref_freq


def calculate_ref_std(ref_freq):
    """
    Helper function to calculate the standard deviation of a frequency.
    """
    freq_upper = ref_freq * 2 ** (1 / 24)
    sigma = np.log(freq_upper - ref_freq)
    return sigma


def calc_just_prob(freq, freq_A4=440.0):
    """
    This helper function calculates the probability of a frequency estimate from a
    just intonation by finding the closest possible match in the just
    system's stencil. The probability is assumed to follow a Gaussian distribution
    centered around this reference frequency.
    """
    # if return_keys:
    #     just_ref_freq, just_keys = find_just_ref_freq(freq, True, freq_A4)
    #     # print(f"just ref freq: {just_ref_freq}")
    #     # ref_std = calculate_ref_std(just_ref_freq)
    #     just_prob = scipy.stats.norm.pdf(freq, loc=just_ref_freq, scale=.5)
    #     return just_prob, just_keys
    # else:
    #     just_ref_freq = find_just_ref_freq(freq, freq_A4)
    #     # print(f"just ref freq: {just_ref_freq}")
    #     # ref_std = calculate_ref_std(just_ref_freq)
    #     just_prob = scipy.stats.norm.pdf(freq, loc=just_ref_freq, scale=.5)
    #     return just_prob

    just_ref_freq, just_keys = find_just_ref_freq(freq, freq_A4)
    # print(f"just ref freq: {just_ref_freq}")
    # ref_std = calculate_ref_std(just_ref_freq)
    just_prob = scipy.stats.norm.pdf(freq, loc=just_ref_freq, scale=.5)
    return just_prob, just_keys


def calc_just_prob_with_key(freq, key, freq_A4=440.0):
    """
    This helper function calculates the probability of a frequency estimate from a
    just intonation by finding the closest possible match in the just
    system's stencil. The probability is assumed to follow a Gaussian distribution
    centered around this reference frequency.
    """
    just_ref_freq = find_just_ref_freq_with_key(freq, key, freq_A4)
    # print(f"just ref freq: {just_ref_freq}")
    # ref_std = calculate_ref_std(just_ref_freq)
    just_prob = scipy.stats.norm.pdf(freq, loc=just_ref_freq, scale=.5)

    return just_prob


def find_edo_ref_freq(freq, freq_A4):
    """
    This helper function finds the closest match frequency in the equal
    tempered stencil.
    """
    edo_template = create_edo_template(freq_A4)
    edo_ref_pitch, edo_ref_freq = find_closest_freq(freq, edo_template)

    return edo_ref_freq


def calc_edo_prob(freq, freq_A4=440.0):
    """
    This helper function calculates the probability of a frequency estimate from
    equal temperament by finding the closest possible match in the equal-tempered
    stencil. The probability is assumed to follow a Gaussian distribution
    centered around this reference frequency.
    """
    edo_ref_freq = find_edo_ref_freq(freq, freq_A4)
    # print(f"edo ref freq: {edo_ref_freq}")
    # ref_std = calculate_ref_std(edo_ref_freq)
    edo_prob = scipy.stats.norm.pdf(freq, loc=edo_ref_freq, scale=.5)

    return edo_prob


def find_pythag_ref_freq(freq, freq_A4):
    """
    This helper function finds the closest match frequency in the pythagorean
    stencil.
    """
    pythag_template = create_pythag_template(freq_A4)
    pythag_ref_pitch, pythag_ref_freq = find_closest_freq(freq, pythag_template)

    return pythag_ref_freq


def calc_pythag_prob(freq, freq_A4=440.0):
    """
    This helper function calculates the probability of a frequency estimate from
    equal temperament by finding the closest possible match in the equal-tempered
    stencil. The probability is assumed to follow a Gaussian distribution
    centered around this reference frequency.
    """
    pythag_ref_freq = find_pythag_ref_freq(freq, freq_A4)
    # print(f"pythag ref freq: {pythag_ref_freq}")
    # ref_std = calculate_ref_std(pythag_ref_freq)
    # std = abs(calculate_cents(pythag_ref_freq, freq))
    # print(std)
    pythag_prob = scipy.stats.norm.pdf(freq, loc=pythag_ref_freq, scale=.5)
    # pythag_prob = scipy.stats.norm.pdf(freq, loc=pythag_ref_freq, scale=std)

    return pythag_prob


def calc_probability_vector(freq, freq_A4=440.0):
    edo_prob = calc_edo_prob(freq, freq_A4)
    pythag_prob = calc_pythag_prob(freq, freq_A4)

    # if return_key:
    #     just_prob, just_key = calc_just_prob(freq, True, freq_A4)
    #     return np.array([just_prob, edo_prob, pythag_prob]), just_key
    # else:
    #     just_prob = calc_just_prob(freq, False, freq_A4)
    #     return np.array([just_prob, edo_prob, pythag_prob])
    just_prob, just_key = calc_just_prob(freq, freq_A4)
    return np.array([just_prob, edo_prob, pythag_prob]), just_key


def calc_probability_vector_with_key(freq, key, freq_A4=440.0):
    return np.array([calc_just_prob_with_key(freq, key, freq_A4), calc_edo_prob(freq, freq_A4),
                     calc_pythag_prob(freq, freq_A4)])


def calculate_result_key(just_keys):
    just_keys_flattened = np.array(np.concatenate(just_keys).ravel(), dtype=object)
    # unique, counts = np.unique(just_keys_flattened, return_counts=True)
    result_key = collections.Counter(just_keys_flattened).most_common()[0][0]
    return result_key


def calc_intonation_system(freq_arr, key, with_key=False, freq_A4=440.0, return_key=False):
    """
    This helper function computes the average probability of a sequence of frequencies
    by taking the maximum average probability estimate among the 3 intonation systems.
    """
    # iterate through frequencies
    prob_estimates = []
    if with_key:
        for freq in freq_arr:
            prob_vector_with_key = calc_probability_vector_with_key(freq, key, freq_A4)
            prob_estimates.append(prob_vector_with_key)
    else:
        # if return_key:
        #     for freq in freq_arr:
        #         prob_vector, just_key = calc_probability_vector(freq, True, freq_A4)
        #         prob_estimates.append(prob_vector)
        # else:
        #     for freq in freq_arr:
        #         prob_vector, just_key = calc_probability_vector(freq, return_key, freq_A4)
        #         prob_estimates.append(prob_vector)
        just_keys = []
        for freq in freq_arr:
            # print(f"actual freq: {freq}")
            prob_vector, just_key = calc_probability_vector(freq, freq_A4)
            # print(prob_vector, just_key)
            prob_estimates.append(prob_vector)
            just_keys.append(just_key)

        result_key = calculate_result_key(just_keys)
        # print(result_key)

    prob_estimates = np.array(prob_estimates)
    avg_estimates = np.mean(prob_estimates, axis=0)
    # print(f"just prob: {avg_estimates[0]}, edo prob: {avg_estimates[1]}, pythag prob: {avg_estimates[2]}")
    max_avg_idx = np.argmax(avg_estimates)

    if max_avg_idx == 0:
        if with_key:
            return key.lower() + ' just'
        else:
            return result_key.lower() + ' just'
    if max_avg_idx == 1:
        return 'edo'
    if max_avg_idx == 2:
        return 'pythag'


if __name__ == "__main__":
    # print(calc_probability_vector(350, 441.0))
    # print(calculate_ref_std(220))
    test_edo = [220 * 2 ** (i / 12) for i in range(13)]
    test_just = [220., 247.5, 275., 293.33333333,
                 330., 366.66666667, 412.5, 440.]
    test_pythag = [220., 247.5, 278.4375, 293.33333333,
                   330., 371.25, 417.65625, 440.]
    print(calc_intonation_system(test_pythag, key='A', with_key=False, freq_A4=220.0))
