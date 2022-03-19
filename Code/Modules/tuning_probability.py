import numpy as np
import scipy.stats

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

    deltas = [c_just_delta, d_just_delta, e_just_delta, f_just_delta, g_just_delta,
              a_just_delta]

    min_delta = min(deltas)
    idx = deltas.index(min_delta)
    # the final result is the frequency that corresponds to the least delta
    if idx == 0:
        return c_ref_freq
    if idx == 1:
        return d_ref_freq
    if idx == 2:
        return e_ref_freq
    if idx == 3:
        return f_ref_freq
    if idx == 4:
        return g_ref_freq
    if idx == 5:
        return a_ref_freq

    raise ValueError


def calculate_ref_std(ref_freq):
    """
    Helper function to calculate the standard deviation of a frequency.
    """
    freq_upper = ref_freq*2**(1/24)
    sigma = np.log(freq_upper - ref_freq)
    return sigma


def calc_just_prob(freq, freq_A4=440.0):
    """
    This helper function calculates the probability of a frequency estimate from a
    just intonation by finding the closest possible match in the just
    system's stencil. The probability is assumed to follow a Gaussian distribution
    centered around this reference frequency.
    """
    just_ref_freq = find_just_ref_freq(freq, freq_A4)
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
    pythag_prob = scipy.stats.norm.pdf(freq, loc=pythag_ref_freq, scale=.5)

    return pythag_prob


def calc_probability_vector(freq, freq_A4=440.0):
    return np.array([calc_just_prob(freq, freq_A4), calc_edo_prob(freq, freq_A4),
                     calc_pythag_prob(freq, freq_A4)])


def calc_probability_cluster(freq_arr, freq_A4=440.0):
    """
    This helper function computes the average probability of a cluster by taking the
    maximum average probability estimate among the 3 intonation systems.
    """
    # iterate through frequencies
    prob_estimates = []
    for freq in freq_arr:
        prob_estimates.append(calc_probability_vector(freq, freq_A4))

    prob_estimates = np.array(prob_estimates)
    avg_estimates = np.mean(prob_estimates, axis=0)
    max_avg_idx = np.argmax(avg_estimates)

    if max_avg_idx == 0:
        return 'just'
    if max_avg_idx == 1:
        return 'edo'
    if max_avg_idx == 2:
        return 'pythag'

    return avg_estimates


if __name__ == "__main__":
    # print(calc_probability_vector(350, 441.0))
    # print(calculate_ref_std(220))
    print(calc_probability_cluster(np.array([441, 441/(2**(1/12)), 441*2**(1/12), 441/2]), 441.0))
