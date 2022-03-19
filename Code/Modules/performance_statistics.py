import numpy as np


def calculate_cents(freq1, freq2):
    return 1200*np.log(freq2/freq1)/np.log(2)


def calculate_cent_deviation(freq1, freq2):
    """
    Calculates per-element cent deviation between two frequency arrays
    """
    cent_deviation = np.zeros(len(freq1))
    for i in range(len(cent_deviation)):
        cent_deviation[i] = calculate_cents(freq2[i], freq1[i])

    return cent_deviation

