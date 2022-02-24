from collections import OrderedDict
from itertools import islice


def calculate_num_switches(tuning_system_dict):
    """
    Helper function to calculate the number of times a label switches
    E.g., 'edo', 'edo', 'just', 'just', 'edo', 'just' -> 3
    """
    vals = tuning_system_dict.values()
    labels = [a_tuple[0] for a_tuple in vals]

    num_switches = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            num_switches += 1

    return num_switches

def compute_cost(tuning_system_dict):
    """
    This function computes the cost of a cluster. Switches between tuning systems
    are penalized as well as deviation cost.
    """
    num_switches = calculate_num_switches(tuning_system_dict)

    return num_switches


def run_tuning_system_clustering(tuning_system_dict):
    """
    This function takes in a dictionary of note timestamps and their tuning system labelings
    and organizes them into clusters as part of the last step of the preprocessing pipeline.

    The final result is a dict of dicts; each entry in the dictionary is a cluster of notes.
    """
    result = OrderedDict()
    # make initial split in the data
    inc = iter(tuning_system_dict.items())
    left = dict(islice(inc, len(tuning_system_dict) // 2))
    right = dict(inc)
    # explore the branch with the higher cost first
    left_cost = compute_cost(left)
    right_cost = compute_cost(right)
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


if __name__ == "__main__":
    test_dict = OrderedDict([((0.285, 0.85499), ('edo', 0.04480369326829134)),
                             ((0.935, 1.19999), ('just', 0.27298208627105863)),
                             ((1.44, 1.44999), ('just', 0.0700107840457917)),
                             ((1.61, 1.7649899999999998),
                             ('just', 1.1667642814622787)),
                             ((1.895, 2.03999), ('just', 0.05524118054511475)),
                             ((2.13, 2.14499), ('just', 0.5857515647787508)),
                             ((2.225, 2.3849899999999997), ('edo', 0.38663953052762284)),
                             ((2.98, 2.98499), ('just', 0.12853761595912744)),
                             ((3.29, 3.29499), ('edo', 0.25718751281495034)),
                             ((3.66, 4.859990000000001), ('edo', 0.12443149249915678))])

    res = calculate_num_switches(test_dict)
    print(res)
