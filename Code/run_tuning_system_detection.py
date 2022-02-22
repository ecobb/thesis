from Modules.freq_detection import identify_tuning_system


def extract_ref_pitch_and_scale(f0_dict):
    """

    """
    f0_dict_items = f0_dict.items()
    # TODO: generalize
    scale = list(f0_dict_items)[:8]

    return ref_pitch, scale


def tuning_system_detection(file, scale_freq):
    # TODO:
    ref_pitch, scale_freq = extract_ref_pitch_and_scale(file) #TODO
    tuning_system = identify_tuning_system(ref_pitch, scale_freq)

    return tuning_system


if __name__ == "__main__":

    tuning_system = tuning_system_detection(file, scale_freq)
    print(tuning_system)
