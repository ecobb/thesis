import sympy as sp
import freq_detection


class Just:
    def __init__(self):
        pass

    def get_just_unison_ratio(self):
        return sp.Integer(1)

    def get_just_minor_second_ratio(self):
        return sp.Rational(25, 24)

    def get_just_major_second_ratio(self):
        return sp.Rational(9, 8)

    def get_just_minor_third_ratio(self):
        return sp.Rational(6, 5)

    def get_just_major_third_ratio(self):
        return sp.Rational(5, 4)

    def get_just_perfect_fourth_ratio(self):
        return sp.Rational(4, 3)

    def get_just_tritone_ratio(self):
        return sp.Rational(45, 32)

    def get_just_perfect_fifth_ratio(self):
        return sp.Rational(3, 2)

    def get_just_minor_sixth_ratio(self):
        return sp.Rational(8, 5)

    def get_just_major_sixth_ratio(self):
        return sp.Rational(5, 3)

    def get_just_minor_seventh_ratio(self):
        return sp.Rational(9, 5)

    def get_just_major_seventh_ratio(self):
        return sp.Rational(15, 8)

    def get_just_octave_ratio(self):
        return sp.Integer(2)

    def calculate_just_interval(self, interval):
        if interval == 'P1':
            return self.get_just_unison_ratio()
        if interval == 'm2':
            return self.get_just_minor_second_ratio()
        if interval == 'M2':
            return self.get_just_major_second_ratio()
        if interval == 'm3':
            return self.get_just_minor_third_ratio()
        if interval == 'M3':
            return self.get_just_major_third_ratio()
        if interval == 'P4':
            return self.get_just_perfect_fourth_ratio()
        # NOTE: TREATING THESE INTERVALS AS SAME RATIO
        if interval == 'd5' or interval == 'A4':
            return self.get_just_tritone_ratio()
        if interval == 'P5':
            return self.get_just_perfect_fifth_ratio()
        if interval == 'm6':
            return self.get_just_minor_sixth_ratio()
        if interval == 'M6':
            return self.get_just_major_sixth_ratio()
        if interval == 'm7':
            return self.get_just_minor_seventh_ratio()
        if interval == 'M7':
            return self.get_just_major_seventh_ratio()
        if interval == 'P8':
            return self.get_just_octave_ratio()
        else:
            return "Unrecognized interval"

    def calculate_just_frequency(self, pitch, ref_pitch):
        """
        Returns the frequency estimate of a given pitch in scientific notation in just
        intonation based on an arbitrary reference pitch.

        pitch: str, pitch name
        ref_pitch: tuple, (ref pitch name, frequency (Hz))

        E.g., calculate_just_frequency('G3', ('C3', 131)) -> 196.5
        """
        # identify interval between pitches (normalized) and return the corresponding just ratio
        # relative to ref_pitch
        ref_pitch_str = ref_pitch[0]
        ref_pitch_freq = ref_pitch[1]

        interval = freq_detection.identify_interval_spn(ref_pitch_str, pitch)
        print(interval)
        just_frequency = (ref_pitch_freq * self.calculate_just_interval(interval)).evalf()

        return just_frequency


if __name__ == "__main__":
    just = Just()
    pitch = 'G3'
    ref_pitch = ('C3', 131)
    print(just.calculate_just_frequency(pitch, ref_pitch))
