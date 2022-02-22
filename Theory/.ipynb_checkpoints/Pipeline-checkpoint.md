Here we will describe the machine learning pipeline to classify intonation systems. 

We need to extract the frequency values of the notes from these recordings with high accuracy. Using Crepe, we can extract the frequency values from a wav file - Crepe employs a deep convolutional network for its prediction. From this, we obtain arrays with the time spacing using in prediction, frequency values at each timestamp, confidence, and activation. 

Due to the nature of the recordings, there is a good deal of noise in the actual measured values of frequencies obtained. We must decide how to best estimate the pitch from these values. 

Once we have frequency values, how will we incorporate these values into the model? 

As a simple case, we want to extract sections of the recording in which there is a linear ascending or descending scale. Thus, we can take the root of the scale as the dominating pitch, equal-tempered values can be obtained by supplying the fundamental frequency of A3. Then we can just compute the just and pythagorean cost based on that dominating pitch and take the minimum cost system as the tuning system classification. 

Approach 1: Unsupervised Learning Based on Context (Text Classification)

If we are able to extract the notes in the piece, we could employ a n-grams approach or 'bag of pitches' model. If identifiying intonation system of a scale, then the network must parse the notes or 'words' of the piece, pick up the context in which there is an ascending or descending scale and from that, perform fundamental frequency estimation and finally classify based on minimum mean squared error. 

Approach 2: Supervised Learning based on Generated Labels 

In this approach, we must split the wav files into frames and capture sequences of notes with which we can compute the tuning classification probabilities. In the simple case where n=2, we only consider two notes. Let's say we have B3 with frequency 243 Hz and G3 with frequency 196 Hz. In isolation, this forms a major third interval with ratio ~= 1.24. The Just, ET, and Pythagorean values for the interval are 1.25, 1.26, and 1.27 respectively so it will be labeled as Just. In this case, the order of the notes doesn't matter; whether the higher-pitched note (B3) appeared first and thus formed a descending interval or the lower-pitched note (G3) appeared first and thus formed an ascending interval does not matter. The n=2 case thus clearly exhibits reflective symmetry. 

Let us consider the n=3 case. Let's say we were given the following succession of notes: B3 (246), G3(196), and D3 (146). If analyzing pair-wise interactions of notes, B3+G3 make up a 1.26 ratio and G3+D3 make up a 1.37 ratio. The first interaction forms a major third as shown before; the second interaction forms a perfect fourth and the Just, ET, and Pythagorean values are 1.333, 1.335, and 1.333. Thus this would be labeled as ET. How would the overall instance be labeled? Since we have only 2 pairs, we must consider which gives more weight. Or we could flip a coin if we consider them to have equal probability. We'd expect this to not work as well given that the Bach Suites are situated in a musical time period in which it is generally expected to contain more Just intonation information. 

The preprocessing pipeline (for an individual recording sample) is as follows: 
- Convert mp3 file to wav file
- Split wav file into short segments of ? notes*
- Generate label for segment (Just, ET, Pythagorean)
- Extract audio features for segment (log spectrogram, MFCCs, etc)
- save audio features as numpy arrays

* we could just stick to extracting two note clips, in which case we only need to compute the frequency ratio of the notes and then we have our answer for the label. For n > 2, estimation needs to be involved as described above. 

As for the features, we want a mix of pure audio features and other more qualitative features (to be converted via one-hot encoding). These more qualitative descriptors could include: name of the cellist, birth year of the cellist, age at time of recording, etc. As for audio features, we want to at least include MFCCs, spectrograms, zero-crossing rate, RMS energy, spectral centroid, bandwidth, etc. 