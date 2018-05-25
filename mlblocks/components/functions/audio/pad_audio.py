import numpy as np
from numpy.lib import pad


class AudioPadder(object):
    def __init__(self):
        self.padding = 0

    def fit(self, audio_features):
        # Just obtain padding.
        for features in audio_features:
            if len(features) > self.padding:
                self.padding = len(features)

    def produce(self, audio_features):
        # Pad our features.
        if len(audio_features[0]) < self.padding:
            padded_features = pad(
                audio_features[0],
                (0, self.padding - len(audio_features[0])),
                'constant',
                constant_values=0
            )
        else:
            padded_features = np.array(audio_features[0][:self.padding])
        for i in range(1, len(audio_features)):
            if len(audio_features[i]) < self.padding:
                padded = pad(
                    audio_features[i],
                    (0, self.padding - len(audio_features[i])),
                    'constant',
                    constant_values=0
                )
            else:
                padded = np.array(audio_features[i][:self.padding])
            padded_features = np.vstack((padded_features, padded))

        return np.nan_to_num(padded_features)
