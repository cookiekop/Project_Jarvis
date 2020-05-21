import numpy as np
import tensorflow as tf

import params
import yamnet as yamnet_model


class SoundDetector:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.yamnet = yamnet_model.yamnet_frames_model(params)
            self.yamnet.load_weights('yamnet/yamnet.h5')
        self.yamnet_classes = yamnet_model.class_names('yamnet/yamnet_class_map.csv')

    def listen(self, wav_data):
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        # Convert to mono and the sample rate expected by YAMNet.

        with self.graph.as_default():
            scores, _ = self.yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
            # Scores is a matrix of (time_frames, num_classes) classifier scores.
            # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.

        toRet = []
        top5_i = np.argsort(prediction)[::-1][:5]
        for i in top5_i:
            toRet.append((self.yamnet_classes[i], prediction[i]))
            print(self.yamnet_classes[i], prediction[i])
        return toRet