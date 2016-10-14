import os
import numpy as np


class Dataset(object):
    """
    This class represents a dataset and consists of a list of SongData along with some metadata about the dataset
    """

    def __init__(self, songs_data=None):
        if songs_data is None:
            self.songs_data = []
        else:
            self.songs_data = songs_data

    def add_song(self, song_data):
        self.songs_data.append(song_data)

    def songs(self):
        for s in self.songs_data:
            yield s

    @property
    def num_features(self):
        if len(self.songs_data):
            return self.songs_data[0].X.shape[1]

    @property
    def size(self):
        return len(self.songs_data)

    def __repr__(self):
        return ', '.join([s.name for s in self.songs()])


class SongData(object):
    """
    This class holds features, labels, and metadata for a song.
    """

    def __init__(self, audio_path, label_path):
        if not os.path.isfile(audio_path):
            raise IOError("Audio file at %s does not exist" % audio_path)
        if label_path and not os.path.isfile(label_path):
            raise IOError("MIDI file at %s does not exist" % label_path)

        self.audio_path = audio_path
        self.label_path = label_path

    """
    x [num_samples,] is the samples of the song
    """
    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    """
    X [num_frames x num_features] is the feature matrix for the song
    """
    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, X):
        if hasattr(self, 'Y') and self.Y.shape[0] != X.shape[0]:
            raise ValueError("Number of feature frames must equal number of label frames")
        self.__X = X

    """
    Y [num_frames x num_pitches] is the label matrix for the song
    """
    @property
    def Y(self):
        return self.__Y

    @Y.setter
    def Y(self, Y):
        if hasattr(self, 'X') and self.X.shape[0] != Y.shape[0]:
            raise ValueError("Number of label frames must equal number of feature frames")
        self.__Y = Y

    @property
    def num_pitches(self):
        if hasattr(self, 'Y'):
            return np.shape(self.Y)[1]
        return 0

    @property
    def num_features(self):
        if hasattr(self, 'X'):
            return self.X.shape[1]

    @property
    def num_frames(self):
        if hasattr(self, 'X'):
            return self.X.shape[0]

    @property
    def name(self):
        return os.path.splitext(os.path.split(self.audio_path)[-1])[0]

