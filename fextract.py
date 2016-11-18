from __future__ import division
from glob import glob
import os
import numpy as np
import sys
import librosa
from scipy import signal

from dataset import SongData, Dataset
from midiio import MidiIO
import matplotlib.pyplot as plt

class FeatureExtractor(object):
    """
    For each audio file, extract features for each window and 
    extract the corresponding pitch labels for each frame.
    """

    VALID_AUDIO_FORMATS = ("wav", "mp3")

    def __init__(self, audio_path, label_path, window_size, hop_size, sampling_rate):
        """
        PARAMETERS:
            audio_path (string): path to audio files
            label_path (string): path to midi files
            window_size (int): window sample size (samples)
            hop_size (int): window hop size (samples)
            sampling_rate (int): audio sampling rate (Hz)
        """

        # starting MIDI number of class labels, i.e., class label 0 is MIDI pitch 36
        self.pitch_offset = 36

        # 51 pitches: MIDI number 36 (C2: 65.406Hz) -- MIDI number 86 (D6: 1174.7Hz)
        self.num_pitches = 51

        self.window_size = window_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate

        audio_files = []
        if os.path.isdir(audio_path):
            for audio_format in FeatureExtractor.VALID_AUDIO_FORMATS:
                audio_files.extend(glob("%s/*.%s" % (audio_path, audio_format)))
        if not len(audio_files):
            raise LookupError("Can not find any audio files (%s) to process" % ", ".join(FeatureExtractor.VALID_AUDIO_FORMATS))

        self.dataset = Dataset()
        for audio_file in audio_files:
            fname = os.path.splitext(os.path.split(audio_file)[-1])[0]
            mid_file = os.path.join(label_path, '%s.mid' % fname)
            if not os.path.isfile(mid_file):
                mid_file = None
            song_data = SongData(audio_file, mid_file)
            self.dataset.add_song(song_data)

    def extract_features_and_labels(self):
        """
        Calculate features and labels for the songs in the given path.
        """

        self._extract_features()
        self._extract_labels()

    def _extract_features(self):
        """
        Extract the windowed features for each audio file in each dataset partition
        """

        num_songs = self.dataset.size
        for i, s in enumerate(self.dataset.songs()):
            s.x, _ = librosa.load(s.audio_path, sr=self.sampling_rate, mono=True)
              
            """
            TODO: calculate your audio features here
            You can use librosa to calculate the audio features you want to use.
            If you want to calculate your own features, you can slice the audio samples into frames using 
                librosa.util.frame(s.x, frame_length=self.window_size, hop_length=self.hop_size) and then
                calculate your features for each frame of audio samples.
            # In the end, you'll have s.X = [window_index, feature_index]

            For now, your audio features are simply the windowed samples of the song. It won't work for pitch detection.
            """
            s.X = librosa.util.frame(s.x, frame_length=self.window_size, hop_length=self.hop_size)

            # update progress bar
            self._speak('\rextracting features: %d%%' % int((i+1)/num_songs * 100))

        self._speak('\n')
        print len(s.X)
        print len(s.X[0])
        # for i in range(len(s.X[0])):
        #     window0 = s.X[:, i]
        # #for window0 in s.X:
        # #window0 = s.X[0]
        #     sp = np.fft.fft(window0)
        #     freq = np.fft.fftfreq(len(window0))
        #     print len(freq)
        #     print freq
        #     print len(sp.real)
        #     print sp.real
        #     plt.plot(freq, sp.real, freq, sp.imag)
        #     plt.show()


    def _extract_labels(self):
        """
        Extract the pitch labels for each analysis window of the corresponding
        audio file by parsing the midi file and finding notes that sound during
        the time of the analysis window.
        """

        num_songs = self.dataset.size
        for i, s in enumerate(self.dataset.songs()):
            if not os.path.isfile(s.label_path):
                raise LookupError('Can not find corresponding midi file for %s' % s.audio_path)

            num_wins = np.shape(s.X)[0]
            midiio = MidiIO(s.label_path)
            note_events = midiio.parse_midi(pitch_low_passband=self.pitch_offset, pitch_high_passband=(self.pitch_offset+self.num_pitches-1))
            nmat = np.array([[float(n.midi_number), n.onset_ts, n.offset_ts] for n in note_events], dtype=np.float32)

            # the label matrix is an indicator vector indicating the presence of a pitch
            s.Y = np.zeros([num_wins, self.num_pitches], dtype=np.float32)

            for iwin in xrange(num_wins):
                # calculate window start and end times (s)
                w_start = iwin*self.hop_size/self.sampling_rate
                w_end = (iwin*self.hop_size + self.window_size)/self.sampling_rate

                # logic to find note indices that sound during the current analysis window
                nidx = np.nonzero(np.logical_or(np.logical_or(
                    np.logical_and(nmat[:,1] < w_start, nmat[:,2] > w_end),
                    np.logical_and(nmat[:,1] > w_start, nmat[:,1] < w_end)),
                    np.logical_and(nmat[:,2] > w_start, nmat[:,2] < w_end)))[0]

                # populate pitch indicators in label matrix
                # label matrix is [num_wins x num_pitches] where num_pitches is number of pitches capable of being produced by the instrument
                # rwows are binary vectors, where a 1 indicates the presence of MIDI number
                pitch_indicators = np.unique(nmat[nidx,0]).astype(np.uint32) - self.pitch_offset
                s.Y[iwin, pitch_indicators] = 1.0

            self._speak('\rextracting labels: %d%%' % int((i+1)/num_songs * 100))

        self._speak('\n')
        print s.Y
        print len(s.Y)
        print len(s.Y[0])

    def _speak(self, msg):
        """
        Helper function to print to the terminal and support printing progress bars
        """

        sys.stdout.write(msg)
        sys.stdout.flush()

