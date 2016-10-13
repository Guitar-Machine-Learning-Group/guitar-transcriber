from fextract import FeatureExtractor


class GuitarTranscriber(object):
    """
    Trains and tests ML model for guitar transcription
    """


    def __init__(self, audio_path, label_path, window_size, hop_size, sampling_rate):
        """
        PARAMETERS:
            audio_path (string): path to audio files
            label_path (string): path to midi files
            window_size (int): window sample size (samples)
            hop_size (int): window hop size (samples)
            sampling_rate (int): audio sampling rate (Hz)
        """

        # extract features for all songs in the dataset
        fe = FeatureExtractor(audio_path, label_path, window_size, hop_size, sampling_rate)
        fe.extract_features_and_labels()

        """
        At this point you have a dataset (fe.dataset) containing several songs that
        can be looped over using the generator fe.dataset.songs()

        Each song has properties s.X containing your calculated feature matrix for the song
        and s.Y, a binary matrix containing the pitch (class) labels for each song.
        s.X is a matrix of floats of size [num_audio_windows, feature_dimensionality]
        s.Y is a matrix of binary digits of size [num_audio_windows, num_pitches=51]
        """

        for s in fe.dataset.songs():
            pass

if __name__ == "__main__":
    audio_path = '/media/gburlet/Beartracks/Frettable/data/guitar/acoustic_synthesized_ground_truth/audio_small'
    label_path = '/media/gburlet/Beartracks/Frettable/data/guitar/acoustic_synthesized_ground_truth/midi_small'
    window_size = 2048
    hop_size = 1024
    sampling_rate = 22050

    gt = GuitarTranscriber(audio_path, label_path, window_size, hop_size, sampling_rate)

