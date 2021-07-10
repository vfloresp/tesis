import os
import pickle
import librosa
import numpy as np
import utils
import pandas as pd
import time


class Loader:
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = None
        try:
            signal = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=self.duration,
                mono=self.mono,
            )[0]
        except:
            print("error with " + file_path)
        return signal


class GenreLabeler:
    def __init__(self):
        self.tracks = utils.load("../fma/data/fma_metadata/tracks.csv")

    def find_genre(self, file_path):
        name = os.path.split(file_path)[1]
        index = int(name.replace('.mp3',''))
        genre = self.tracks.loc[index]["track"]["genre_top"]
        return genre


class Padder:
    """Adds padding to array"""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """Obtains the logaritmic spectogram of a signal"""

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def generate_spectrogram(self, signal):
        stft = librosa.stft(
            signal, n_fft=self.frame_size, hop_length=self.hop_length
        )[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, filepath, genre):
        save_path = self._generate_save_path(filepath,genre)
        print('Se guardo el archivo: {}'.format(save_path))
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        print('Se guardo el archivo de min max: {}'.format(save_path))
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path, genre):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, genre+'_'+file_name.replace('.mp3','') + ".npy")
        return save_path


class PreprocessingAudioPipeline:
    def __init__(self):
        self._loader = None
        self.padder = None
        self.generator = None
        self.normaliser = None
        self.saver = None
        self.genrelabel = None
        self.min_max_values = {}
        self._num_expected_samples = None
        self.genres_to_process = ['Rock', 'Experimental', 'Electronic', 'Hip-Hop', 'Folk', 'Pop']

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, subdirectories, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                genre_signal = self.genrelabeler.find_genre(file_path)
                if genre_signal in self.genres_to_process:
                    self._process_file(file_path)
                    print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if signal is None:
            print("señal no se pudo cargar")
        if signal is not None:
            if self._is_padding_neccesary(signal):
                signal = self._apply_padding(signal)
            feature = self.extractor.generate_spectrogram(signal)
            norm_feature = self.normaliser.normalize(feature)
            genre_signal = self.genrelabeler.find_genre(file_path)
            save_path = self.saver.save_feature(
                norm_feature, file_path, genre_signal
            )
            self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_neccesary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {"min": min_val, "max": max_val}
        print("stored min max :{} {}".format(min_val,max_val))


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 30.0
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "/home/vfloresp/Documents/tesis/tesis/src/spectrograms"
    MIN_MAX_VALUES_SAVE_DIR = "/home/vfloresp/Documents/tesis/tesis/src/min_max_values"
    FILES_DIR = "/home/vfloresp/Documents/tesis/tesis/fma/data/fma_medium"

    starttime = time.time()

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    genrelabler = GenreLabeler()

    preprocessing_pipeline = PreprocessingAudioPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.genrelabeler=  genrelabler

    preprocessing_pipeline.process(FILES_DIR)

    print('Tiempo de ejecución: {} s'.format((time.time()-starttime))/3600)