import os
import numpy as np
from va_autoencoder import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 1
EPOCHS = 20

genres_to_process = [
    "Rock",
    "Experimental",
    "Electronic",
    "Hip-Hop",
    "Folk",
    "Pop",
]


def load_fsdd(spectrogram_path):
    x_train = []
    for root, _, file_names in os.walk(spectrogram_path):
        for file_name in file_names:
            genre = file_name.split("_")[0]
            if genre == "Rock":
                file_path = os.path.join(root, file_name)  # (n_bins, n_frames)
                spectrogram = np.load(file_path)
                spectrogram_padded = np.empty(shape=(256, 432))
                for i, sample in enumerate(spectrogram):
                    spectrogram_padded[i] = np.append(sample, [0])
                x_train.append(spectrogram_padded)
                if len(x_train) >= 2000:
                    break
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000,256,64,1)
    print("cantidad de spectrogramas: {}".format(len(x_train)))
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 432, 1),
        conv_filters=(512, 256, 128, 32),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=128,
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    SPECTROGRAM_PATH = "/home/vfloresp/Documents/tesis/src/spectrograms_shorter"
    x_train = load_fsdd(SPECTROGRAM_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    autoencoder2 = VAE.load("model")
    autoencoder2.summary()
