import os
import librosa
import math
import json

DATASET_PATH = (
    "/home/vflores/Documents/ITAM/tesis/tesis/tutoriales/Data/genres_original"
)
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(
    dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5
):
    # dictionary to store data
    data = {"mapping": [], "mfcc": [], "labels": []}

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        samples_per_segment / hop_length
    )  # 1.2 -> 2

    # loop trhough all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure that were not at root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/")  # genre/blues => ['genere','blues]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # processs files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # processs segments extracting mfcc and storing data
                for s in range(num_segments):
                    start = samples_per_segment * s
                    finish = start + samples_per_segment

                    # store mfcc for segment if it has the expected length
                    mfcc = librosa.feature.mfcc(
                        signal[start:finish],
                        sr,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )

                    mfcc = mfcc.T
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=1)
