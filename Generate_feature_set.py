# extracts features from a data set orginized by genre. saving to csv.

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

##modify for data set--the genras taken from dir name
GENRE_PATH = "Data/genres_original"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds max
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        features = np.hstack([mfcc, chroma, contrast, zcr, tempo])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_dataset(genre_folder):
    data = []
    genres = os.listdir(genre_folder)
    for genre in genres:
        genre_path = os.path.join(genre_folder, genre)
        if not os.path.isdir(genre_path):
            continue
        print(f"Processing genre: {genre}")
        for file in tqdm(os.listdir(genre_path)):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            if features is not None:
                data.append([genre] + features.tolist())
    columns = ["genre"] + [f"f{i}" for i in range(len(data[0])-1)]
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    df = build_dataset(GENRE_PATH)
    df.to_csv("genre_features.csv", index=False)
    print("Feature extraction complete! Saved to genre_features.csv")
