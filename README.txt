Project uses machine learning to classify audio recordings of format .wav into trained generes, based on features.
- Extracts audio features (MFCCs, chroma, etc.) using `librosa`
- Trains a Random Forest classifier to predict genres
- Supports prediction for new `.wav` files


Model trained using data set in the below format:

Data/genres_original/ then all the genres get folders. their name is used for classification. 
or just use this premade set:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

To use:
edit the path line on run generate_feature_set.py to the folder containing the genre folders. then run this file. 
it will generate a .csv feature set that can be used by predict new song.py. 
