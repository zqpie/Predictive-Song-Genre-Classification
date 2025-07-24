import joblib
from generate_feature_set import extract_features

def main():
    file_path = input("Enter the path to your .wav file: ")

    features = extract_features(file_path)

    if features is not None:
        model = joblib.load("genre_classifier.joblib")
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        print(f"Predicted genre: {prediction}")
    else:
        print("Could not extract features. Please check the audio file.")

if __name__ == "__main__":
    main()
