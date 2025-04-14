import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_data(batch_size=32):
    print("Loading data...")

    # Paths to preprocessed data
    train_dir = 'Preprocessed_Videos/preprocessed_train'
    val_dir = 'Preprocessed_Videos/preprocessed_val'
    test_dir = 'Preprocessed_Videos/preprocessed_test'

    # Load CSV files to get labels
    train_csv = 'Videos/ASL_Citizen/splits/train.csv'
    val_csv = 'Videos/ASL_Citizen/splits/val.csv'
    test_csv = 'Videos/ASL_Citizen/splits/test.csv'

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Encode labels
    labels = train_df["Gloss"].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Batch generator function
    def load_dataset_in_batches(data_df, data_dir, batch_size):
        X_batch, y_batch = [], []
        for _, row in data_df.iterrows():
            video_file = row['Video file']
            label = row['Gloss']
            #npy_path = os.path.join(data_dir, f"{os.path.splitext(video_file)[0]}.npy")

            # Replace .mp4 with .npy to match your preprocessed data
            npy_filename = os.path.splitext(video_file)[0] + '.npy'
            npy_path = os.path.join(data_dir, npy_filename)

            print(f"Loading: {npy_filename} | Label: {label}")

            if os.path.exists(npy_path):
                frames = np.load(npy_path)  # Load preprocessed frames
                X_batch.append(frames)
                y_batch.append(label)

            # Yield the batch once we reach batch_size
            if len(X_batch) == batch_size:
                X_batch = np.array(X_batch)  # Shape: (batch_size, 30, 64, 64, 3)
                y_batch = label_encoder.transform(y_batch)  # Convert labels to numerical values
                y_batch = to_categorical(y_batch, num_classes=len(labels))  # One-hot encoding
                yield X_batch, y_batch
                X_batch, y_batch = [], []  # Reset batch for next samples

        # If there are remaining samples in the batch, yield them
        if X_batch:
            X_batch = np.array(X_batch)
            y_batch = label_encoder.transform(y_batch)
            y_batch = to_categorical(y_batch, num_classes=len(labels))
            yield X_batch, y_batch

    # Return generators for training, validation, and test data
    train_generator = load_dataset_in_batches(train_df, train_dir, batch_size)
    val_generator = load_dataset_in_batches(val_df, val_dir, batch_size)
    test_generator = load_dataset_in_batches(test_df, test_dir, batch_size)

    print("Data loading complete. Returning data generators.")

    return train_generator, val_generator, test_generator, len(labels)