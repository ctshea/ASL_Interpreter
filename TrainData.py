import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import LoadData

def train_model():
    # Load data generators and number of classes
    train_generator, val_generator, test_generator, num_classes = LoadData.load_data(batch_size=32)

    # Define model
    model = Sequential([
        Input(shape=(30, 64, 64, 3)),
        Conv3D(32, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        BatchNormalization(),

        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        BatchNormalization(),

        Conv3D(128, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_generator,
              validation_data=val_generator,
              epochs=10,
              steps_per_epoch=100,
              validation_steps=20)

    # Save the trained model
    model.save("asl_model.h5")
    print("✅ Model training complete. Model saved as asl_model.h5.")