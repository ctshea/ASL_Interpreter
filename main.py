import pandas as pd
import numpy as np
import ProcessData
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import mediapipe as mp
import LoadData
import TrainData



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    TrainData.train_model()
    # # Load the data generators
    # train_generator, val_generator, test_generator, num_classes = LoadData.load_data(batch_size=32)
    #
    # # Define the model
    # model = Sequential()
    # model.add(Input(shape=(30, 64, 64, 3)))  # 30 frames of 64x64 RGB
    # model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    #
    # model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # # Fit the model using the generator
    # model.fit(train_generator,
    #           validation_data=val_generator,
    #           epochs=10,
    #           steps_per_epoch=100,
    #           validation_steps=20)





    # # Initialize MediaPipe Hands
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    # mp_draw = mp.solutions.drawing_utils
    #
    # # Open the webcam
    # cap = cv2.VideoCapture(0)
    #
    # if not cap.isOpened():
    #     print("Error: Could not access the webcam.")
    #     exit()
    #
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Error: Could not read frame.")
    #         break
    #
    #     # Convert the frame to RGB (required by MediaPipe)
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     # Process the frame for hand landmarks
    #     results = hands.process(rgb_frame)
    #
    #     # Draw hand landmarks
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #
    #     # Display the frame
    #     cv2.imshow('Hand Detection', frame)
    #
    #     # Break the loop if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # Release resources
    # cap.release()
    # cv2.destroyAllWindows()
