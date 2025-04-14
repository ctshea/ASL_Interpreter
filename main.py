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
