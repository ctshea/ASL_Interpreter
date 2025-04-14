# import pandas as pd
# import os
# import cv2
# import numpy as np
#
# train_csv = 'Videos/ASL_Citizen/splits/train.csv'
# val_csv = 'Videos/ASL_Citizen/splits/val.csv'
# test_csv = 'Videos/ASL_Citizen/splits/test.csv'
#
# train_df = pd.read_csv(train_csv)
# val_df = pd.read_csv(val_csv)
# test_df = pd.read_csv(test_csv)
#
# video_dir = 'Videos/ASL_Citizen/videos'
# output_dir_train = 'Preprocessed_Videos/preprocessed_train'
# output_dir_val = 'Preprocessed_Videos/preprocessed_val'
# output_dir_test = 'Preprocessed_Videos/preprocessed_test'
# i=1
#
#
# def extract_frames(video_path, num_frames=30, resize_shape=(64, 64)):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_interval = max(total_frames // num_frames, 1)
#
#     for i in range(num_frames):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, resize_shape)
#         frames.append(frame)
#
#     cap.release()
#     if len(frames) < num_frames:
#         # If fewer frames, pad with zeros
#         frames.extend([np.zeros_like(frames[0])] * (num_frames - len(frames)))
#     return np.array(frames)
#
#
# def process_videos(data_frame, video_dir, output_dir, num_frames=30, resize_shape=(64, 64)):
#     os.makedirs(output_dir, exist_ok=True)
#     for _, row in data_frame.iterrows():
#         video_file = row['Video file']
#         video_path = os.path.join(video_dir, video_file)
#         output_file = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.npy")
#
#         if not os.path.exists(output_file):  # Skip if already processed
#             frames = extract_frames(video_path, num_frames, resize_shape)
#             np.save(output_file, frames)
#
#
#
#
# ProcessData.process_videos(train_df, video_dir, output_dir_train, i)
# ProcessData.process_videos(val_df, video_dir, output_dir_val, i)
# ProcessData.process_videos(test_df, video_dir, output_dir_test, i)