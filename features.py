import logging
import os
import shutil
from io import BytesIO
from zipfile import ZipFile

import cv2
import numpy.typing as npt
import streamlit as st

# initialize Python Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# use Python logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

@st.cache_data
def extract_frames_with_diff_edges(frames: list[npt.NDArray], threshold: float = 1) -> list[npt.NDArray]:
    frame_with_diff_edges_inds = [0]

    first_frame = frames[0]
    prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = cv2.Canny(prev_frame, 100, 200)

    # Find frames where changes in edges happen
    for frame_ind, frame in enumerate(frames, 1):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)

        # Calculate the mean of the absolute difference
        abs_diff = cv2.absdiff(edges, prev_edges)
        mean_diff = abs_diff.mean()

        if mean_diff > threshold:
            frame_with_diff_edges_inds.append(frame_ind)

        prev_edges = edges

    # Representative frames are between two consecutive frames with different edges
    representative_frame_inds = [0] # include the first frame
    for i, left_frame_ind in enumerate(frame_with_diff_edges_inds[:-1]):
        right_frame_ind = frame_with_diff_edges_inds[i+1]

        representative_frame_inds.append((left_frame_ind + right_frame_ind) // 2)

    logging.debug(f"e: ({len(representative_frame_inds)} {representative_frame_inds}")

    return [ frames[int(frame_i)] for frame_i in representative_frame_inds ]

@st.cache_data
def remove_similar_frames(frames: list[npt.NDArray], threshold: int = 15) -> list[npt.NDArray]:
    transition_frame_inds = []

    # Find frames where transition happens
    initial_frame = frames[0]
    prev_grey_transition_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    for frame_ind, frame in enumerate(frames, 1):
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        abs_diff = cv2.absdiff(gray_frame, prev_grey_transition_frame)
        mean_diff = abs_diff.mean()

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            transition_frame_inds.append(frame_ind)

        prev_grey_transition_frame = gray_frame

    logging.debug(f"t: {transition_frame_inds}")

    # Distinct frames are between two consecutive frames where transition happens
    distinct_frame_inds = [0]
    for i, left_frame_ind in enumerate(transition_frame_inds[:-1]):
        right_frame_ind = transition_frame_inds[i+1]

        # not really sure why we need `-1` here
        # without `-1` we will get a different output
        distinct_frame_inds.append((left_frame_ind + right_frame_ind) // 2 - 1)

    return [ frames[frame_i] for frame_i in distinct_frame_inds ]

def read_frames_from_video_file(video_file_path: str) -> list[npt.NDArray]:
    cap = cv2.VideoCapture(video_file_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    # clean up
    cap.release()
    os.remove(video_file_path)

    return frames

@st.cache_data
def zip_images(frames, dir: str):
    # Create a temporary directory to store the frames
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Save the frames as JPEG files in the temporary directory
    for i, frame in enumerate(frames):
        file_path = os.path.join(dir, f'frame_{i}.jpg')
        cv2.imwrite(file_path, frame)

    # Compress the files in the temporary directory as a ZIP archive
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for file_name in os.listdir(dir):
            file_path = os.path.join(dir, file_name)
            zip_file.write(file_path, file_name)

    shutil.rmtree(dir)

    return zip_buffer.getvalue()
