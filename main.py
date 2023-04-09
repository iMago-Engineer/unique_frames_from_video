import logging
import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# initialize Python Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# use Python logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class ImageSaveError(Exception):
    pass

def save_frame_as_image(output_dir: str, cap: cv2.VideoCapture, frame: cv2.VideoCapture):
    file_name = f'frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg'
    file_path = os.path.join(output_dir, file_name)
    success = cv2.imwrite(file_path, frame)
    if not success:
        print("Error writing file:", file_path)
        raise ImageSaveError

def compare_edge(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 1,
        output_dir: str = 'output',
    ):
    frame_ids = [0]

    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray_frame, 100, 200)
        prev_edges = cv2.Canny(prev_frame, 100, 200)
        abs_diff = cv2.absdiff(edges, prev_edges)
        
        # Calculate the mean of the absolute difference
        mean_diff = abs_diff.mean()
        print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES)},{mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            frame_ids.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # save_frame_as_image(output_dir, cap, frame)

        # Update the previous frame
        prev_frame = gray_frame

    frames_to_extract = []
    for i, left_frame_id in enumerate(frame_ids):
        if i + 1 == len(frame_ids):
            break

        right_frame_id = frame_ids[i+1]
        frames_to_extract.append((left_frame_id + right_frame_id) // 2)
    
    read_fps= cap.get(cv2.CAP_PROP_FPS) # 1秒あたりのフレーム数を取得
    for frame_id in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1 * read_fps) # 秒数と１秒あたりフレーム数をかけたフレームからスタート
        _, frame = cap.read()
        save_frame_as_image(output_dir, cap, frame)

def group_and_split(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 50,
        output_dir: str = 'output',
    ):
    frame_ids = [0]

    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        abs_diff = cv2.absdiff(gray_frame, prev_frame)
        
        # Calculate the mean of the absolute difference
        mean_diff = abs_diff.mean()
        print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES)},{mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            frame_ids.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # save_frame_as_image(output_dir, cap, frame)

        # Update the previous frame
        prev_frame = gray_frame

    frames_to_extract = []
    for i, left_frame_id in enumerate(frame_ids):
        if i + 1 == len(frame_ids):
            break

        right_frame_id = frame_ids[i+1]
        frames_to_extract.append((left_frame_id + right_frame_id) // 2)
    
    read_fps= cap.get(cv2.CAP_PROP_FPS) # 1秒あたりのフレーム数を取得
    for frame_id in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1 * read_fps) # 秒数と１秒あたりフレーム数をかけたフレームからスタート
        _, frame = cap.read()
        save_frame_as_image(output_dir, cap, frame)

def compare_prev_frame_and_save_with_abs_and_color(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 50,
        output_dir: str = 'output',
    ):
    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        abs_diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))
        
        # Calculate the mean of the absolute difference
        mean_diff = abs_diff.mean(axis=(0,1))
        print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES)},{mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff[0] > threshold or mean_diff[1] > threshold or mean_diff[2] > threshold:
            save_frame_as_image(output_dir, cap, frame)

        # Update the previous frame
        prev_frame = frame

def compare_prev_frame_and_save_with_abs(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 50,
        output_dir: str = 'output',
    ):
    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        abs_diff = cv2.absdiff(gray_frame, prev_frame)
        
        # Calculate the mean of the absolute difference
        mean_diff = abs_diff.mean()
        print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES)},{mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            save_frame_as_image(output_dir, cap, frame)

        # Update the previous frame
        prev_frame = gray_frame

def compare_selected_frame_and_save_with_abs(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 50,
        output_dir: str = 'output',
    ):
    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        abs_diff = cv2.absdiff(gray_frame, prev_frame)
        
        # Calculate the mean of the absolute difference
        mean_diff = abs_diff.mean()
        print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES)},{mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            save_frame_as_image(output_dir, cap, frame)
  
            # Update the previous frame
            prev_frame = gray_frame

def compare_selected_frame_and_save_with_ssim(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 0.9,
        output_dir: str = 'output',
    ):
    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ssim_index = ssim(gray_frame, prev_frame, data_range=prev_frame.max()-prev_frame.min())
        
        # -1 < ssim_index < 1
        # SSIM index > 0.9: The two images are very similar
        # 0.7 < SSIM index < 0.9: The two images are fairly similar, but there are some noticeable differences in brightness or contrast.
        # 0.5 < SSIM index < 0.7: The two images are somewhat similar, but there are significant differences in brightness or contrast.
        # SSIM index < 0.5: The two images are very different.
        if ssim_index < threshold:
            save_frame_as_image(output_dir, cap, frame)

            # Update the previous frame
            prev_frame = gray_frame

def main():
    # Load the video file
    cap = cv2.VideoCapture('sample.mp4')
    logger.info(cap)

    # Create the output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the first frame and output it unconditionally
    _, frame = cap.read()
    save_frame_as_image(output_dir, cap, frame)

    # Define the initial frame to compare against
    prev_output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loop through each frame of the video
    compare_edge(cap, prev_output_frame, output_dir=output_dir)

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
