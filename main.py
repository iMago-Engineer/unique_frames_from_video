import logging
import os
import cv2

# initialize Python Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# use Python logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class ImageSaveError(Exception):
    pass

def save_frame_as_image(output_dir: str, frame: npt.NDArray, n: int) -> None:
    file_path = os.path.join(output_dir, f'frame_{n}.jpg')

    success = cv2.imwrite(file_path, frame)
    if not success:
        logger.error(f"Error writing file: {file_path}")
        raise ImageSaveError
    else:
        logger.info(f"Saved file: {file_path}")

def compare_edge(
        cap: cv2.VideoCapture,
        prev_frame: cv2.VideoCapture,
        threshold: float = 1,
        output_dir: str = 'output',
    ):
    frame_ids = [0]
    frames = []

    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        frames.append(frame)
        
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
    
    return_frames = []
    for i in frames_to_extract:
        return_frames.append(frames[int(i)])
    return return_frames

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
        save_frame_as_image(output_dir, frame, frame_id)

def main():
    threshold = 30
    # Load the video file
    cap = cv2.VideoCapture('sample.mp4')
    logger.info(cap)

    # Create the output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the first frame and output it unconditionally
    _, frame = cap.read()
    save_frame_as_image(output_dir, frame, 0)

    # Define the initial frame to compare against
    prev_output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loop through each frame of the video
    # edge 検知されて、抽出したframe
    edges = compare_edge(cap, prev_output_frame, output_dir=output_dir)

    # Delete similar images
    i = 0
    transition_frames = []

    # Loop through each frame of the video
    for frame in edges:
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ssim_index = ssim(gray_frame, prev_output_frame, data_range=prev_output_frame.max()-prev_output_frame.min())
        abs_diff = cv2.absdiff(gray_frame, prev_output_frame)
        
        # Calculate the mean of the absolute difference
        mean_diff = abs_diff.mean()
        # logger.debug(f"mean_diff: {mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            transition_frames.append(i)
            prev_output_frame = gray_frame

        # Update the previous frame
        i += 1

    print(transition_frames)
    # Save image between transition frames
    for i in range(len(transition_frames)-1):
        frame = edges[int((transition_frames[i] + transition_frames[i+1])/2)]
        save_frame_as_image(output_dir, frame, i+1)

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
