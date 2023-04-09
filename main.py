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

def extract_frames_with_diff_edge(frames: list, threshold: float = 1):
    frame_with_diff_edges_ids = [0]

    first_frame = frames[0]
    prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = cv2.Canny(prev_frame, 100, 200)

    # Find frames where changes in edges happen
    for frame_id, frame in enumerate(frames, 1):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)

        # Calculate the mean of the absolute difference
        abs_diff = cv2.absdiff(edges, prev_edges)
        mean_diff = abs_diff.mean()

        if mean_diff > threshold:
            frame_with_diff_edges_ids.append(frame_id)

        prev_edges = edges

    # Representative frames are between two consecutive frames with different edges
    representative_frame_ids = []
    for i, left_frame_id in enumerate(frame_with_diff_edges_ids[:-1]):
        right_frame_id = frame_with_diff_edges_ids[i+1]

        representative_frame_ids.append((left_frame_id + right_frame_id) // 2)

    logging.debug(f"e: ({len(representative_frame_ids)} {representative_frame_ids}")

    # TODO: want to include `first_frame` as the first element
    return [ frames[int(frame_i)] for frame_i in representative_frame_ids ]

def remove_similar_frames(
        frames: list,
        threshold: int = 30
):
    # the index of `frames` where changes are detected
    transition_frame_ids = []

    # detect where changes happen
    initial_frame = frames[0]
    prev_grey_transition_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    for frame_id, frame in enumerate(frames, 1):
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        abs_diff = cv2.absdiff(gray_frame, prev_grey_transition_frame)
        mean_diff = abs_diff.mean()

        # If the mean difference is greater than the threshold, output the frame
        if mean_diff > threshold:
            transition_frame_ids.append(frame_id)
            prev_grey_transition_frame = gray_frame

    print(f"t: {transition_frame_ids}")

    # find the middle frame between each transition
    distinct_frames = []
    for i, left_frame_id in enumerate(transition_frame_ids[:-1]):
        right_frame_id = transition_frame_ids[i+1]

        frame_id = (left_frame_id + right_frame_id) // 2 - 1

        distinct_frames.append(frames[frame_id])

    return distinct_frames

def main():
    # Load the video file
    cap = cv2.VideoCapture('sample.mp4')

    # Create the output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the first frame and output it unconditionally
    # _, first_frame = cap.read()
    # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # save_frame_as_image(output_dir, first_frame, 0)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    edges = extract_frames_with_diff_edge(frames)

    # edges.insert(0, first_frame)
    ff = remove_similar_frames(edges, threshold=30)
    for i, frame in enumerate(ff):
        save_frame_as_image(output_dir, frame, (i + 1) * 200)

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
