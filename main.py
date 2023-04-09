import logging
import os
import cv2
from skimage.metrics import structural_similarity as ssim

# initialize Python Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# use Python logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def main():
    # Load the video file
    cap = cv2.VideoCapture('sample.mp4')
    logger.info(cap)

    # Define the threshold value
    threshold = 0

    # Create the output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the first frame and output it unconditionally
    _, frame = cap.read()
    file_name = f'frame_{0}.jpg'
    file_path = os.path.join(output_dir, file_name)
    success = cv2.imwrite(file_path, frame)
    if not success:
        print("Error writing file:", file_path)

    # Define the initial frame to compare against
    prev_output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loop through each frame of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale and calculate the absolute difference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ssim_index = ssim(gray_frame, prev_output_frame, data_range=prev_output_frame.max()-prev_output_frame.min())
        # abs_diff = cv2.absdiff(gray_frame, prev_output_frame)
        
        # Calculate the mean of the absolute difference
        # mean_diff = abs_diff.mean()
        # logger.debug(f"mean_diff: {mean_diff}")

        # If the mean difference is greater than the threshold, output the frame
        # if mean_diff > threshold:
        if ssim_index < 0.7:
            file_path = 'output/frame_{}.jpg'.format(cap.get(cv2.CAP_PROP_POS_FRAMES))
            success = cv2.imwrite(file_path, frame)
            if not success:
                print(f"Error writing file: {file_path}")
  
        # Update the previous frame
        prev_output_frame = gray_frame
        
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
