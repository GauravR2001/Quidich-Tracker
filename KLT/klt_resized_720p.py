import numpy as np
import cv2 as cv
import argparse
import os
import subprocess
import time

def resize_video(input_path, output_path, width=1280, height=720):
    # Construct the ffmpeg command
    command = [
        'ffmpeg', 
        '-i', input_path,
        '-vf', f'scale={width}:{height}',
        '-c:a', 'copy', 
        output_path
    ]
    
    # Execute the command
    subprocess.run(command, check=True)
    print(f"Resized video saved to '{output_path}'")

def track_features(video_path):
    # Check if the provided video file path exists
    if not os.path.isfile(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        return

    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Scaling factor for ROI selection and visualization
    scale_factor = 0.25

    # Take the first frame and allow ROI selection
    ret, old_frame = cap.read()
    if not ret:
        print('Error: Could not read the first frame from the video.')
        cap.release()
        return

    # Resize the frame for ROI selection
    frame_resized = cv.resize(old_frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Allow the user to select ROI on the resized frame
    roi = cv.selectROI("Select ROI", frame_resized, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Select ROI")

    if roi == (0, 0, 0, 0):
        print('ROI selection was cancelled.')
        cap.release()
        return

    # Scale ROI coordinates back to the original frame size
    (x_r, y_r, w_r, h_r) = roi
    x, y, w, h = int(x_r / scale_factor), int(y_r / scale_factor), int(w_r / scale_factor), int(h_r / scale_factor)

    # Convert the original frame to grayscale and define ROI
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    roi_gray = old_gray[y:y+h, x:x+w]
    p0 = cv.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

    # Adjust feature points to the original frame coordinates
    if p0 is not None:
        p0 += np.array([x, y])  # Adjust coordinates to the original frame

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    total_start_time = start_time  # Track the total start time

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed or end of video reached!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks on the original frame
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw only if the point is inside the ROI
                if x <= a <= x + w and y <= b <= y + h:
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Combine frame with mask
        img = cv.add(frame, mask)

        # Resize the image for display
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        img_resized = cv.resize(img, (new_w, new_h))

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time

        # Draw a background rectangle for the FPS text
        text_background = img_resized.copy()
        text_size, _ = cv.getTextSize(f'FPS: {fps:.2f}', cv.FONT_HERSHEY_SIMPLEX, 2, 3)
        text_w, text_h = text_size
        cv.rectangle(text_background, (10, 20 - text_h), (10 + text_w, 20 + 10), (0, 0, 0), thickness=-1)
        
        # Put text on the image
        cv.putText(text_background, f'FPS: {fps:.2f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv.LINE_AA)

        # Display the resized image
        cv.imshow('frame', text_background)

        k = cv.waitKey(30) & 0xff
        if k == 27:  # Exit on 'Esc' key
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Calculate and print average FPS
    total_elapsed_time = time.time() - total_start_time
    if total_elapsed_time > 0:
        average_fps = frame_count / total_elapsed_time
        print(f"Average FPS: {average_fps:.2f}")

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize video to 720p and perform feature tracking.')
    parser.add_argument('input', type=str, help='path to input video file')
    parser.add_argument('output', type=str, help='path to resized video file')
    args = parser.parse_args()

    # Resize the video
    resize_video(args.input, args.output)

    # Perform feature tracking on the resized video
    track_features(args.output)
