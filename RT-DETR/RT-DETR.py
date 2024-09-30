import os
import cv2
from ultralytics import RTDETR

def preprocess_video(video_path, output_path, target_resolution=(854, 480)):
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    # Get original video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a VideoWriter object to save the resized video
    out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, target_resolution)
        
        # Write the resized frame to the output video
        out.write(resized_frame)
    
    # Release the video objects
    cap.release()
    out.release()
    print(f"Preprocessed video saved to {output_path}")

def process_videos(video_folder, model_path, target_resolution=(854, 480)):
    # Initialize the RTDETR model
    model = RTDETR(model_path)

    # Iterate over each file in the folder
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.mov', '.avi')):  # Adjust file extensions as needed
            video_path = os.path.join(video_folder, filename)
            preprocessed_video_path = os.path.join(video_folder, f"preprocessed_{filename}")
            
            try:
                # Preprocess the video (resize)
                preprocess_video(video_path, preprocessed_video_path, target_resolution)

                # Process the preprocessed video
                result = model(preprocessed_video_path, show=True)
                
                # Optionally, you can delete the preprocessed video if no longer needed
                os.remove(preprocessed_video_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Define paths
video_folder = 'videos'
model_path = 'rtdetr-x.pt'

# Process videos
process_videos(video_folder, model_path)
