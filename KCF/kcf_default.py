import cv2

# Initialize video capture
cap = cv2.VideoCapture('P1000214.MOV')  # Replace with your video path
#cap = cv2.VideoCapture('P1000267.MOV') 

# Get frame dimensions (original size)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def select_bbox(frame):
    """
    Selects bounding box on a downscaled frame for better user experience.

    Args:
        frame: The original frame from the video.

    Returns:
        A tuple containing the bounding box coordinates (x, y, w, h) or None if cancelled.
    """
    # Reduce frame size for display while maintaining aspect ratio
    display_ratio = 0.25  # Adjust as needed to fit your screen (e.g., 0.5 for 50% reduction)
    new_width = int(width * display_ratio)
    new_height = int(height * display_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    bbox = cv2.selectROI("Select object to track (press 'c' to cancel)", resized_frame, False)

    # If selection was cancelled, check based on original coordinates
    if bbox[0] < 0:
        return None

    # Scale bounding box coordinates back to original frame size
    x, y, w, h = bbox
    scaled_bbox = (int(x / display_ratio), int(y / display_ratio),
                   int(w / display_ratio), int(h / display_ratio))
    return scaled_bbox


# Read first frame
ret, frame = cap.read()

# Select bounding box on the downscaled frame
while True:
    bbox = select_bbox(frame.copy())
    if bbox is not None:  # Check if selection was not cancelled
        break

    # User cancelled selection, handle it here (e.g., print message)
    print("Selection cancelled.")

# Create KCF tracker (assuming OpenCV version >= 4.5)
tracker = cv2.TrackerKCF_create()  # Or other tracker like cv2.TrackerGOTURN_create()

# Initialize tracker using original frame dimensions
tracker.init(frame, bbox)

# Variables for FPS calculation
total_frames = 0
average_fps = 0.0
start_time = cv2.getTickCount()

# Define a downscaled window size for display
display_width = int(width * 0.3)  # Reduce width by 50% (adjust as needed)
display_height = int(height * 0.3)  # Reduce height by 50% (adjust as needed)

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Update tracker on the original frame
    success, bbox = tracker.update(frame)

    # Draw bounding box on the original frame
    if success:
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Calculate FPS using cv2.getTickCount() and cv2.getTickFrequency()
    end_time = cv2.getTickCount()
    elapsed_time = (end_time - start_time) / cv2.getTickFrequency()

    # Update average FPS only if a significant time has passed
    if elapsed_time > 1.0:
        fps = total_frames / elapsed_time
        average_fps = (average_fps + fps) / 2.0  # Simple moving average for smoother display
        start_time = end_time
        total_frames = 0

    # Display average FPS on original frame
    cv2.putText(frame, f"Average FPS: {int(average_fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Resize the frame for display
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Display the resized frame
    cv2.imshow("Tracking", resized_frame)
    # Exit on 'q' key press or 'c' to cancel selection
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('c'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

