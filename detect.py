import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO()  # Use a specific pre-trained model

# Video capture
video_path = "detect.mp4"
cap = cv2.VideoCapture(video_path)

# Save Video
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("human_detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Human detection counter
human_count = 0
crossing_count = 0

# Line coordinates
# 511,319,550,371
line_start = (511,319)
line_end = (550,371)

# Store previous positions of detected humans
previous_positions = {}

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break

    # Perform prediction (only for humans)
    results = model(frame, classes=[0])  # 0 is the class index for humans in COCO dataset
    # Reset human count for this frame
    human_count = 0

    # Process each detected human
    current_positions = {}
    for result in results:
        
        boxes = result.boxes

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence
            conf = float(box.conf[0])

            # Only draw if confidence is above threshold
            if conf > 0.5:
                human_count += 1

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Assign a unique ID to each detected human
                human_id = f"{center_x}_{center_y}"

                # Track the current position
                current_positions[human_id] = center_x
                print("Current positions",current_positions)
                # Check if the human crossed the line from left to right
                if human_id in previous_positions:
                    if previous_positions[human_id] < line_start[0] and center_x > line_end[0]:
                        crossing_count += 1

                # Choose color (BGR format)
                color = (100, 0, 0)  # Green

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Update previous positions
    previous_positions = current_positions

    # Draw the line
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # Add human count and crossing count to the frame
    cv2.putText(frame, f"Humans Detected: {human_count}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
    cv2.putText(frame, f"Crossings: {crossing_count}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

    # Display the frame
    cv2.imshow("Human Detection", frame)

    # Write to video
    video_writer.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Print total human count
print(f"Total humans detected in the video: {crossing_count}")