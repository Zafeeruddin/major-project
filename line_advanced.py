import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Define the entrance coordinates (example coordinates)
entrance_coords = (1041,374,1209,420,1149,724,1029,530)  # Update with your entrance coordinates
polygon = np.array([[807, 449], [979, 438], [1021, 276], [1214, 270], [1207, 688], [1135, 852], [912, 912], [791, 889]], dtype=np.int32)

# Load the YOLOv11 model
model = YOLO()

# Open the video file
video_path = "detect.mp4"
cap = cv2.VideoCapture(video_path)

# Save Video
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Store the track history and current positions
track_history = defaultdict(lambda: {'prev_coords': None, 'current_coords': None})
LEFT = 0
RIGHT = 0
FrameCount = 0
passedPeople = []

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    FrameCount += 1
    if FrameCount % (fps * 5) == 0:
        passedPeople.clear()

    if success:
        # Run YOLOv11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Process each detected box
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id != 0:  # Only process humans (assuming class_id 0 is human)
                continue

            x, y, w, h = box
            cx, cy = x + w / 2, y + h / 2  # Center of the box
            isInside = cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False)

            if isInside >= 0:
                # Point is inside the ROI
                if track_id in track_history:
                    # Update the existing track
                    track_info = track_history[track_id]
                    track_info['prev_coords'] = track_info['current_coords']
                    track_info['current_coords'] = (cx, cy)
                else:
                    # Add new track
                    track_history[track_id] = {'prev_coords': None, 'current_coords': (cx, cy)}
            else:
                # Point is outside the ROI
                if track_id in track_history:
                    del track_history[track_id]

        for track_id, coords in track_history.items():
            prev_coords = coords['prev_coords']
            current_coords = coords['current_coords']

            if prev_coords is not None:
                prevX, prevY = prev_coords
                currX, currY = current_coords
                if track_id in passedPeople:
                    continue

                if prevX >= currX:
                    LEFT += 1
                else:
                    RIGHT += 1
                passedPeople.append(track_id)

        # Annotate the frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (1450, 910), (1750, 1070), (0, 0, 0), -1)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"LEFT: {LEFT}", (1500, 950), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"RIGHT: {RIGHT}", (1500, 1000), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow("detected",frame)
        video_writer.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()