import cv2
from ultralytics import solutions
import pytesseract
import re
from datetime import datetime
import psycopg2

def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = cv2.convertScaleAbs(gray, alpha=1.2, beta=60)
    _, binary = cv2.threshold(bright, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, h=30)
    return denoised

def extract_timestamp(frame, coordinates):
    """Extract timestamp from a specific region of a video frame using OCR."""
    x1, y1, x2, y2 = coordinates
    cropped_frame = frame[y1:y2, x1:x2]
    processed_frame = preprocess_image(cropped_frame)
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(processed_frame, config=custom_config)
    match = re.search(r'\d{2}-\d{2}-\d{4} \w{3} \d{2}:\d{2}:\d{2}', extracted_text)
    if match:
        return match.group(0)
    return extracted_text

def parse_timestamp(timestamp):
    """Parse the timestamp into its components."""
    try:
        dt = datetime.strptime(timestamp, "%m-%d-%Y %a %H:%M:%S")
        return {
            "month": dt.month,
            "day": dt.day,
            "year": dt.year,
            "weekday": dt.strftime("%A"),
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second
        }
    except ValueError:
        return None

def connect_db():
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(
        dbname='major_project',
        user='postgres',
        password='mysecretpassword',
        host='localhost',
        port='5432'
    )

def check_and_update_db(parsed_timestamp, in_count):
    """Check if the timestamp exists in the database and update or insert accordingly."""
    conn = connect_db()
    cur = conn.cursor()

    # Create table if it doesn't exist
    cur.execute('''
    CREATE TABLE IF NOT EXISTS footfall_data (
        datetime TIMESTAMP PRIMARY KEY,
        footfall INTEGER,
        is_holiday BOOLEAN,
        day_of_week INTEGER,
        is_weekend BOOLEAN
    );
    ''')

    # Convert parsed_timestamp to a datetime object
    timestamp_dt = datetime(
        year=parsed_timestamp['year'],
        month=parsed_timestamp['month'],
        day=parsed_timestamp['day'],
        hour=parsed_timestamp['hour'],
        minute=parsed_timestamp['minute'],
        second=parsed_timestamp['second']
    )

    # Map weekday to integer
    weekday_map = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }
    day_of_week = weekday_map[parsed_timestamp['weekday']]

    # Check if the timestamp exists
    cur.execute("SELECT * FROM footfall_data WHERE datetime = %s", (timestamp_dt,))
    result = cur.fetchone()

    if result:
        # Update footfall count
        cur.execute("UPDATE footfall_data SET footfall = footfall + %s WHERE datetime = %s", (in_count, timestamp_dt))
    else:
        # Insert new record
        is_holiday = parsed_timestamp['weekday'] in ['Friday', 'Saturday']
        is_weekend = parsed_timestamp['weekday'] in ['Saturday', 'Sunday']
        cur.execute('''
        INSERT INTO footfall_data (datetime, footfall, is_holiday, day_of_week, is_weekend)
        VALUES (%s, %s, %s, %s, %s)
        ''', (timestamp_dt, 0, is_holiday, day_of_week, is_weekend))
        
        # Log the creation of a new row
        print(f"Inserted new row: {timestamp_dt}, footfall: 0, is_holiday: {is_holiday}, day_of_week: {day_of_week}, is_weekend: {is_weekend}")

    conn.commit()
    cur.close()
    conn.close()

def count_objects_in_region(video_path, output_video_path, model_path, start_time_minutes=3):
    """Count cars in a specific region within a video starting after a given time."""
    prev_count = 0
    curr_count = 0
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    start_frame = int(start_time_minutes * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    region_points = [(538, 201), (552, 183), (565, 196), (555, 349), (533, 337), (538, 206)]
    car_class_index = 0
    
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path, classes=[car_class_index])

    timestamp_extracted = False
    timestamp = None

    timestamp_coordinates = (11, 28, 260, 66)

    # Variable to store the count of "In" objects
    in_count = 0

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        
        if not timestamp_extracted:
            timestamp = extract_timestamp(im0, timestamp_coordinates)
            if timestamp:
                parsed_timestamp = parse_timestamp(timestamp)
                if parsed_timestamp:
                    print(f"Extracted Timestamp: {parsed_timestamp}")
                    check_and_update_db(parsed_timestamp, in_count)
                timestamp_extracted = True

        results = counter(im0)
        
        if results.classwise_count:
            if curr_count != prev_count:
                prev_count = curr_count 
                curr_count = results.classwise_count['person']['IN']

            print("current_count", results.classwise_count['person']['IN'])

        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Total 'In' Count: {in_count}")

count_objects_in_region("detect.mp4", "output_video.avi", "yolo11n.pt", start_time_minutes=2)