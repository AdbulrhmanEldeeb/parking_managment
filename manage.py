import cv2
import torch
import numpy as np

def points_callback(event, x, y, flags, param):
    """
    Callback function to print coordinates on right double-click event.
    
    Args:
        event (int): OpenCV mouse event type
        x (int): X-coordinate of mouse pointer
        y (int): Y-coordinate of mouse pointer
        flags (int): Event flags
        param (object): Additional parameters (not used)
    """
    if event == cv2.EVENT_RBUTTONDBLCLK:
        print(f"Clicked coordinates: [x: {x}, y: {y}]")

# Create a named window for displaying the frame
cv2.namedWindow('FRAME')

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video writer configuration
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video writing
output_video = cv2.VideoWriter('videos/output.avi', fourcc, 20.0, (640, 480))

# Open input video
cap = cv2.VideoCapture('videos/parking.mp4')

# Region of Interest (ROI) for parking area
# Coordinates defining the polygon of the parking area
parking_area = [(26, 433), (9, 516), (389, 492), (786, 419), (720, 368)]

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for processing
    frame = cv2.resize(frame, (1020, 600))
    
    # Perform object detection
    results = model(frame)
    
    # List to store car points within ROI
    points = []
    
    # Iterate through detected objects
    for index, row in results.pandas().xyxy[0].iterrows():
        # Extract bounding box coordinates
        x1, y1 = int(row['xmin']), int(row['ymin'])
        x2, y2 = int(row['xmax']), int(row['ymax'])
        
        # Get object class
        object_class = row['name']
        
        # Calculate center of detection
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Check if detected object is a car
        if 'car' in object_class:
            # Check if car is inside the parking area
            roi_result = cv2.pointPolygonTest(np.array(parking_area, np.int32), (cx, cy), False)
            
            # If car is in ROI, draw bounding box and label
            if roi_result >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, str(object_class), (x1, y1), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                points.append([cx])
    
    # Count cars in parking area
    car_count = len(points)
    
    # Draw parking area boundary
    cv2.polylines(frame, [np.array(parking_area, np.int32)], True, (0, 255, 0), 2)
    
    # Display car count
    cv2.putText(frame, f'Number of cars in parking: {car_count}', 
                (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    # Show frame and write to output video
    cv2.imshow("FRAME", frame)
    output_video.write(frame)
    
    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()