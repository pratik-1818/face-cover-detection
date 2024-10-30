import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("/home/vmukti/Downloads/FaceCover/runs/detect/Mask_Detection/weights/best.pt")

# Define class names (ensure they match the order in your data.yaml file)
class_names = ['Mask', 'Person']

# Confidence threshold
confidence_threshold = 0.7



# Open the webcam
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("rtmp://media12.ambicam.com:1938/dvr7/torquedept")

# Check if the webcam is opened co
if not cap.isOpened():
    print("Error: Could not open stream.")
    print("Stream URL:", "rtmp://media12.ambicam.com:1938/dvr7/torquedept")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    # Perform object detection
    results = model(frame)

    # Iterate over each detection and draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # get confidences
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # get class ids

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # Check if confidence is greater than the threshold
            if confidence > confidence_threshold and class_id < len(class_names):
                # Get the class name
                class_name = class_names[class_id]

                # Determine color based on class name
                if class_name == "Mask":
                    color = (0, 0, 255)  # Red for "Mask_Person"
                else:
                    color = (0, 255, 0)  # Green for "Person"
                                                                 
                # Draw the bounding box
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_name}: {confidence:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame with detections
    cv2.imshow("YOLOv8 FaceCover Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
