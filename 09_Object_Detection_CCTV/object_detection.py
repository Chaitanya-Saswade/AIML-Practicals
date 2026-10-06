import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")
print("Press 'q' to stop.")

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects
    results = model(frame)

    # Draw detected objects
    annotated_frame = results[0].plot()

    # Display webcam
    cv2.imshow(
        "Webcam Object Detection",
        annotated_frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()

print("Object detection completed.")