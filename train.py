from ultralytics import YOLO

# Train a custom model
model = YOLO("yolov8l.pt")

model.train(
    data="/home/vmukti/Downloads/FaceCover/Dataset/data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=16, 
    name="Mask_Detection",
    device = 0
    )