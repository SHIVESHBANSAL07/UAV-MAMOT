from ultralytics import YOLO
import os

def train_model(
    data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    project="models",
    name="maritime_v1"
):
    """
    Train YOLOv8n on VisDrone dataset.
    Architecture: YOLOv8n — 130 layers, 3M parameters, 8.2 GFLOPs
    Framework: PyTorch + Ultralytics
    Input size: 640x640
    """
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=project,
        name=name,
        patience=10,
        save=True,
        plots=True,
        verbose=True
    )
    return results

if __name__ == "__main__":
    train_model(
        data_yaml="data/VisDrone/VisDrone.yaml",
        epochs=50,
        project="models",
        name="maritime_v1"
    )