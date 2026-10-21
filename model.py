import ultralytics
import os


def TrainModel(
        base_model: str = "yolov8n.pt",
        dataset_dir: str = "dataset",
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        training_device: str = "cpu"
    ) -> ultralytics.YOLO:
    print(f"BASE MODEL: {base_model}")
    model = ultralytics.YOLO(base_model)

    data_yaml_path = os.path.join(os.getcwd(), dataset_dir, "data.yaml")
    print(f"TRAINING OFF DATA AT: {data_yaml_path}")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=training_device,
    )

    print("TRAINING COMPLETE!")

    # ==========================================
    # 3. USE THE MODEL FOR INFERENCE
    # ==========================================
    # Ultralytics saves the best weights during training usually at runs/detect/train/weights/best.pt
    best_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")

    # Load your newly trained custom model
    trained_model = ultralytics.YOLO(best_model_path)

    return trained_model

def TestModel(
        model_to_test: ultralytics.YOLO,
        test_image_path = "path/to/your/test_image.jpg"
    ) -> int:
    if os.path.exists(test_image_path):
        inference_results = model_to_test.predict(
            source=test_image_path, 
            conf=0.25, 
            save=True
        )
        if inference_results:
            print(f"Inference complete! Results saved in runs/detect/predict")
            return 0
        else:
            return -1
    else:
        print(
            f"Test image not found at {test_image_path}. Skipping inference test script."
        )
        return -1

def main():
    trained_model = TrainModel()

if __name__ == "__main__":
    main()