from ultralytics.models import YOLO


def main():

    model = YOLO("yolov8s.pt")

    model.train(
        data="yolo/data_config.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        workers=4,
        lr0=0.0001,
        plots=True,
        project="",
        name="yolov8s",
    )
    print("Модель успешно обучена")


if __name__ == "__main__":
    main()
