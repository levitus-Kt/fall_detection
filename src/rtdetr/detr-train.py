from ultralytics import RTDETR


def main():
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data="data_config.yaml",
        epochs=5,
        imgsz=480,
        batch=3,
        device=0,
        workers=0,
        lr0=0.0001,
        cache="ram",
        amp=True,
        val=False,
        plots=False,
        save=False,
        project="",
        name="rtdetr",
    )


if __name__ == "__main__":
    main()
