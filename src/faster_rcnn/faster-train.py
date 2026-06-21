import os

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((640, 640)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    train_dataset = CocoDetection(
        root="datasets/Fall.v1i.coco/train",
        annFile="datasets/Fall.v1i.coco/train/_annotations.coco.json",
        transforms=train_transform,
    )

    val_dataset = CocoDetection(
        root="datasets/Fall.v1i.coco/valid",
        annFile="datasets/Fall.v1i.coco/valid/_annotations.coco.json",
        transforms=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    best_val_loss = float("inf")  # Переменная для сохранения ЛУЧШИХ весов

    num_classes = 4
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    model.to(device)

    # Замораживаем всю модель
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только голову детектора
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = True

    # Оптимизатор и планировщик
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Цикл обучения
    epochs = 50
    project_dir = "runs/fall_detect/faster_rcnn"
    os.makedirs(project_dir, exist_ok=True)

    print("Старт обучения...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_loss = 0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Эпоха {epoch + 1}/{epochs}",
            leave=True,
        )

        # Инициализируем скалер градиентов:
        scaler = torch.amp.GradScaler("cuda")

        for batch_idx, (images, targets) in progress_bar:
            images = list(image.to(device) for image in images)

            class_mapping = {
                0: 1,  # fall -> 1 (FALL)
                2: 1,  # down -> 1 (FALL)
                1: 2,  # 10-  -> 2 (Person)
                3: 2,  # person -> 2 (Person)
            }

            formatted_targets = []
            for target in targets:
                # Проверяем, есть ли объекты на картинке
                if target is None or len(target) == 0:
                    boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                    labels = torch.zeros((0,), dtype=torch.int64).to(device)
                else:
                    # Извлекаем BBox из формата COCO
                    raw_boxes = [ann["bbox"] for ann in target if "bbox" in ann]

                    if len(raw_boxes) == 0:
                        boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                        labels = torch.zeros((0,), dtype=torch.int64).to(device)
                    else:
                        converted_boxes = []
                        valid_labels = []
                        # Перебираем аннотации по одной, чтобы синхронизировать боксы и их классы
                        for ann in target:
                            if "bbox" in ann and "category_id" in ann:
                                xmin, ymin, w, h = ann["bbox"]
                                xmax = xmin + w
                                ymax = ymin + h

                                # Валидация геометрии для CUDA
                                if xmax <= xmin:
                                    xmax = xmin + 1.0
                                if ymax <= ymin:
                                    ymax = ymin + 1.0

                                raw_id = int(ann["category_id"])

                                if raw_id in class_mapping:
                                    converted_boxes.append([xmin, ymin, xmax, ymax])
                                    valid_labels.append(class_mapping[raw_id])

                        if len(converted_boxes) == 0:
                            boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                            labels = torch.zeros((0,), dtype=torch.int64).to(device)
                        else:
                            boxes = torch.tensor(
                                converted_boxes, dtype=torch.float32
                            ).to(device)
                            labels = torch.tensor(valid_labels, dtype=torch.int64).to(
                                device
                            )

                formatted_targets.append({"boxes": boxes, "labels": labels})

            # Защита от пустого батча на видеокарте (деление на ноль в лоссе)
            if sum(len(t["boxes"]) for t in formatted_targets) == 0:
                continue

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, formatted_targets)
                losses = sum(loss for loss in loss_dict.values())

            # Обратный проход (Backward pass)
            optimizer.zero_grad()

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(
            f"Эпоха [{epoch + 1}/{epochs}] — Ошибка (Loss): {epoch_loss / len(train_loader):.4f}"
        )

        avg_train_loss = train_loss / len(train_loader)

        val_loss = 0

        print(f"Запуск валидации для эпохи {epoch + 1}...")
        with torch.no_grad():
            for images, targets in val_loader:
                if images is None:
                    continue
                images = [img.to(device) for img in images]
                formatted_targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images, formatted_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\n=== ИТОГ ЭПОХИ [{epoch + 1}/{epochs}] ===")
        print(f" Train Loss: {avg_train_loss:.4f}")
        print(f" Val Loss:   {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(project_dir, "best.pth"))
            print(" Найдена лучшая модель! Веса сохранены в best.pth")

        if epoch == len(range(epochs)) - 1:
            torch.save(model.state_dict(), os.path.join(project_dir, "last.pth"))
        else:
            torch.save(
                model.state_dict(), os.path.join(project_dir, f"epoch_{epoch + 1}.pth")
            )

    print("Обучение завершено!")


if __name__ == "__main__":
    main()
