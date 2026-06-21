import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from tqdm import tqdm


def collate_fn(batch):
    images = []
    targets = []

    class_mapping = {
        0: 1,  # fall -> 1 (FALL)
        2: 1,  # down -> 1 (FALL)
        1: 2,  # 10-  -> 2 (Person)
        3: 2,  # person -> 2 (Person)
    }

    for img, orig_target in batch:
        if orig_target is None or len(orig_target) == 0:
            continue

        base_tensor = torchvision.transforms.functional.to_tensor(img)
        _, orig_h, orig_w = base_tensor.shape

        img_resized = torchvision.transforms.functional.resize(base_tensor, (300, 300))

        converted_boxes = []
        valid_labels = []

        for ann in orig_target:
            if "bbox" in ann and "category_id" in ann:
                xmin, ymin, w, h = ann["bbox"]
                xmax = xmin + w
                ymax = ymin + h

                # Масштабируем координаты под размер 300x300
                xmin = xmin * (300.0 / float(orig_w))
                xmax = xmax * (300.0 / float(orig_w))
                ymin = ymin * (300.0 / float(orig_h))
                ymax = ymax * (300.0 / float(orig_h))

                if xmax <= xmin:
                    xmax = xmin + 1.0
                if ymax <= ymin:
                    ymax = ymin + 1.0

                xmin = max(0.0, min(xmin, 299.0))
                ymin = max(0.0, min(ymin, 299.0))
                xmax = max(1.0, min(xmax, 300.0))
                ymax = max(1.0, min(ymax, 300.0))

                raw_id = int(ann["category_id"])
                if raw_id in class_mapping:
                    converted_boxes.append([xmin, ymin, xmax, ymax])
                    valid_labels.append(class_mapping[raw_id])

        if len(converted_boxes) > 0:
            images.append(img_resized)
            targets.append(
                {
                    "boxes": torch.tensor(converted_boxes, dtype=torch.float32),
                    "labels": torch.tensor(valid_labels, dtype=torch.int64),
                }
            )

    if len(images) == 0:
        return None, None

    return images, targets


def main():
    # Настройка устройства
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    TRAIN_IMG_DIR = "datasets/Fall.v1i.coco/train"
    TRAIN_ANN_FILE = "datasets/Fall.v1i.coco/train/_annotations.coco.json"
    VAL_IMG_DIR = "datasets/Fall.v1i.coco/valid"
    VAL_ANN_FILE = "datasets/Fall.v1i.coco/valid/_annotations.coco.json"

    project_dir = "runs/fall_detect/ssd"
    os.makedirs(project_dir, exist_ok=True)

    train_dataset = CocoDetection(root=TRAIN_IMG_DIR, annFile=TRAIN_ANN_FILE)
    val_dataset = CocoDetection(root=VAL_IMG_DIR, annFile=VAL_ANN_FILE)

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

    num_classes = 4
    print("Загрузка предобученной модели SSD300...")
    model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")

    in_channels = torchvision.models.detection._utils.retrieve_out_channels(
        model.backbone, (300, 300)
    )
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = torchvision.models.detection.ssd.SSDHead(
        in_channels, num_anchors, num_classes
    )

    model.to(device)

    # Оптимизатор и планировщик
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scaler = torch.amp.GradScaler("cuda")
    best_val_loss = float("inf")
    epochs = 50

    print("Старт обучения SSD300...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Эпоха {epoch + 1}/{epochs}",
        )

        for batch_idx, (images, targets) in progress_bar:
            if images is None:
                continue

            images = [img.to(device) for img in images]
            formatted_targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, formatted_targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss = losses.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        lr_scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        model.train()
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

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(project_dir, "best.pth"))
            print(" Найдена лучшая модель! Веса сохранены в best.pth")

        torch.save(model.state_dict(), os.path.join(project_dir, "last.pth"))

    print("Обучение SSD300 успешно завершено!")


if __name__ == "__main__":
    main()
