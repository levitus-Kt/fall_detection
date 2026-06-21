import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision


def calculate_map_score(all_predictions, precision, recall):
    if not all_predictions:
        return 0.0

    # Сортируем по убыванию уверенности
    all_predictions.sort(reverse=True)

    precisions = []
    recalls = []

    for _ in all_predictions:
        precisions.append(precision)
        recalls.append(recall)

    # Расчет площади под PR-кривой методом 11 точек
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_t:
            ap += max(prec_at_t)

    return round((ap / 11.0), 4)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 4
    CONFIDENCE_THRESHOLD = 0.6

    video_path = "test.mp4"
    weights_path = "runs/fall_detect/retinanet/best.pth"
    output_path = "reports/retina_result.mp4"
    model_name = "retinanet"

    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")

    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = (
        torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео по пути {video_path}")
        return

    # Получаем параметры исходного видео
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    # Настройка записи итогового видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    print("Обработка видео...")

    # Переменные для подсчета FPS инференса
    prev_time = 0
    frame_idx = 0
    metrics = []
    total_inference_time = 0.0

    # Сбор данных для mAP
    all_preds_for_map = []
    # Переменные для матрицы ошибок
    tp, fp, fn, tn = 0, 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Конец видео

        frame_idx += 1

        detected_fall = False
        max_conf_this_frame = 0.0

        score = float()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (640, 640))

        # Переводим в тензор
        img_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device).unsqueeze(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            predictions = model(img_tensor)[0]

        end_time = time.time()
        total_inference_time += (end_time - start_time) * 1000

        # Вытаскиваем координаты, классы и уверенность (scores)
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        torch_boxes = torch.from_numpy(boxes).to(device)
        torch_scores = torch.from_numpy(scores).to(device)

        keep_indices = torchvision.ops.nms(torch_boxes, torch_scores, iou_threshold=0.3)

        boxes = boxes[keep_indices.cpu().numpy()]
        labels = labels[keep_indices.cpu().numpy()]
        scores = scores[keep_indices.cpu().numpy()]

        # Отрисовка результатов
        for box, label, score in zip(boxes, labels, scores):
            if score > CONFIDENCE_THRESHOLD:
                # Возвращаем координаты рамок от 640x640 к исходному разрешению видео
                xmin = int(box[0] * (orig_w / 640.0))
                ymin = int(box[1] * (orig_h / 640.0))
                xmax = int(box[2] * (orig_w / 640.0))
                ymax = int(box[3] * (orig_h / 640.0))

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(orig_w, xmax), min(orig_h, ymax)

                if score > max_conf_this_frame:
                    max_conf_this_frame = score

                if label == 2:
                    class_name = "Person"
                    color = (0, 255, 0)  # Зеленый
                elif label == 1:
                    class_name = "FALL"
                    color = (0, 0, 255)  # Красный
                    detected_fall = True
                else:
                    class_name = f"Class {label}"
                    color = (255, 0, 0)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                text = f"{class_name}: {score:.2f}"

                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    frame, (xmin, ymin - text_h - 10), (xmin + text_w, ymin), color, -1
                )
                cv2.putText(
                    frame,
                    text,
                    (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # Расчет и вывод текущего FPS
        current_time = time.time()
        inference_fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        cv2.putText(
            frame,
            f"FPS: {inference_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        out.write(frame)

        cv2.imshow("Faster R-CNN Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Рассчет глобального mAP
        if max_conf_this_frame > 0.0:
            all_preds_for_map.append(max_conf_this_frame)

        if detected_fall and score >= CONFIDENCE_THRESHOLD:
            tp += 1
        elif detected_fall and score <= CONFIDENCE_THRESHOLD:
            fn += 1  # Пропуск кадра с падением
        elif not detected_fall and score <= CONFIDENCE_THRESHOLD:
            tn += 1
        else:
            fp += 1  # Ложная тревога на кадре

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео сохранено в {output_path}")

    # Расчет финальных метрик
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mAP = calculate_map_score(all_preds_for_map, precision, recall)
    avg_latency = total_inference_time / frame_idx if frame_idx > 0 else 0
    model_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    metrics.append(
        {
            "Архитектура": model_name,
            "Размер модели (Мб)": round(model_size_mb, 2),
            "Скорость инференса (мс)": round(avg_latency, 2),
            "mAP": round(mAP, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Ложные тревоги (кадры)": fp,
            "Пропущенные кадры": fn,
        }
    )

    # Сохранение агрегированных отчетов
    save_final_reports(metrics, model_name)


def save_final_reports(metrics, model_name):
    # Запись в JSON
    with open(f"reports/{model_name}_history.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # Выгрузка в Excel
    df = pd.DataFrame(metrics)
    if df.empty:
        return

    with pd.ExcelWriter(
        f"reports/{model_name}_fall_report.xlsx", engine="openpyxl"
    ) as writer:
        df.to_excel(writer, sheet_name="Результаты метрик", index=False)

        worksheet = writer.sheets["Результаты метрик"]

        # Автоматическое выравнивание ширины столбцов под текст
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

    print(f"\nМетрики сохранены в 'reports/{model_name}_history.json'")
    print(f"Аналитический отчет выгружен в 'reports/{model_name}_fall_report.xlsx'")


if __name__ == "__main__":
    main()
