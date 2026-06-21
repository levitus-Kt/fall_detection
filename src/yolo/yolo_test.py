import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

VIDEO_SOURCE = "test.mp4"
model_name = "yolov8s"
MODEL_WEIGHTS = f"runs/fall_detect/{model_name}/weights/best.pt"
ANNOTATIONS = "datasets/Fall.v1i.coco/train/_annotations.coco.json"

CONF_THRESHOLD = 0.6  # Минимальный порог для фиксации класса падения


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


def run_pipeline():

    model = YOLO(MODEL_WEIGHTS)
    print(f"Размер модели: {os.path.getsize(MODEL_WEIGHTS) / (1024 * 1024):.2f} MB")

    output_path = (
        f"reports/{model_name}_result.mp4"  # Куда сохранить обработанное видео
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {VIDEO_SOURCE}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Настройка записи результата
    os.makedirs("reports", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    print("Обработка видео...")

    frame_idx = 0
    fall_counter = 0
    metrics = []
    total_inference_time = 0.0

    # Сбор данных для mAP
    all_preds_for_map = []  # Хранит conf
    # Переменные для матрицы ошибок
    tp, fp, fn, tn = 0, 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        start_time = time.time()

        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        end_time = time.time()
        total_inference_time += (end_time - start_time) * 1000

        detected_fall = False
        max_conf_this_frame = 0.0

        conf = float()
        # Результаты детекции
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()

            # В датасете класс 0 — это "Fall"
            if cls_id == 0 and conf >= CONF_THRESHOLD:
                detected_fall = True
                if conf > max_conf_this_frame:
                    max_conf_this_frame = conf

            # Отрисовка дефолтных боксов детектора
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = model.names[cls_id]

            # Отрисуем рамку объекта
            color = (0, 0, 255) if cls_id == 0 else (255, 120, 0)
            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1
            )
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Фильтр постоянства тренда
        if detected_fall:
            fall_counter += 1
        else:
            fall_counter = max(0, fall_counter - 1)

        # Решение системы

        if fall_counter == 0:
            system_status = "OK"
            ui_color = (0, 255, 0)  # Зеленый
        else:
            system_status = "Fall"
            ui_color = (0, 0, 255)  # Красный

        latency = (time.time() - start_time) * 1000
        fps_real = 1000 / latency if latency > 0 else 30.0

        # Визуализация панели телеметрии в верхнем левом углу кадра
        cv2.rectangle(frame, (10, 10), (110, 110), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{system_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            ui_color,
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {fps_real:.1f} FPS",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Запись и вывод кадра
        out_video.write(frame)
        cv2.imshow(f"{model_name} Detection", frame)

        # Рассчет глобального mAP
        if max_conf_this_frame > 0.0:
            all_preds_for_map.append(max_conf_this_frame)

        if detected_fall and conf >= CONF_THRESHOLD:
            tp += 1
        elif detected_fall and conf <= CONF_THRESHOLD:
            fn += 1  # Пропуск кадра с падением
        elif not detected_fall and conf <= CONF_THRESHOLD:
            tn += 1
        else:
            fp += 1  # Ложная тревога на кадре

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # Расчет финальных метрик
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mAP = calculate_map_score(all_preds_for_map, precision, recall)
    avg_latency = total_inference_time / frame_idx if frame_idx > 0 else 0
    model_size_mb = os.path.getsize(MODEL_WEIGHTS) / (1024 * 1024)

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
    save_final_reports(metrics)


def save_final_reports(metrics):
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

        # Стилизация через openpyxl
        worksheet = writer.sheets["Результаты метрик"]

        # Автоматическое выравнивание ширины столбцов под текст
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

    print(f"\nМетрики сохранены в 'reports/{model_name}_history.json'")
    print(f"Аналитический отчет выгружен в 'reports/{model_name}_fall_report.xlsx'")


if __name__ == "__main__":
    run_pipeline()
