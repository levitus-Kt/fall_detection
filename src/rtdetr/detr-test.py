import json
import os
import time

import numpy as np
import pandas as pd
from ultralytics import RTDETR

MODEL_WEIGHTS = "runs/fall_detect/rtdetr/weights/best.pt"
VIDEO_PATH = "test.mp4"
model_name = "rtdetr"
model = RTDETR(MODEL_WEIGHTS)
frames = 9137
metrics = []


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
    start = time.time()
    results = model.predict(
        source=VIDEO_PATH,
        conf=0.7,
        device=0,
        save=True,
        show=True,
    )
    total_inference_time = time.time() - start
    print(results)
    precision = results[0].metrics["precision"]
    recall = results[0].metrics["recall"]
    mAP = calculate_map_score(results[0].metrics["map"], precision, recall)
    avg_latency = total_inference_time / frames
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
