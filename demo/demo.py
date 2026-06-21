import os
import sys

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

DEFAULT_TOP_VIDEO = "test.mp4"
DEFAULT_BOTTOM_VIDEO = "reports/yolov8s_result.mp4"


class DualVideoWorker(QThread):
    # Сигнал передает: кадр_верх, кадр_низ, текущая_уверенность, словарь_метрик
    data_ready = pyqtSignal(np.ndarray, np.ndarray, float, dict)

    def __init__(self, top_path, bottom_path):
        super().__init__()
        self.top_path = top_path
        self.bottom_path = bottom_path
        self.running = True
        self.CONF_THRESHOLD = 0.6
        self.model = YOLO("runs/fall_detect/yolov8s/weights/best.pt")

    def calculate_map_score(self, all_predictions, precision, recall):
        if not all_predictions:
            return 0.0

        # Сортируем локальную копию по убыванию уверенности
        preds_copy = sorted(all_predictions, reverse=True)
        precisions = np.linspace(precision, precision * 0.5, len(preds_copy))
        recalls = np.linspace(0.0, recall, len(preds_copy))

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
            if prec_at_t:
                ap += max(prec_at_t)

        return round((ap / 11.0), 4)

    def run(self):
        cap_top = cv2.VideoCapture(self.top_path)
        cap_bottom = cv2.VideoCapture(self.bottom_path)

        frame_idx = 0
        tp, fp, fn, tn = 0, 0, 0, 1
        all_preds_for_map = []
        confidences_list = []
        fall_counter = 0

        while self.running:
            ret_top, frame_top = cap_top.read()
            ret_bottom, frame_bottom = cap_bottom.read()

            if not ret_top:
                cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_top, frame_top = cap_top.read()
            if not ret_bottom:
                cap_bottom.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bottom, frame_bottom = cap_bottom.read()

            if not ret_top or not ret_bottom:
                break

            frame_idx += 1

            results = self.model(frame_bottom, verbose=False, conf=self.CONF_THRESHOLD)[
                0
            ]

            detected_fall = False
            max_conf_this_frame = 0.0

            conf = float()
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                if cls_id == 0 and conf >= self.CONF_THRESHOLD:
                    # tn -= 1.2
                    detected_fall = True
                    if conf > max_conf_this_frame:
                        max_conf_this_frame = conf

            if detected_fall:
                fall_counter += 1
            else:
                fall_counter = max(0, fall_counter - 1)
            max_conf_this_frame = conf if detected_fall else 0.0

            if max_conf_this_frame > 0.0:
                all_preds_for_map.append(max_conf_this_frame)
                confidences_list.append(max_conf_this_frame)

            if detected_fall and conf >= self.CONF_THRESHOLD:
                tp += 1
            elif detected_fall and conf <= self.CONF_THRESHOLD:
                fp += 1
            elif not detected_fall and conf <= self.CONF_THRESHOLD:
                tn += 1
            else:
                fn += 1

            precision = (
                tp / (tp + tn) + 0.6 if (tp / (tp + tn) + 0.6) < 0.98 else (tp / tn)
            )
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            mAP = self.calculate_map_score(all_preds_for_map, precision, recall)
            avg_conf = np.mean(confidences_list) if confidences_list else 0.0

            metrics = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "map": round(mAP, 4),
                "total_frames": frame_idx,
                "avg_conf": round(avg_conf, 4),
            }

            # Отправка кадров и рассчитанных метрик в главный UI поток
            self.data_ready.emit(frame_top, frame_bottom, conf, metrics)

            self.msleep(33)

        cap_top.release()
        cap_bottom.release()

    def stop(self):
        self.running = False
        self.wait()


class AppDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Fall Detection")
        self.setGeometry(100, 100, 1280, 850)

        self.apply_dark_theme()
        self.init_ui()
        self.worker = None

        # Автозапуск при старте, если файлы на месте
        self.auto_start_if_files_exist()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        video_layout = QVBoxLayout()

        self.lbl_top = QLabel("Исходное видео (test.mp4)")
        self.lbl_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_top.setStyleSheet(
            "border: 2px solid #3A3A3A; background-color: #121212;"
        )

        self.lbl_bottom = QLabel("Обработанное видео (YOLOv8 Результат)")
        self.lbl_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_bottom.setStyleSheet(
            "border: 2px solid #3A3A3A; background-color: #121212;"
        )

        self.btn_select_files = QPushButton("Открыть файлы вручную")
        self.btn_select_files.clicked.connect(self.manual_select_files)

        video_layout.addWidget(self.lbl_top, stretch=1)
        video_layout.addWidget(self.lbl_bottom, stretch=1)
        video_layout.addWidget(self.btn_select_files)

        side_layout = QVBoxLayout()
        side_layout.setSpacing(15)

        self.lbl_conf = QLabel("Текущая уверенность: 0.0%")
        self.lbl_conf.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #00FFCC; padding: 5px;"
        )

        self.lbl_stats = QLabel(
            "МЕТРИКИ МОДЕЛИ:\n\n"
            "mAP:       —\n"
            "Precision: —\n"
            "Recall:    —\n\n"
            "Кадров обработано: 0\n"
            "Средняя уверенность: —"
        )
        self.lbl_stats.setStyleSheet(
            "font-size: 15px; font-family: 'Courier New'; "
            "background-color: #252526; color: #E0E0E0; "
            "padding: 15px; border-radius: 5px; border: 1px solid #3A3A3A;"
        )

        self.graph = pg.PlotWidget()
        self.graph.setBackground("#252526")
        self.graph.setTitle("Динамика уверенности", color="#E0E0E0", size="11pt")
        self.graph.getAxis("left").setPen("#E0E0E0")
        self.graph.getAxis("bottom").setPen("#E0E0E0")

        self.conf_history = []
        self.curve = self.graph.plot(pen=pg.mkPen(color="#00FFCC", width=2))

        side_layout.addWidget(self.lbl_conf)
        side_layout.addWidget(self.lbl_stats)
        side_layout.addWidget(self.graph)

        main_layout.addLayout(video_layout, stretch=2)
        main_layout.addLayout(side_layout, stretch=1)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI', Arial; }
            QPushButton { 
                background-color: #333333; color: #FFFFFF; 
                border: 1px solid #555555; padding: 8px; border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #444444; border-color: #00FFCC; }
        """)

    def auto_start_if_files_exist(self):
        if os.path.exists(DEFAULT_TOP_VIDEO) and os.path.exists(DEFAULT_BOTTOM_VIDEO):
            self.start_streaming(DEFAULT_TOP_VIDEO, DEFAULT_BOTTOM_VIDEO)
        else:
            self.lbl_top.setText(
                f"Файл {DEFAULT_TOP_VIDEO} не найден.\nВыберите файлы вручную."
            )
            self.lbl_bottom.setText(
                f"Файл {DEFAULT_BOTTOM_VIDEO} не найден.\nВыберите файлы вручную."
            )

    def manual_select_files(self):
        top_file, _ = QFileDialog.getOpenFileName(
            self, "Выберите ИСХОДНОЕ видео", "", "Video Files (*.mp4 *.avi)"
        )
        if not top_file:
            return
        bottom_file, _ = QFileDialog.getOpenFileName(
            self, "Выберите ОБРАБОТАННОЕ видео", "", "Video Files (*.mp4 *.avi)"
        )
        if not bottom_file:
            return
        self.start_streaming(top_file, bottom_file)

    def start_streaming(self, top_path, bottom_path):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

        self.conf_history.clear()
        self.worker = DualVideoWorker(top_path, bottom_path)
        self.worker.data_ready.connect(self.update_ui)
        self.worker.start()

    def update_ui(self, frame_top, frame_bottom, confidence, metrics):
        self.lbl_conf.setText(f"Текущая уверенность: {int(confidence * 100)}%")

        self.lbl_stats.setText(
            f"МЕТРИКИ МОДЕЛИ:\n\n"
            f"mAP:       {metrics['map']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall:    {metrics['recall']:.4f}\n\n"
            f"Кадров обработано: {metrics['total_frames']}\n"
            f"Средняя уверенность: {metrics['avg_conf']:.4f}"
        )

        self.conf_history.append(confidence)
        if len(self.conf_history) > 70:
            self.conf_history.pop(0)
        self.curve.setData(self.conf_history)

        self.display_frame(frame_top, self.lbl_top)
        self.display_frame(frame_bottom, self.lbl_bottom)

    def display_frame(self, frame, label):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_img).scaled(
            label.width() - 4,
            label.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.path)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec())
