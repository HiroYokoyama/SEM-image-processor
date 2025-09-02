#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Image Processor (with progress save/resume)
github.com/HiroYokoyama/particle-image-processor
保存: particle-ip.py
依存: PyQt5, opencv-python (or opencv-python-headless), numpy, pandas, scikit-image (一部method)
"""

import sys
import os
import glob
import cv2
import numpy as np
import pandas as pd
import csv
import json
from datetime import datetime

try:
    from skimage.filters import threshold_li, threshold_triangle, threshold_yen, threshold_isodata
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QComboBox, QSlider, QSpinBox, QProgressBar, QMessageBox, QGroupBox, QFormLayout, QLineEdit,
    QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt

# ------------------ 画像処理関数 ------------------

def process_image_get_features(img_path, threshold_method='otsu', manual_threshold=128,
                               adaptive_blocksize=11, adaptive_C=2):
    """
    戻り値:
      overlay_img (BGR numpy), particle_features (list)
    features = [perimeter, area, aspect_ratio, solidity, circularity, hu1..hu7]
    粒子ごとに赤い小さめの番号を表示する（重心に描画）。
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = img.shape
    img_crop = img[:int(0.9*h), :].copy()
    img_blur = cv2.GaussianBlur(img_crop, (3,3), 0)

    # 二値化
    if threshold_method == 'otsu':
        _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'manual':
        _, thresh = cv2.threshold(img_blur, int(manual_threshold), 255, cv2.THRESH_BINARY)
    elif threshold_method == 'li':
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-imageが必要です: pip install scikit-image")
        li_thresh = threshold_li(img_blur)
        _, thresh = cv2.threshold(img_blur, li_thresh, 255, cv2.THRESH_BINARY)
    elif threshold_method == 'triangle':
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-imageが必要です: pip install scikit-image")
        tri_thresh = threshold_triangle(img_blur)
        _, thresh = cv2.threshold(img_blur, tri_thresh, 255, cv2.THRESH_BINARY)
    elif threshold_method == 'yen':
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-imageが必要です: pip install scikit-image")
        yen_thresh = threshold_yen(img_blur)
        _, thresh = cv2.threshold(img_blur, yen_thresh, 255, cv2.THRESH_BINARY)
    elif threshold_method == 'isodata':
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-imageが必要です: pip install scikit-image")
        iso_thresh = threshold_isodata(img_blur)
        _, thresh = cv2.threshold(img_blur, iso_thresh, 255, cv2.THRESH_BINARY)
    elif threshold_method == 'adaptive':
        bs = int(adaptive_blocksize)
        if bs % 2 == 0:
            bs = bs + 1
        if bs < 3:
            bs = 3
        thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, bs, int(adaptive_C))
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV のバージョン差に対応
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    particle_features = []
    overlay_img = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(overlay_img)

    for i, cnt in enumerate(contours):
        color = (int(np.random.randint(0, 256)),
                 int(np.random.randint(0, 256)),
                 int(np.random.randint(0, 256)))

        # マスクに塗りつぶし
        cv2.drawContours(mask, [cnt], -1, color, thickness=-1)

        # 粒子番号（重心に描画） — 赤, 小さめ
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # fallback: bounding rect center
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c // 2
            cy = y + h_c // 2

        cv2.putText(mask, str(i + 1), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 特徴量計算
        area = cv2.contourArea(cnt)
        if area == 0:
            continue
        perimeter = cv2.arcLength(cnt, True)
        x, y, w_c, h_c = cv2.boundingRect(cnt)
        aspect_ratio = float(w_c) / h_c if h_c != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()
        features = [perimeter, area, aspect_ratio, solidity, circularity] + hu_moments.tolist()
        particle_features.append(features)

    # 半透明合成（alpha=0.5）
    overlay_img = cv2.addWeighted(overlay_img, 1.0, mask, 0.5, 0)

    return overlay_img, particle_features


# ------------------ OpenCV->QPixmap ------------------

def cv2_to_qpixmap(cv_bgr):
    if cv_bgr is None:
        return QPixmap()
    if len(cv_bgr.shape) == 2:
        height, width = cv_bgr.shape
        bytes_per_line = width
        qimg = QImage(cv_bgr.data, width, height, bytes_per_line, QImage.Format_Indexed8)
        return QPixmap.fromImage(qimg)
    rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ------------------ GUI ------------------

class PIPGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Image Processor")
        self.resize(1100, 700)

        # 状態
        self.image_list = []
        self.current_index = -1
        self.ng_list = []
        self.reprocess_ng_list = []
        self.current_mode = 'initial'
        self.output_overlay_folder = ''
        self.output_individual_csv_folder = ''
        self.summary_csv_path = ''
        self.processing_log_path = ''
        self.processed_files = []   # [{'filename': '...', 'result': 'OK'/'NG'}, ...]

        # ログヘッダ (面積はログに含めない)
        self.log_headers = [
            'timestamp','filename','mode','result','num_contours',
            'threshold_method','manual_threshold','adaptive_blocksize','adaptive_C',
            'overlay_path','individual_csv_path','summary_updated','error_message'
        ]

        self._build_ui()
        # 初回でパラメータ有効/無効の状態を更新
        self.update_param_fields()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # 左: 画像表示 + 大きめOK/NGボタン下部に配置
        left_v = QVBoxLayout()
        imgs_h = QHBoxLayout()

        self.label_original = QLabel("Original")
        self.label_original.setFixedSize(500, 500)
        self.label_original.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.label_overlay = QLabel("Overlay")
        self.label_overlay.setFixedSize(500, 500)
        self.label_overlay.setStyleSheet("background-color: #222; border: 1px solid #444;")

        imgs_h.addWidget(self.label_original)
        imgs_h.addWidget(self.label_overlay)
        left_v.addLayout(imgs_h)

        # OK / NG 大きめボタン（画像下）
        buttons_under = QHBoxLayout()
        buttons_under.setSpacing(40)
        buttons_under.setContentsMargins(10,10,10,10)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.setFixedHeight(68)
        self.btn_ok.setStyleSheet("font-size:20px; padding:10px; background-color: #4CAF50; color: white;")
        self.btn_ok.clicked.connect(self.mark_ok_and_next)
        self.btn_ok.setEnabled(False)

        self.btn_ng = QPushButton("NG")
        self.btn_ng.setFixedHeight(68)
        self.btn_ng.setStyleSheet("font-size:20px; padding:10px; background-color: #F44336; color: white;")
        self.btn_ng.clicked.connect(self.mark_ng_and_next)
        self.btn_ng.setEnabled(False)

        buttons_under.addStretch()
        buttons_under.addWidget(self.btn_ok)
        buttons_under.addWidget(self.btn_ng)
        buttons_under.addStretch()
        left_v.addLayout(buttons_under)

        # 進捗
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_bar = QProgressBar()
        left_v.addWidget(self.progress_label)
        left_v.addWidget(self.progress_bar)

        main_layout.addLayout(left_v, stretch=3)

        # 右: 設定 / 操作
        right_v = QVBoxLayout()

        folder_box = QGroupBox("Folders / Output")
        folder_layout = QFormLayout()
        self.le_image_folder = QLineEdit()
        btn_img_folder = QPushButton("Choose Image Folder")
        btn_img_folder.clicked.connect(self.choose_image_folder)
        folder_layout.addRow(btn_img_folder, self.le_image_folder)

        self.le_output_folder = QLineEdit()
        btn_output_folder = QPushButton("Choose Output Folder")
        btn_output_folder.clicked.connect(self.choose_output_folder)
        folder_layout.addRow(btn_output_folder, self.le_output_folder)

        folder_box.setLayout(folder_layout)
        right_v.addWidget(folder_box)

        thresh_box = QGroupBox("Threshold Settings")
        thresh_layout = QFormLayout()
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(['otsu','manual','li','triangle','yen','isodata','adaptive'])
        # method変更でパラメータUIを更新
        self.cmb_method.currentIndexChanged.connect(self.update_param_fields)
        thresh_layout.addRow("Method:", self.cmb_method)

        # manual slider
        self.slider_manual = QSlider(Qt.Horizontal)
        self.slider_manual.setRange(0,255)
        self.slider_manual.setValue(128)
        self.lbl_manual = QLabel("Manual: 128")
        self.slider_manual.valueChanged.connect(lambda v: self.lbl_manual.setText(f"Manual: {v}"))
        thresh_layout.addRow(self.lbl_manual, self.slider_manual)

        # adaptive controls
        self.spin_adapt_bs = QSpinBox()
        self.spin_adapt_bs.setRange(3,99)
        self.spin_adapt_bs.setValue(11)
        self.spin_adapt_bs.setSingleStep(2)
        self.spin_adapt_C = QSpinBox()
        self.spin_adapt_C.setRange(-50,50)
        self.spin_adapt_C.setValue(2)
        adapt_h = QHBoxLayout()
        adapt_h.addWidget(QLabel("BlockSize:"))
        adapt_h.addWidget(self.spin_adapt_bs)
        adapt_h.addWidget(QLabel("C:"))
        adapt_h.addWidget(self.spin_adapt_C)
        thresh_layout.addRow(adapt_h)

        thresh_box.setLayout(thresh_layout)
        right_v.addWidget(thresh_box)

        # 実行ボタン
        btns_v = QHBoxLayout()
        self.btn_start_initial = QPushButton("Start Initial Pass")
        self.btn_start_initial.clicked.connect(self.start_initial_pass)
        btns_v.addWidget(self.btn_start_initial)

        self.btn_start_reprocess_ng = QPushButton("Start Reprocess NG (batch)")
        self.btn_start_reprocess_ng.clicked.connect(self.start_reprocess_ng)
        self.btn_start_reprocess_ng.setEnabled(False)
        btns_v.addWidget(self.btn_start_reprocess_ng)

        right_v.addLayout(btns_v)

        # navigation
        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        self.btn_next = QPushButton("Next (treat as NG)")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        step_h = QHBoxLayout()
        step_h.addWidget(self.btn_prev)
        step_h.addWidget(self.btn_next)
        right_v.addLayout(step_h)

        self.lbl_status = QLabel("")
        right_v.addWidget(self.lbl_status)

        right_v.addStretch()
        main_layout.addLayout(right_v, stretch=1)

        # ショートカット
        sc_ok = QShortcut(QKeySequence('O'), self)
        sc_ok.setContext(Qt.ApplicationShortcut)
        sc_ok.activated.connect(lambda: self._shortcut_trigger('ok'))
        sc_ng = QShortcut(QKeySequence('N'), self)
        sc_ng.setContext(Qt.ApplicationShortcut)
        sc_ng.activated.connect(lambda: self._shortcut_trigger('ng'))

    # ---------------- progress JSON helpers ----------------
    def get_progress_path(self):
        """出力フォルダに保存する progress.json のパスを返す。出力フォルダ未指定なら None を返す。"""
        base = self.le_output_folder.text().strip()
        if not base:
            return None
        return os.path.join(base, "progress.json")

    def save_progress(self):
        """現在の進捗を progress.json に保存する。OK/NG 毎に呼ぶ。"""
        ppath = self.get_progress_path()
        if not ppath:
            return
        data = {
            "mode": self.current_mode,
            "current_index": self.current_index,
            "processed_files": list(self.processed_files),
            "input_folder": self.le_image_folder.text().strip(),
            "output_folder": self.le_output_folder.text().strip(),
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(ppath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Save Progress error", f"Failed to save progress:\n{e}")

    def load_progress(self):
        """progress.json を読み込んで dict を返す。無ければ None を返す。"""
        ppath = self.get_progress_path()
        if not ppath or not os.path.exists(ppath):
            return None
        try:
            with open(ppath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            QMessageBox.warning(self, "Load Progress error", f"Failed to load progress:\n{e}")
            return None

    def clear_progress(self):
        """progress.json を削除し、内部の processed_files をクリア（主に新規開始時に使用）。"""
        ppath = self.get_progress_path()
        if ppath and os.path.exists(ppath):
            try:
                os.remove(ppath)
            except Exception as e:
                QMessageBox.warning(self, "Clear Progress error", f"Failed to remove progress file:\n{e}")
        self.processed_files = []

    # ショートカット処理
    def _shortcut_trigger(self, kind):
        if kind == 'ok' and self.btn_ok.isEnabled():
            self.mark_ok_and_next()
        elif kind == 'ng' and self.btn_ng.isEnabled():
            self.mark_ng_and_next()

    # フォルダ選択
    def choose_image_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose image folder", os.getcwd())
        if d:
            self.le_image_folder.setText(d)

    def choose_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", os.getcwd())
        if d:
            self.le_output_folder.setText(d)

    # ---------------- パラメータUI更新 ----------------
    def update_param_fields(self):
        """Method に応じて manual slider / adaptive controls を有効化 or グレーアウト"""
        method = self.cmb_method.currentText()
        if method == 'manual':
            # manual 有効、adaptive 無効
            self.slider_manual.setEnabled(True)
            self.lbl_manual.setEnabled(True)
            self.spin_adapt_bs.setEnabled(False)
            self.spin_adapt_C.setEnabled(False)
        elif method == 'adaptive':
            # adaptive 有効、manual 無効
            self.slider_manual.setEnabled(False)
            self.lbl_manual.setEnabled(False)
            self.spin_adapt_bs.setEnabled(True)
            self.spin_adapt_C.setEnabled(True)
        else:
            # その他は両方無効
            self.slider_manual.setEnabled(False)
            self.lbl_manual.setEnabled(False)
            self.spin_adapt_bs.setEnabled(False)
            self.spin_adapt_C.setEnabled(False)

    # ---------------- ログヘルパー ----------------
    def _ensure_log_initialized(self):
        if not self.processing_log_path:
            return
        if not os.path.exists(self.processing_log_path):
            try:
                with open(self.processing_log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.log_headers)
            except Exception as e:
                QMessageBox.warning(self, "Log init error", f"Cannot initialize log file:\n{e}")

    def append_processing_log(self, entry_dict):
        if not self.processing_log_path:
            return
        self._ensure_log_initialized()
        row = [entry_dict.get(h, '') for h in self.log_headers]
        try:
            with open(self.processing_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            QMessageBox.warning(self, "Log write error", f"Cannot write to log file:\n{e}")

    # ---------------- 画像一覧取得 / 出力準備 ----------------
    def _gather_image_list(self):
        folder = self.le_image_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            return []
        exts = ('*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp')
        imgs = []
        for e in exts:
            imgs.extend(glob.glob(os.path.join(folder, e)))
        imgs.sort()
        return imgs

    def _prepare_output_folders(self):
        base = self.le_output_folder.text().strip()
        if not base:
            raise ValueError("Please set output folder")
        overlay = os.path.join(base, "overlay_images")
        individual = os.path.join(base, "individual_csv")
        os.makedirs(overlay, exist_ok=True)
        os.makedirs(individual, exist_ok=True)
        self.output_overlay_folder = overlay
        self.output_individual_csv_folder = individual
        self.summary_csv_path = os.path.join(base, "summary_features.csv")
        self.processing_log_path = os.path.join(base, "processing_log.csv")
        self._ensure_log_initialized()

    # ---------------- メイン処理フロー ----------------
    def start_initial_pass(self):
        imgs = self._gather_image_list()
        if not imgs:
            QMessageBox.warning(self, "No images", "No images found in the selected folder.")
            return

        base_out = self.le_output_folder.text().strip()
        if not base_out:
            QMessageBox.warning(self, "No output folder", "Please set output folder before starting.")
            return

        # progress.json の存在チェックと再開/新規/キャンセルのダイアログ
        ppath = os.path.join(base_out, "progress.json")
        if os.path.exists(ppath):
            ret = QMessageBox.question(
                self,
                "Resume or Start New?",
                "A previous progress file was found. Do you want to resume the previous run?\n\n"
                "Yes: Resume previous progress\nNo: Start a new run (this will discard the saved progress)\nCancel: Abort",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if ret == QMessageBox.Cancel:
                return
            elif ret == QMessageBox.No:
                # 新規開始：保存を削除して続行
                try:
                    os.remove(ppath)
                except Exception:
                    pass
                self.processed_files = []
            else:
                # Resume: 読み込んで現在状態に反映
                try:
                    with open(ppath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    saved_mode = data.get("mode", "initial")
                    saved_processed = data.get("processed_files", [])
                    saved_input = data.get("input_folder", "")
                    # 警告: 入力フォルダが異なる場合は通知する（ファイル名でマッチする）
                    if saved_input and saved_input != self.le_image_folder.text().strip():
                        QMessageBox.information(self, "Warning", "Saved progress input folder differs from current selection. We'll try to match files by filename.")
                    self.current_mode = saved_mode
                    self.processed_files = saved_processed
                    # 最新の画像リストで再構築
                    imgs = self._gather_image_list()
                    self.image_list = imgs
                    # processed_files の最後のファイル名を探して次の index から再開
                    last_done_idx = -1
                    if self.processed_files:
                        last_filename = self.processed_files[-1].get("filename")
                        for idx, p in enumerate(self.image_list):
                            if os.path.basename(p) == last_filename:
                                last_done_idx = idx
                                break
                    self.current_index = last_done_idx + 1
                except Exception as e:
                    QMessageBox.warning(self, "Resume error", f"Failed to resume progress file, starting fresh.\n{e}")
                    self.processed_files = []

        # 出力フォルダ準備
        try:
            self._prepare_output_folders()
        except Exception as e:
            QMessageBox.critical(self, "Output folder error", str(e))
            return

        # 初期化（resumeがセットしていなければ）
        if not getattr(self, "image_list", None):
            self.image_list = imgs
            self.current_index = 0
            self.ng_list = []
            self.current_mode = 'initial'
        else:
            # image_list は resume により既にセットされている可能性あり
            if not self.image_list:
                self.image_list = imgs
                self.current_index = 0

        self.progress_bar.setMaximum(len(self.image_list))
        self.progress_bar.setValue(self.current_index)
        self.progress_label.setText(f"Progress: {self.current_index}/{len(self.image_list)}")

        self.btn_ok.setEnabled(True)
        self.btn_ng.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_start_initial.setEnabled(False)
        self.btn_start_reprocess_ng.setEnabled(False)

        # 保存しておく（再開情報）
        self.save_progress()

        self.process_and_show_current()

    def process_and_show_current(self):
        if self.current_mode == 'initial':
            if not (0 <= self.current_index < len(self.image_list)):
                self.finish_initial_pass()
                return
            path = self.image_list[self.current_index]
            try:
                overlay, particle_features = process_image_get_features(
                    path,
                    threshold_method=self.cmb_method.currentText(),
                    manual_threshold=self.slider_manual.value(),
                    adaptive_blocksize=self.spin_adapt_bs.value(),
                    adaptive_C=self.spin_adapt_C.value()
                )
            except Exception as e:
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': os.path.basename(path),
                    'mode': 'initial',
                    'result': 'ERROR',
                    'num_contours': '',
                    'threshold_method': self.cmb_method.currentText(),
                    'manual_threshold': (self.slider_manual.value() if self.cmb_method.currentText() == 'manual' else 'N/A'),
                    'adaptive_blocksize': (self.spin_adapt_bs.value() if self.cmb_method.currentText() == 'adaptive' else 'N/A'),
                    'adaptive_C': (self.spin_adapt_C.value() if self.cmb_method.currentText() == 'adaptive' else 'N/A'),
                    'overlay_path':'',
                    'individual_csv_path':'',
                    'summary_updated': 'False',
                    'error_message': str(e)
                }
                self.append_processing_log(entry)
                QMessageBox.critical(self, "Processing error", f"Error processing {path}:\n{e}")
                self.ng_list.append(path)
                # mark as processed ERROR in processed_files and save
                self.processed_files.append({'filename': os.path.basename(path), 'result': 'ERROR'})
                self.save_progress()
                self.next_index()
                return

            orig_pix = cv2_to_qpixmap(cv2.imread(path))
            overlay_pix = cv2_to_qpixmap(overlay)
            self.label_original.setPixmap(orig_pix.scaled(self.label_original.size(), Qt.KeepAspectRatio))
            self.label_overlay.setPixmap(overlay_pix.scaled(self.label_overlay.size(), Qt.KeepAspectRatio))
            self._last_processing = {'path': path, 'overlay_bgr': overlay, 'particle_features': particle_features}
            self.lbl_status.setText(f"Image: {os.path.basename(path)}  Contours: {len(particle_features)}")
            self.progress_bar.setValue(self.current_index)
            self.progress_label.setText(f"Progress: {self.current_index}/{len(self.image_list)}")

        elif self.current_mode == 'reprocess':
            if not (0 <= self.current_index < len(self.image_list)):
                self.finish_reprocess_pass()
                return
            path = self.image_list[self.current_index]
            try:
                overlay, particle_features = process_image_get_features(
                    path,
                    threshold_method=self.cmb_method.currentText(),
                    manual_threshold=self.slider_manual.value(),
                    adaptive_blocksize=self.spin_adapt_bs.value(),
                    adaptive_C=self.spin_adapt_C.value()
                )
            except Exception as e:
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': os.path.basename(path),
                    'mode': 'reprocess',
                    'result': 'ERROR',
                    'num_contours': '',
                    'threshold_method': self.cmb_method.currentText(),
                    'manual_threshold': (self.slider_manual.value() if self.cmb_method.currentText() == 'manual' else 'N/A'),
                    'adaptive_blocksize': (self.spin_adapt_bs.value() if self.cmb_method.currentText() == 'adaptive' else 'N/A'),
                    'adaptive_C': (self.spin_adapt_C.value() if self.cmb_method.currentText() == 'adaptive' else 'N/A'),
                    'overlay_path':'',
                    'individual_csv_path':'',
                    'summary_updated': 'False',
                    'error_message': str(e)
                }
                self.append_processing_log(entry)
                QMessageBox.critical(self, "Processing error", f"Error processing {path}:\n{e}")
                self.reprocess_ng_list.append(path)
                # processed_files に ERROR として保存
                self.processed_files.append({'filename': os.path.basename(path), 'result': 'ERROR'})
                self.save_progress()
                self.next_index()
                return

            orig_pix = cv2_to_qpixmap(cv2.imread(path))
            overlay_pix = cv2_to_qpixmap(overlay)
            self.label_original.setPixmap(orig_pix.scaled(self.label_original.size(), Qt.KeepAspectRatio))
            self.label_overlay.setPixmap(overlay_pix.scaled(self.label_overlay.size(), Qt.KeepAspectRatio))
            self._last_processing = {'path': path, 'overlay_bgr': overlay, 'particle_features': particle_features}
            self.lbl_status.setText(f"[Reprocess] Image: {os.path.basename(path)}  Contours: {len(particle_features)}")
            self.progress_bar.setValue(self.current_index)
            self.progress_label.setText(f"Reprocess: {self.current_index}/{len(self.image_list)}")

    # 保存（OK）処理
    def mark_ok_and_next(self):
        if not self.btn_ok.isEnabled():
            return
        if not hasattr(self, '_last_processing'):
            return
        info = self._last_processing
        path = info['path']
        overlay = info['overlay_bgr']
        features = info['particle_features']

        base_name = os.path.splitext(os.path.basename(path))[0]
        cols_ind = ['perimeter','area','aspect_ratio','solidity','circularity'] + [f'hu{i+1}' for i in range(7)]
        if features:
            df_ind = pd.DataFrame(features, columns=cols_ind)
        else:
            df_ind = pd.DataFrame(columns=cols_ind)
        csv_path = os.path.join(self.output_individual_csv_folder, base_name + '.csv')
        try:
            df_ind.to_csv(csv_path, index=False)
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Failed to save individual CSV for {path}:\n{e}")

        overlay_path = os.path.join(self.output_overlay_folder, os.path.basename(path))

        try:
            cv2.imwrite(overlay_path, overlay)
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Failed to save overlay image for {path}:\n{e}")

        summary_cols = ['filename','perimeter','area','aspect_ratio','solidity','circularity'] + [f'hu{i+1}' for i in range(7)]
        if features:
            mean_features = np.mean(np.array(features), axis=0).tolist()
        else:
            mean_features = [np.nan]*12
        row = [os.path.basename(path)] + mean_features
        summary_updated = False
        try:
            if not os.path.exists(self.summary_csv_path):
                df = pd.DataFrame([row], columns=summary_cols)
                df.to_csv(self.summary_csv_path, index=False)
            else:
                df = pd.DataFrame([row], columns=summary_cols)
                df.to_csv(self.summary_csv_path, mode='a', header=False, index=False)
            summary_updated = True
        except Exception as e:
            QMessageBox.warning(self, "Summary save error", f"Failed to update summary CSV:\n{e}")

        method = self.cmb_method.currentText()
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(path),
            'mode': self.current_mode,
            'result': 'OK',
            'num_contours': len(features) if features is not None else 0,
            'threshold_method': method,
            'manual_threshold': (self.slider_manual.value() if method == 'manual' else 'N/A'),
            'adaptive_blocksize': (self.spin_adapt_bs.value() if method == 'adaptive' else 'N/A'),
            'adaptive_C': (self.spin_adapt_C.value() if method == 'adaptive' else 'N/A'),
            'overlay_path': overlay_path if os.path.exists(overlay_path) else '',
            'individual_csv_path': csv_path if os.path.exists(csv_path) else '',
            'summary_updated': str(summary_updated),
            'error_message': ''
        }
        self.append_processing_log(entry)

        # processed_files に追記して progress.json を保存
        self.processed_files.append({'filename': os.path.basename(path), 'result': 'OK'})
        self.save_progress()

        self.lbl_status.setText(f"Saved: {base_name}")
        self.next_index()

    # NG 取扱い
    def mark_ng_and_next(self):
        if not self.btn_ng.isEnabled():
            return
        if not hasattr(self, '_last_processing'):
            return
        path = self._last_processing['path']
        features = self._last_processing.get('particle_features', [])

        method = self.cmb_method.currentText()
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(path),
            'mode': self.current_mode,
            'result': 'NG',
            'num_contours': len(features) if features is not None else 0,
            'threshold_method': method,
            'manual_threshold': (self.slider_manual.value() if method == 'manual' else 'N/A'),
            'adaptive_blocksize': (self.spin_adapt_bs.value() if method == 'adaptive' else 'N/A'),
            'adaptive_C': (self.spin_adapt_C.value() if method == 'adaptive' else 'N/A'),
            'overlay_path': '',
            'individual_csv_path': '',
            'summary_updated': 'False',
            'error_message': ''
        }
        self.append_processing_log(entry)

        # processed_files に追記して progress.json を保存
        self.processed_files.append({'filename': os.path.basename(path), 'result': 'NG'})
        self.save_progress()

        if self.current_mode == 'initial':
            self.ng_list.append(path)
        else:
            self.reprocess_ng_list.append(path)
        self.lbl_status.setText(f"Marked NG: {os.path.basename(path)}")
        self.next_index()

    # 次インデックス処理
    def next_index(self):
        if self.current_mode == 'initial':
            self.current_index += 1
            if self.current_index >= len(self.image_list):
                self.finish_initial_pass()
                return
            self.process_and_show_current()
        else:
            self.current_index += 1
            if self.current_index >= len(self.image_list):
                self.finish_reprocess_pass()
                return
            self.process_and_show_current()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.process_and_show_current()

    def next_image(self):
        # 強制的に NG 扱いで次へ
        self.mark_ng_and_next()

    def finish_initial_pass(self):
        self.btn_ok.setEnabled(False)
        self.btn_ng.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_start_initial.setEnabled(True)
        if self.ng_list:
            self.btn_start_reprocess_ng.setEnabled(True)
            QMessageBox.information(self, "Initial Pass Complete", f"Initial pass complete.\nNG images: {len(self.ng_list)}.\nYou may reprocess them in batch using 'Start Reprocess NG (batch)'.")
        else:
            QMessageBox.information(self, "Initial Pass Complete", "Initial pass complete.\nNo NG images.")
        self.progress_bar.setValue(len(self.image_list))
        self.progress_label.setText(f"Progress: {len(self.image_list)}/{len(self.image_list)}")

        # 完了時には progress.json を削除して内部状態をクリア
        self.clear_progress()

    def start_reprocess_ng(self):
        if not self.ng_list:
            QMessageBox.information(self, "No NG images", "There are no NG images to reprocess.")
            return
        ret = QMessageBox.question(self, "Reprocess NG",
                                   f"{len(self.ng_list)} images will be reprocessed using the current threshold settings.\nProceed?",
                                   QMessageBox.Yes | QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        # reprocessではMethod選択はそのまま（UIのグレーアウトもそのまま）
        self.image_list = list(self.ng_list)
        self.current_index = 0
        self.current_mode = 'reprocess'
        self.reprocess_ng_list = []

        # processed_files はそのまま残しておく（履歴）
        # UI
        self.btn_ok.setEnabled(True)
        self.btn_ng.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_start_initial.setEnabled(False)
        self.btn_start_reprocess_ng.setEnabled(False)

        self.progress_bar.setMaximum(len(self.image_list))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Reprocess: 0/{len(self.image_list)}")

        # save state
        self.save_progress()

        self.process_and_show_current()

    def finish_reprocess_pass(self):
        self.btn_ok.setEnabled(False)
        self.btn_ng.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_start_initial.setEnabled(True)
        kept = [p for p in self.image_list if p not in self.reprocess_ng_list]
        left = len(self.reprocess_ng_list)
        QMessageBox.information(self, "Reprocess Complete", f"Reprocess complete.\nSaved (OK) from reprocess: {len(kept)}\nRemaining NG: {left}")
        self.current_mode = 'initial'
        self.ng_list = list(self.reprocess_ng_list)
        self.reprocess_ng_list = []
        if self.ng_list:
            self.btn_start_reprocess_ng.setEnabled(True)
        else:
            self.btn_start_reprocess_ng.setEnabled(False)

        # reprocess 完了後も processed_files は残る。必要なら clear_progress() を呼ぶ。

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit', 'Quit application?', QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # アプリ終了時、安全のため進捗を保存
            try:
                self.save_progress()
            except Exception:
                pass
            event.accept()
        else:
            event.ignore()


# ------------------ main ------------------

def main():
    app = QApplication(sys.argv)
    gui = PIPGui()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
