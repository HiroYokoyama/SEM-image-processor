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
except ImportError:
    SKIMAGE_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QComboBox, QSlider, QSpinBox, QProgressBar, QMessageBox, QGroupBox, QFormLayout, QLineEdit,
    QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt

# ------------------ Image Processing Functions ------------------

def process_image_get_features(img_path, threshold_method='otsu', manual_threshold=128,
                               adaptive_blocksize=11, adaptive_C=2):
    """
    Returns:
      overlay_img (numpy array in BGR format), particle_features (list of features)
    Features = [Perimeter, Area, Aspect Ratio, Solidity, Circularity, Hu Moments 1..7]
    Draws a small red number at the centroid of each particle (contour).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    h, w = img.shape
    img_crop = img[:int(0.9*h), :].copy()
    img_blur = cv2.GaussianBlur(img_crop, (3,3), 0)

    # Binarization Process
    thresh = None # Binarized image variable
    if threshold_method == 'otsu':
        _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'manual':
        _, thresh = cv2.threshold(img_blur, int(manual_threshold), 255, cv2.THRESH_BINARY)
    elif SKIMAGE_AVAILABLE and threshold_method in ['li', 'triangle', 'yen', 'isodata']:
        if threshold_method == 'li':
            thresh_val = threshold_li(img_blur)
        elif threshold_method == 'triangle':
            thresh_val = threshold_triangle(img_blur)
        elif threshold_method == 'yen':
            thresh_val = threshold_yen(img_blur)
        elif threshold_method == 'isodata':
            thresh_val = threshold_isodata(img_blur)
        _, thresh = cv2.threshold(img_blur, thresh_val, 255, cv2.THRESH_BINARY)
    elif threshold_method == 'adaptive':
        bs = int(adaptive_blocksize)
        # Ensure block size is odd and at least 3, as required by OpenCV.
        if bs % 2 == 0:
            bs += 1
        bs = max(3, bs)
        thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, bs, int(adaptive_C))
    else:
        if not SKIMAGE_AVAILABLE and threshold_method in ['li', 'triangle', 'yen', 'isodata']:
             raise RuntimeError("This method requires scikit-image: pip install scikit-image")
        raise ValueError(f"Unknown binarization method: {threshold_method}")

    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Handle OpenCV version differences
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    particle_features = []
    overlay_img = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(overlay_img)

    for i, cnt in enumerate(contours):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.drawContours(mask, [cnt], -1, color, thickness=-1)

        # Calculate the centroid of the contour
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else: # If centroid calculation fails, use the center of the bounding rectangle
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx, cy = x + w_c // 2, y + h_c // 2

        cv2.putText(mask, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Calculate features
        area = cv2.contourArea(cnt)
        if area == 0: continue
        perimeter = cv2.arcLength(cnt, True)
        x, y, w_c, h_c = cv2.boundingRect(cnt)
        aspect_ratio = float(w_c) / h_c if h_c != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
        features = [perimeter, area, aspect_ratio, solidity, circularity] + hu_moments.tolist()
        particle_features.append(features)

    # Blend the original image and the mask with transparency
    overlay_img = cv2.addWeighted(overlay_img, 1.0, mask, 0.5, 0)
    return overlay_img, particle_features

# ------------------ Convert OpenCV to QPixmap ------------------

def cv2_to_qpixmap(cv_img):
    """Converts an OpenCV image (numpy array) to a PyQt QPixmap."""
    if cv_img is None: return QPixmap()
    if len(cv_img.shape) == 2: # Grayscale image
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    else: # Color image
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(qimg)

# ------------------ GUI Class ------------------

class PIPGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Image Processor")
        self.resize(1100, 700)

        # State variables
        self.image_list = [] # The original, full list of images. This should not be modified after initial load.
        self.processing_queue = [] # The list of images for the current pass (can be the full list or ng_list).
        self.current_index = -1
        self.ng_list = []
        self.reprocess_ng_list = []
        self.current_mode = 'initial'
        self.output_overlay_folder = ''
        self.output_individual_csv_folder = ''
        self.summary_csv_path = ''
        self.processing_log_path = ''
        self.processed_files = []

        # Log file headers
        self.log_headers = [
            'timestamp', 'filename', 'mode', 'result', 'num_contours',
            'threshold_method', 'manual_threshold', 'adaptive_blocksize', 'adaptive_C',
            'overlay_path', 'individual_csv_path', 'summary_updated', 'error_message'
        ]

        self._build_ui()
        self.update_param_fields()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Left Panel: Image Display
        left_v = QVBoxLayout()
        imgs_h = QHBoxLayout()
        self.label_original = QLabel("Original Image")
        self.label_original.setFixedSize(500, 500)
        self.label_original.setStyleSheet("background-color: #222; border: 1px solid #444; color: #fff; qproperty-alignment: 'AlignCenter';")
        self.label_overlay = QLabel("Processed Image")
        self.label_overlay.setFixedSize(500, 500)
        self.label_overlay.setStyleSheet("background-color: #222; border: 1px solid #444; color: #fff; qproperty-alignment: 'AlignCenter';")
        imgs_h.addWidget(self.label_original)
        imgs_h.addWidget(self.label_overlay)
        left_v.addLayout(imgs_h)

        # OK / NG Buttons
        buttons_under = QHBoxLayout()
        self.btn_ok = QPushButton("OK (O)")
        self.btn_ok.setFixedHeight(68)
        self.btn_ok.setStyleSheet("font-size:20px; padding:10px; background-color: #4CAF50; color: white;")
        self.btn_ok.clicked.connect(self.mark_ok_and_next)
        self.btn_ok.setEnabled(False)
        self.btn_ng = QPushButton("NG (N)")
        self.btn_ng.setFixedHeight(68)
        self.btn_ng.setStyleSheet("font-size:20px; padding:10px; background-color: #F44336; color: white;")
        self.btn_ng.clicked.connect(self.mark_ng_and_next)
        self.btn_ng.setEnabled(False)
        buttons_under.addStretch()
        buttons_under.addWidget(self.btn_ok)
        buttons_under.addWidget(self.btn_ng)
        buttons_under.addStretch()
        left_v.addLayout(buttons_under)

        # Progress Bar
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_bar = QProgressBar()
        left_v.addWidget(self.progress_label)
        left_v.addWidget(self.progress_bar)
        main_layout.addLayout(left_v, stretch=3)

        # Right Panel: Controls
        right_v = QVBoxLayout()
        folder_box = QGroupBox("Folder Settings")
        folder_layout = QFormLayout()
        self.le_image_folder = QLineEdit()
        self.le_image_folder.textChanged.connect(self.update_start_button_state)
        self.btn_img_folder = QPushButton("Select Image Folder")
        self.btn_img_folder.clicked.connect(self.choose_image_folder)
        folder_layout.addRow(self.btn_img_folder, self.le_image_folder)
        self.le_output_folder = QLineEdit()
        self.le_output_folder.textChanged.connect(self.update_start_button_state)
        self.btn_output_folder = QPushButton("Select Output Folder")
        self.btn_output_folder.clicked.connect(self.choose_output_folder)
        folder_layout.addRow(self.btn_output_folder, self.le_output_folder)
        folder_box.setLayout(folder_layout)
        right_v.addWidget(folder_box)

        # Binarization Settings
        thresh_box = QGroupBox("Binarization Settings")
        thresh_layout = QFormLayout()
        self.cmb_method = QComboBox()
        # Modified list order: otsu first, then scikit-image, then manual and adaptive
        items = ['otsu']
        if SKIMAGE_AVAILABLE:
            items.extend(['li', 'triangle', 'yen', 'isodata'])
        items.extend(['manual', 'adaptive'])
        self.cmb_method.addItems(items)
        self.cmb_method.currentIndexChanged.connect(self.update_param_fields)
        thresh_layout.addRow("Method:", self.cmb_method)

        self.slider_manual = QSlider(Qt.Horizontal)
        self.slider_manual.setRange(0, 255); self.slider_manual.setValue(128)
        self.lbl_manual = QLabel("Manual Threshold: 128")
        self.slider_manual.valueChanged.connect(lambda v: self.lbl_manual.setText(f"Manual Threshold: {v}"))
        thresh_layout.addRow(self.lbl_manual, self.slider_manual)

        self.spin_adapt_bs = QSpinBox(); self.spin_adapt_bs.setRange(3, 99); self.spin_adapt_bs.setValue(11); self.spin_adapt_bs.setSingleStep(2)
        self.spin_adapt_C = QSpinBox(); self.spin_adapt_C.setRange(-50, 50); self.spin_adapt_C.setValue(2)
        adapt_h = QHBoxLayout(); adapt_h.addWidget(QLabel("Block Size:")); adapt_h.addWidget(self.spin_adapt_bs); adapt_h.addWidget(QLabel("C:")); adapt_h.addWidget(self.spin_adapt_C)
        thresh_layout.addRow(adapt_h)
        thresh_box.setLayout(thresh_layout)
        right_v.addWidget(thresh_box)

        # Start Button
        self.btn_start = QPushButton("Start / Resume")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("font-size: 16px; background-color: #2196F3; color: white;")
        self.btn_start.clicked.connect(self.start_or_resume_processing)
        right_v.addWidget(self.btn_start)

        # Navigation Buttons
        nav_h = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        self.btn_next = QPushButton("Next (as NG)")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        nav_h.addWidget(self.btn_prev)
        nav_h.addWidget(self.btn_next)
        right_v.addLayout(nav_h)

        # Status Label
        self.lbl_status = QLabel("Select folders and press 'Start'.")
        right_v.addWidget(self.lbl_status)
        right_v.addStretch()
        main_layout.addLayout(right_v, stretch=1)

        # Shortcut Keys
        QShortcut(QKeySequence('O'), self, lambda: self.btn_ok.click() if self.btn_ok.isEnabled() else None)
        QShortcut(QKeySequence('N'), self, lambda: self.btn_ng.click() if self.btn_ng.isEnabled() else None)
        
        # Update initial button state
        self.update_start_button_state()

    # ---------------- Progress Saving (JSON) Helper Functions ----------------
    def get_progress_path(self):
        """Returns the path to the progress file (progress.json)."""
        base = self.le_output_folder.text().strip()
        return os.path.join(base, "progress.json") if base else None

    def save_progress(self, completed=False):
        """Saves the current session state to a JSON file."""
        ppath = self.get_progress_path()
        if not ppath: return

        data = {
            "status": "Completed" if completed else "In Progress",
            "mode": self.current_mode,
            "image_list": self.image_list,  # Always save the original, full list
            "ng_list": self.ng_list,
            "reprocess_ng_list": self.reprocess_ng_list,
            "processed_files": list(self.processed_files),
            "input_folder": self.le_image_folder.text().strip(),
            "output_folder": self.le_output_folder.text().strip(),
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(ppath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Progress Save Error", f"Failed to save progress:\n{e}")

    def load_progress(self):
        """Loads the session state from a JSON file."""
        ppath = self.get_progress_path()
        if not ppath or not os.path.exists(ppath): return None
        try:
            with open(ppath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Basic validation
            if "status" not in data or "image_list" not in data:
                 QMessageBox.warning(self, "Invalid Progress File", "The progress file is old or corrupted. Starting a new session if possible.")
                 return None
            return data
        except Exception as e:
            QMessageBox.warning(self, "Progress Load Error", f"Failed to load progress:\n{e}")
            return None

    # ---------------- UI and Control Logic ----------------
    def choose_image_folder(self):
        """Opens a dialog to select the image folder."""
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder", os.getcwd())
        if d: self.le_image_folder.setText(d)

    def choose_output_folder(self):
        """Opens a dialog to select the output folder."""
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", os.getcwd())
        if d: self.le_output_folder.setText(d)
        
    def update_start_button_state(self):
        """Enables/disables the start button based on folder selection and image file presence."""
        image_folder = self.le_image_folder.text().strip()
        output_folder = self.le_output_folder.text().strip()

        if not image_folder or not output_folder or not os.path.isdir(image_folder):
            self.btn_start.setEnabled(False)
            return
        
        # Check if at least one image file exists
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
        has_images = any(glob.glob(os.path.join(image_folder, e)) for e in exts)
        
        self.btn_start.setEnabled(has_images)
        
        # Check for completed status but keep Start button enabled.
        progress_data = self.load_progress()
        if progress_data and progress_data.get("status") == "Completed":
            self.lbl_status.setText("This folder is marked as 'Completed'.")
            self.btn_img_folder.setEnabled(False)
            self.btn_output_folder.setEnabled(False)

    def update_param_fields(self):
        """Enables/disables related UI elements based on the selected binarization method."""
        method = self.cmb_method.currentText()
        is_manual = method == 'manual'
        is_adaptive = method == 'adaptive'
        self.slider_manual.setEnabled(is_manual)
        self.lbl_manual.setEnabled(is_manual)
        self.spin_adapt_bs.setEnabled(is_adaptive)
        self.spin_adapt_C.setEnabled(is_adaptive)

    # ---------------- Log-related Helper Functions ----------------
    def _ensure_log_initialized(self):
        """If the log file does not exist, initialize it by writing the header."""
        if not self.processing_log_path or os.path.exists(self.processing_log_path): return
        try:
            with open(self.processing_log_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.log_headers)
        except Exception as e:
            QMessageBox.warning(self, "Log Initialization Error", f"Failed to initialize the log file:\n{e}")

    def append_processing_log(self, entry_dict):
        """Appends a row to the log file."""
        if not self.processing_log_path: return
        self._ensure_log_initialized()
        row = [entry_dict.get(h, '') for h in self.log_headers]
        try:
            with open(self.processing_log_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            QMessageBox.warning(self, "Log Write Error", f"Failed to write to the log file:\n{e}")
    
    def _log_processing_event(self, result, path, features=None, error_msg=''):
        """Helper function to create and append a log entry."""
        method = self.cmb_method.currentText()
        base_name = os.path.basename(path)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': base_name,
            'mode': self.current_mode,
            'result': result,
            'num_contours': len(features) if features is not None else 0,
            'threshold_method': method,
            'manual_threshold': self.slider_manual.value() if method == 'manual' else 'N/A',
            'adaptive_blocksize': self.spin_adapt_bs.value() if method == 'adaptive' else 'N/A',
            'adaptive_C': self.spin_adapt_C.value() if method == 'adaptive' else 'N/A',
            'overlay_path': os.path.join(self.output_overlay_folder, base_name) if result == 'OK' else '',
            'individual_csv_path': os.path.join(self.output_individual_csv_folder, f"{os.path.splitext(base_name)[0]}.csv") if result == 'OK' else '',
            'summary_updated': result == 'OK',
            'error_message': error_msg
        }
        self.append_processing_log(entry)


    # ---------------- Core Processing Flow ----------------
    def start_or_resume_processing(self):
        """Main process when the 'Start/Resume' button is pressed."""
        try:
            base = self.le_output_folder.text().strip()
            self.output_overlay_folder = os.path.join(base, "overlay_images")
            self.output_individual_csv_folder = os.path.join(base, "individual_csv")
            os.makedirs(self.output_overlay_folder, exist_ok=True)
            os.makedirs(self.output_individual_csv_folder, exist_ok=True)
            self.summary_csv_path = os.path.join(base, "summary_features.csv")
            self.processing_log_path = os.path.join(base, "processing_log.csv")
        except Exception as e:
            QMessageBox.critical(self, "Output Folder Error", str(e))
            return

        progress_data = self.load_progress()
        
        if progress_data and progress_data.get("status") == "Completed":
            QMessageBox.information(self, "Completed", "Processing for this folder is already marked as complete.\nTo re-run, please delete the 'progress.json' file in the output folder.")
            self.lbl_status.setText("Processing Complete.")
            return

        if progress_data and progress_data.get("status") == "In Progress":
            self.image_list = progress_data["image_list"]
            self.processed_files = progress_data.get("processed_files", [])
            self.current_mode = progress_data.get("mode", "initial")
            self.ng_list = progress_data.get("ng_list", [])
            self.reprocess_ng_list = progress_data.get("reprocess_ng_list", [])
            
            # Determine the correct list to process for the current session
            if self.current_mode == 'reprocess':
                self.processing_queue = self.ng_list
            else:
                self.processing_queue = self.image_list

            next_index = len(self.processed_files)

            if next_index < len(self.processing_queue): # Resuming an interrupted pass
                self.lbl_status.setText(f"Resuming '{self.current_mode}' mode...")
                self.current_index = next_index
            else: # If the pass is complete, check for NG images to start the next pass
                if self.ng_list:
                    self.lbl_status.setText(f"Reprocessing {len(self.ng_list)} NG images...")
                    self.processing_queue = list(self.ng_list) # Set the new queue
                    self.current_mode = 'reprocess'
                    self.ng_list = []
                    self.reprocess_ng_list = []
                    self.processed_files = []
                    self.current_index = 0
                    self.save_progress() # Save state for the new reprocessing session
                else:
                    QMessageBox.information(self, "Complete", "All processing is complete. There are no remaining NG images.")
                    self.lbl_status.setText("Processing Complete.")
                    self.save_progress(completed=True)
                    return
        else: # Starting a new session
            self.lbl_status.setText("Starting a new session...")
            exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
            folder = self.le_image_folder.text().strip()
            self.image_list = [f for e in exts for f in sorted(glob.glob(os.path.join(folder, e)))]
            self.processing_queue = self.image_list[:] # The initial queue is the full list
            self.current_index = 0; self.ng_list = []; self.reprocess_ng_list = []; self.processed_files = []; self.current_mode = 'initial'
            self.save_progress()

        # Set the UI to the processing state
        self.btn_start.setEnabled(False)
        self.btn_ok.setEnabled(True); self.btn_ng.setEnabled(True); self.btn_prev.setEnabled(True); self.btn_next.setEnabled(True)
        self.progress_bar.setMaximum(len(self.processing_queue))
        self.process_and_show_current()

    def process_and_show_current(self):
        """Processes the image at the current index and displays the result."""
        if not (0 <= self.current_index < len(self.processing_queue)):
            # If all images in the current pass have been processed
            if self.current_mode == 'initial': self.finish_initial_pass()
            else: self.finish_reprocess_pass()
            return
        
        path = self.processing_queue[self.current_index]
        try:
            overlay, features = process_image_get_features(
                path, self.cmb_method.currentText(), self.slider_manual.value(),
                self.spin_adapt_bs.value(), self.spin_adapt_C.value())
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred while processing the image: {path}\n\n{e}")
            self.processed_files.append({'filename': os.path.basename(path), 'result': 'ERROR'})
            self._log_processing_event('ERROR', path, error_msg=str(e)) # Log the error
            self.save_progress()
            self.next_index()
            return
        
        # Display the image
        original = cv2.imread(path)
        self.label_original.setPixmap(cv2_to_qpixmap(original).scaled(self.label_original.size(), Qt.KeepAspectRatio))
        self.label_overlay.setPixmap(cv2_to_qpixmap(overlay).scaled(self.label_overlay.size(), Qt.KeepAspectRatio))
        self._last_processing = {'path': path, 'overlay_bgr': overlay, 'particle_features': features}

        # Update status
        mode_str = "[Reprocess] " if self.current_mode == 'reprocess' else ""
        self.lbl_status.setText(f"{mode_str}{os.path.basename(path)} | Contours: {len(features)}")
        self.progress_label.setText(f"Progress: {self.current_index + 1}/{len(self.processing_queue)}")
        self.progress_bar.setValue(self.current_index + 1)

    def mark_ok_and_next(self):
        """Handles the 'OK' button click. Saves the results and moves to the next image."""
        if not self.btn_ok.isEnabled() or not hasattr(self, '_last_processing'): return
        info = self._last_processing
        path, overlay, features = info['path'], info['overlay_bgr'], info['particle_features']
        
        # Log the OK event
        self._log_processing_event('OK', path, features=features)

        base_name = os.path.splitext(os.path.basename(path))[0]
        cols = ['perimeter', 'area', 'aspect_ratio', 'solidity', 'circularity'] + [f'hu{i+1}' for i in range(7)]
        df = pd.DataFrame(features, columns=cols) if features else pd.DataFrame(columns=cols)
        
        csv_path = os.path.join(self.output_individual_csv_folder, f"{base_name}.csv")
        df.to_csv(csv_path, index=False)
        cv2.imwrite(os.path.join(self.output_overlay_folder, os.path.basename(path)), overlay)

        # Append to the summary CSV file
        mean_features = df.mean().values.tolist() if not df.empty else [np.nan] * 12
        row = [os.path.basename(path)] + mean_features
        pd.DataFrame([row], columns=['filename']+cols).to_csv(self.summary_csv_path, mode='a', header=not os.path.exists(self.summary_csv_path), index=False)
        
        self.processed_files.append({'filename': os.path.basename(path), 'result': 'OK'})
        self.save_progress()
        self.next_index()

    def mark_ng_and_next(self):
        """Handles the 'NG' button click. Records the image in the NG list and moves on."""
        if not self.btn_ng.isEnabled() or not hasattr(self, '_last_processing'): return
        info = self._last_processing
        path = info['path']
        features = info.get('particle_features', [])

        # Log the NG event
        self._log_processing_event('NG', path, features=features)
        
        if self.current_mode == 'initial': self.ng_list.append(path)
        else: self.reprocess_ng_list.append(path)
            
        self.processed_files.append({'filename': os.path.basename(path), 'result': 'NG'})
        self.save_progress()
        self.next_index()

    def next_index(self):
        """Increments the index and processes the next image."""
        self.current_index += 1
        self.process_and_show_current()

    def prev_image(self):
        """Decrements the index to re-process the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            # Remove the processing result of the previous image from the list
            if self.processed_files:
                last_file = os.path.basename(self.processing_queue[self.current_index])
                if self.processed_files[-1]['filename'] == last_file:
                    entry = self.processed_files.pop()
                    if entry['result'] == 'NG':
                        path = self.processing_queue[self.current_index]
                        if self.current_mode == 'initial' and path in self.ng_list: self.ng_list.remove(path)
                        elif self.current_mode == 'reprocess' and path in self.reprocess_ng_list: self.reprocess_ng_list.remove(path)
            self.save_progress()
            self.process_and_show_current()

    def next_image(self): 
        """Handles the 'Next (as NG)' button click."""
        self.mark_ng_and_next()

    def _end_pass_ui_state(self):
        """Sets the UI state when a pass (initial or reprocess) is completed."""
        self.btn_ok.setEnabled(False); self.btn_ng.setEnabled(False); self.btn_prev.setEnabled(False); self.btn_next.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.progress_bar.setValue(len(self.processing_queue))

    def finish_initial_pass(self):
        """Logic for when the initial pass is complete."""
        self._end_pass_ui_state()
        if self.ng_list:
            msg = f"Initial processing complete. There are {len(self.ng_list)} NG images.\nPress 'Start / Resume' to begin reprocessing."
            self.lbl_status.setText("Initial pass complete. NG images found.")
            QMessageBox.information(self, "Initial Pass Complete", msg)
            self.save_progress()
        else:
            QMessageBox.information(self, "Processing Complete", "Initial processing complete. There are no NG images.")
            self.lbl_status.setText("Processing Complete.")
            self.save_progress(completed=True)

    def finish_reprocess_pass(self):
        """Logic for when a reprocessing pass is complete."""
        self._end_pass_ui_state()
        self.ng_list = list(self.reprocess_ng_list)
        self.reprocess_ng_list = []
        if self.ng_list:
            msg = f"Reprocessing complete. {len(self.ng_list)} NG images remain.\nPress 'Start / Resume' to reprocess them again."
            self.lbl_status.setText("Reprocessing complete. Some NG images remain.")
            QMessageBox.information(self, "Reprocess Pass Complete", msg)
            self.save_progress()
        else:
            QMessageBox.information(self, "Processing Complete", "Reprocessing complete. All images have been marked OK.")
            self.lbl_status.setText("Processing Complete.")
            self.save_progress(completed=True)

    def closeEvent(self, event):
        """Handles the window close event. Shows a confirmation dialog if processing has been started."""
        # A session is considered active if the output paths have been configured by starting the process.
        if self.processing_log_path:
             reply = QMessageBox.question(self, 'Confirm Exit',
                'Are you sure you want to exit?\nYour current progress will be saved.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

             if reply == QMessageBox.Yes:
                 # Save progress unless it's already marked as fully completed.
                 progress_data = self.load_progress()
                 is_completed = progress_data and progress_data.get("status") == "Completed"
                 if not is_completed:
                     self.save_progress()
                 event.accept()
             else:
                 event.ignore()
        else:
            event.accept()

# ------------------ Main Execution Block ------------------
def main():
    app = QApplication(sys.argv)
    gui = PIPGui()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
