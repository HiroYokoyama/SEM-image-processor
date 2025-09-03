# Particle Image Processor

A graphical tool for analyzing particle images using **OpenCV** and **PyQt5**.
It allows batch processing of images, contour detection, and extraction of particle features such as area, perimeter, circularity, and Hu moments.
The application supports multiple thresholding methods and provides interactive validation with **OK / NG marking**.

---

## Features

* 📂 Select input and output folders via GUI
* 🖼️ Displays **original** and **processed** images side by side
* ⚙️ Multiple binarization methods:

  * Otsu
  * Manual threshold
  * Adaptive threshold
  * Scikit-image methods (Li, Triangle, Yen, IsoData) if available
* ✅/❌ Mark images as **OK** or **NG**
* 🔄 Reprocess NG images until all are processed
* 💾 Saves:

  * Overlay images (with contours and IDs)
  * Individual CSV (per image) with extracted particle features
  * Summary CSV (mean features across images)
  * Processing log (CSV)
  * Progress state (JSON, resumable)
* ⏸️ Safe exit handling (prevents quitting during processing)

---

## Extracted Features

For each particle contour:

* Perimeter
* Area
* Aspect Ratio
* Solidity
* Circularity
* Hu Moments (1–7)

---

## Requirements

* Python 3.7+
* Dependencies:

```bash
pip install PyQt5 opencv-python numpy pandas scikit-image
```

If you only need headless (no GUI), use `opencv-python-headless`.

---

## Usage

1. Clone or download the repository:

```bash
git clone https://github.com/HiroYokoyama/particle-image-processor.git
cd particle-image-processor
```

2. Run the application:

```bash
python particle-ip.py
```

3. In the GUI:

   * Select **Image Folder** (input)
   * Select **Output Folder** (results)
   * Choose **Thresholding Method**
   * Click **Start / Resume**

4. Mark images as **OK** or **NG**

   * OK → Saves overlay, CSV, and updates summary
   * NG → Added to reprocessing queue

5. After initial pass, you can reprocess NG images until completed.

---

## Output Files

* `overlay_images/` → processed images with contours
* `individual_csv/` → per-image features
* `summary_features.csv` → aggregated features
* `processing_log.csv` → record of processing events
* `progress.json` → session state (for resume)

---

## License

This project is open-source under the Apache 2.0 License.
See [LICENSE](LICENSE) for details.

