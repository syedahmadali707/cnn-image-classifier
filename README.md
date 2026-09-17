# 🖼️ CNN Image Classifier

A deep learning project that trains a **Convolutional Neural Network (CNN)**
for image classification, wrapped in a polished **Streamlit** dashboard —
complete with **Grad-CAM** visual explanations.

## Features
- **Quick Demo mode**: instantly train on a subset of CIFAR-10 (10 classes,
  auto-downloaded) — no dataset setup required, great for testing the pipeline.
- **Custom Dataset mode**: point the app at your own folder-per-class image
  dataset (plant disease, product defects, waste sorting, X-rays, etc.).
- Configurable CNN architecture (conv blocks, filters, dropout, learning rate).
- Live training in-browser with progress, loss/accuracy curves.
- Confusion matrix + full classification report (precision/recall/F1).
- Upload any image and get a prediction with **Grad-CAM heatmap** showing
  what the model focused on, plus a class-probability bar chart.
- Download the trained model (`.keras`) and class list (`.json`).

## Project structure
```
cnn-image-classifier/
├── app.py              # Streamlit app (UI)
├── model_utils.py       # CNN architecture, data loading, training, Grad-CAM
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually `http://localhost:8501`).

## How to use

### Option A — Quick Demo (no dataset needed)
1. In the sidebar, keep mode set to **🚀 Quick Demo (CIFAR-10)**.
2. Adjust samples-per-class / image size / CNN hyperparameters if you like.
3. Click **Train CNN Model** (first run downloads CIFAR-10 automatically, ~170MB).
4. Check the **Train & Evaluate** tab for curves and confusion matrix.
5. Go to **Predict**, upload any photo (a plane, car, animal, etc.), and see
   the prediction + Grad-CAM heatmap.

### Option B — Custom Dataset (e.g. plant disease detection)
1. Organize your images like this:
   ```
   data/train/
   ├── healthy/
   │   ├── img001.jpg
   │   └── ...
   ├── disease_a/
   │   └── ...
   └── disease_b/
       └── ...
   ```
   (A popular free dataset for this exact use case is the **PlantVillage**
   dataset, available on Kaggle.)
2. In the sidebar, switch mode to **📁 Custom Dataset (folder)**.
3. Enter the folder path (e.g. `data/train`). Leave the validation folder
   blank to auto-split, or provide a separate `data/val` folder.
4. Pick an image size (96 or 128 is a good default for real photos) and
   train.
5. Use the **Predict** tab to upload new leaf photos and see the diagnosis
   with a Grad-CAM overlay.

## Notes
- TensorFlow install is large (~500MB+); use `tensorflow-cpu` in
  `requirements.txt` if you don't have a GPU and want a smaller install.
- Grad-CAM automatically finds the last Conv2D layer to explain predictions.
- This project is for educational purposes only. For sensitive domains
  (e.g. medical imaging), never treat model output as a diagnosis.
