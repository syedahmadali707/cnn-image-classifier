"""
CNN Image Classifier — Streamlit App
Train a Convolutional Neural Network on a built-in quick-demo dataset
(CIFAR-10) or your own custom image folder dataset, then classify
uploaded images with Grad-CAM visual explanations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

from model_utils import (
    load_cifar10_subset,
    build_array_generators,
    build_directory_generators,
    build_cnn_model,
    train_cnn,
    preprocess_image,
    make_gradcam_heatmap,
    overlay_heatmap,
    model_to_bytes,
    class_names_to_bytes,
)

# --------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #12151d 100%); }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Segoe UI', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1c2333 0%, #232a3d 100%);
        border: 1px solid #2d3548; border-radius: 14px;
        padding: 18px 20px; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-card h3 { font-size: 13px; color: #9aa4b2; margin-bottom: 6px; font-weight: 500; }
    .metric-card p { font-size: 26px; font-weight: 700; margin: 0; color: #f0f2f6; }

    .pred-box {
        background: linear-gradient(135deg, #1c2333 0%, #232a3d 100%);
        border: 1px solid #4f6df5; border-radius: 16px;
        padding: 26px; text-align: center; margin-top: 10px; margin-bottom: 10px;
    }
    .pred-box h1 { font-size: 36px; margin: 0; color: #7c93ff; text-transform: capitalize; }
    .pred-box p { color: #c3c9d4; font-size: 15px; margin-top: 6px; }

    section[data-testid="stSidebar"] { background-color: #12151d; border-right: 1px solid #232a3d; }

    div.stButton > button {
        background: linear-gradient(135deg, #4f6df5 0%, #3752d6 100%);
        color: white; border: none; border-radius: 10px;
        padding: 10px 18px; font-weight: 600; width: 100%;
    }
    div.stButton > button:hover { background: linear-gradient(135deg, #5c78ff 0%, #4560e8 100%); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
for key in ["model", "history", "class_names", "img_size", "results", "x_sample"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    mode = st.radio("Dataset mode", ["🚀 Quick Demo (CIFAR-10)", "📁 Custom Dataset (folder)"])

    if mode.startswith("🚀"):
        samples_per_class = st.slider("Samples per class", 100, 1000, 300, 50)
        img_size = st.select_slider("Image size", options=[32, 48, 64], value=32)
        train_dir = val_dir = None
        validation_split = None
    else:
        train_dir = st.text_input("Train folder path", value="data/train",
                                   help="Folder containing one sub-folder per class, e.g. data/train/classA/*.jpg")
        val_dir = st.text_input("Validation folder path (optional)", value="",
                                 help="Leave empty to auto-split the train folder instead")
        validation_split = None if val_dir.strip() else st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
        img_size = st.select_slider("Image size", options=[64, 96, 128, 160], value=96)
        samples_per_class = None

    st.markdown("---")
    st.markdown("### 🧠 CNN Hyperparameters")
    conv_blocks = st.select_slider("Conv blocks", options=[2, 3, 4], value=3)
    base_filters = st.select_slider("Base filters", options=[16, 32, 64], value=32)
    dropout = st.slider("Dropout rate", 0.0, 0.6, 0.4, 0.05)
    lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
    epochs = st.slider("Epochs", 5, 60, 15, 5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64], value=32)

    st.markdown("---")
    train_clicked = st.button("🚀 Train CNN Model")

    if st.session_state.model is not None:
        st.markdown("---")
        st.download_button(
            "⬇️ Download trained model (.keras)",
            data=model_to_bytes(st.session_state.model),
            file_name="cnn_image_classifier.keras",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download class names (.json)",
            data=class_names_to_bytes(st.session_state.class_names),
            file_name="class_names.json",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("Educational deep-learning demo project.")

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown("# 🖼️ CNN Image Classifier")
st.markdown(
    "Train a **Convolutional Neural Network** on the built-in CIFAR-10 demo "
    "dataset, or point it at your own image folder (e.g. plant disease, "
    "product defects, X-rays). Then classify new images with **Grad-CAM** "
    "visual explanations of what the model focused on."
)

# --------------------------------------------------------------------------
# TRAINING TRIGGER
# --------------------------------------------------------------------------
if train_clicked:
    try:
        with st.spinner("Preparing data..."):
            if mode.startswith("🚀"):
                (x_train, y_train, _), (x_test, y_test, _), class_names = load_cifar10_subset(
                    samples_per_class=samples_per_class, img_size=img_size
                )
                train_gen, val_gen = build_array_generators(x_train, y_train, x_test, y_test, batch_size=batch_size)
                st.session_state.x_sample = x_train[:8]
            else:
                if not train_dir.strip():
                    st.error("Please provide a train folder path.")
                    st.stop()
                train_gen, val_gen, class_names = build_directory_generators(
                    train_dir.strip(),
                    val_dir.strip() if val_dir.strip() else None,
                    img_size=img_size,
                    batch_size=batch_size,
                    validation_split=validation_split,
                )
                sample_batch, _ = next(train_gen)
                st.session_state.x_sample = sample_batch[:8]

        with st.spinner(f"Training CNN for up to {epochs} epochs... this can take a few minutes."):
            model = build_cnn_model(
                input_shape=(img_size, img_size, 3),
                num_classes=len(class_names),
                conv_blocks=conv_blocks,
                base_filters=base_filters,
                dropout=dropout,
                lr=lr,
            )
            history = train_cnn(model, train_gen, val_gen, epochs=epochs)

            val_gen.reset() if hasattr(val_gen, "reset") else None
            y_true, y_pred = [], []
            if mode.startswith("🚀"):
                preds = model.predict(x_test, verbose=0)
                y_pred = np.argmax(preds, axis=1)
                y_true = np.argmax(y_test, axis=1)
            else:
                steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
                preds = model.predict(val_gen, steps=steps, verbose=0)
                y_pred = np.argmax(preds, axis=1)[: val_gen.samples]
                y_true = val_gen.classes[: len(y_pred)]

        st.session_state.model = model
        st.session_state.history = history
        st.session_state.class_names = class_names
        st.session_state.img_size = img_size
        st.session_state.results = dict(y_true=np.array(y_true), y_pred=np.array(y_pred))
        st.success(f"Training complete! {len(class_names)} classes: {', '.join(class_names)}")
    except Exception as e:
        st.error(f"Training failed: {e}")

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab_train, tab_predict, tab_about = st.tabs(["🏋️ Train & Evaluate", "🔮 Predict", "ℹ️ About"])

# ---- TRAIN & EVALUATE TAB --------------------------------------------------
with tab_train:
    if st.session_state.model is None:
        st.info("👈 Configure the dataset and hyperparameters in the sidebar, then click **Train CNN Model**.")
    else:
        history = st.session_state.history
        results = st.session_state.results
        class_names = st.session_state.class_names

        final_val_acc = history.history["val_accuracy"][-1]
        final_val_loss = history.history["val_loss"][-1]
        best_val_acc = max(history.history["val_accuracy"])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card"><h3>CLASSES</h3><p>{len(class_names)}</p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><h3>FINAL VAL ACC</h3><p>{final_val_acc*100:.1f}%</p></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><h3>BEST VAL ACC</h3><p>{best_val_acc*100:.1f}%</p></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card"><h3>FINAL VAL LOSS</h3><p>{final_val_loss:.3f}</p></div>""", unsafe_allow_html=True)

        if st.session_state.x_sample is not None:
            st.markdown("### Sample Training Images")
            cols = st.columns(8)
            for i, col in enumerate(cols):
                if i < len(st.session_state.x_sample):
                    col.image(st.session_state.x_sample[i], use_container_width=True)

        st.markdown("### Training Curves")
        col1, col2 = st.columns(2)
        with col1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss", line=dict(color="#4f6df5")))
            fig_loss.add_trace(go.Scatter(y=history.history["val_loss"], name="Val Loss", line=dict(color="#ff4d6d")))
            fig_loss.update_layout(title="Loss", template="plotly_dark", height=320,
                                    margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_loss, use_container_width=True)
        with col2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=history.history["accuracy"], name="Train Acc", line=dict(color="#00d68f")))
            fig_acc.add_trace(go.Scatter(y=history.history["val_accuracy"], name="Val Acc", line=dict(color="#ff9f43")))
            fig_acc.update_layout(title="Accuracy", template="plotly_dark", height=320,
                                   margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(results["y_true"], results["y_pred"], labels=list(range(len(class_names))))
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=class_names, y=class_names, colorscale="Blues",
            text=cm, texttemplate="%{text}",
        ))
        fig_cm.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("Classification report"):
            report = classification_report(
                results["y_true"], results["y_pred"], target_names=class_names,
                zero_division=0, output_dict=True,
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        with st.expander("Model architecture summary"):
            lines = []
            st.session_state.model.summary(print_fn=lambda x: lines.append(x))
            st.code("\n".join(lines))

# ---- PREDICT TAB -----------------------------------------------------------
with tab_predict:
    if st.session_state.model is None:
        st.info("Train a model first (sidebar), then come back here to classify images.")
    else:
        model = st.session_state.model
        class_names = st.session_state.class_names
        img_size = st.session_state.img_size

        uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png", "bmp", "webp"])

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            img_array = preprocess_image(pil_img, img_size)

            preds = model.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_class = class_names[top_idx]
            top_conf = float(preds[top_idx])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original Image")
                st.image(pil_img, use_container_width=True)

            with col2:
                st.markdown("#### Grad-CAM — where the model looked")
                try:
                    heatmap = make_gradcam_heatmap(img_array, model, pred_index=top_idx)
                    overlaid = overlay_heatmap(pil_img, heatmap)
                    st.image(overlaid, use_container_width=True)
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable for this model: {e}")

            st.markdown(
                f"""
                <div class="pred-box">
                    <h1>{top_class}</h1>
                    <p>Confidence: {top_conf*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Class probabilities")
            sorted_idx = np.argsort(preds)[::-1]
            fig_bar = go.Figure(go.Bar(
                x=[preds[i] * 100 for i in sorted_idx],
                y=[class_names[i] for i in sorted_idx],
                orientation="h",
                marker_color="#4f6df5",
            ))
            fig_bar.update_layout(
                template="plotly_dark", height=max(300, 30 * len(class_names)),
                margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Probability (%)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Upload a JPG/PNG image above to see the model's prediction.")

# ---- ABOUT TAB -----------------------------------------------------------
with tab_about:
    st.markdown("""
### About this project

This app trains a **Convolutional Neural Network (CNN)** for image
classification, entirely from the browser.

**Two ways to use it:**
1. **Quick Demo (CIFAR-10)** — instantly trains on a subset of the classic
   10-class CIFAR-10 dataset (airplane, automobile, bird, cat, deer, dog,
   frog, horse, ship, truck), auto-downloaded via Keras. No setup needed.
2. **Custom Dataset** — point it at your own folder of images organized as
   one sub-folder per class:
   ```
   data/train/
   ├── classA/  (img1.jpg, img2.jpg, ...)
   ├── classB/  ...
   └── classC/  ...
   ```
   Great for projects like plant-disease detection, product-defect
   inspection, waste sorting, or any custom image classification task.

**Architecture**: A configurable CNN — stacked Conv2D blocks (two conv
layers + BatchNorm + MaxPooling + Dropout per block, doubling filters each
block) → Global Average Pooling → Dense layer → Softmax output. Trained
with Adam, categorical cross-entropy, early stopping and learning-rate
reduction on plateau.

**Explainability**: Predictions come with a **Grad-CAM** heatmap overlay,
showing which regions of the image most influenced the model's decision —
useful for sanity-checking that the CNN is "looking" at the right things.

**Evaluation**: Training/validation loss & accuracy curves, confusion
matrix, and a full classification report (precision/recall/F1 per class).

**Disclaimer**: This is an educational/demo project. For sensitive domains
(e.g. medical imaging) any model output should never be treated as a
diagnosis — always defer to qualified professionals.
""")
