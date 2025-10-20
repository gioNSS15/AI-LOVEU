
import streamlit as st
from PIL import Image
import io
import tempfile
import os

st.set_page_config(page_title="Image Detection / Segmentation App", layout="wide")

st.title("🔎 Image Detection & Segmentation (Streamlit)")
st.write("Upload an image and (optionally) a model (.pt for Ultralytics YOLO). "
         "If you don't have a model handy, the app will run a dummy inference for demo.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    model_file = st.file_uploader("Upload a model file (.pt) — optional", type=["pt","onnx","yaml"])
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    run = st.button("Run inference")

with col2:
    st.header("Preview / Result")
    if uploaded_image is None:
        st.info("Upload an image to get started. Use the sample screenshots (in repo) as a reference.")
    else:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)
        if not run:
            st.caption("Click **Run inference** to detect/segment objects.")

# Utility: attempt to load Ultralytics YOLO model if available
def load_ultralytics_model(path):
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    except Exception as e:
        st.warning(f"Could not load Ultralytics model: {e}")
        return None

def dummy_inference_pil(image_pil):
    # Draw a simple red box and label as a demo
    import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
    im = image_pil.copy()
    draw = ImageDraw.Draw(im)
    w,h = im.size
    box = (int(w*0.15), int(h*0.15), int(w*0.75), int(h*0.75))
    draw.rectangle(box, outline="red", width=6)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=24)
    except:
        font = ImageFont.load_default()
    draw.text((box[0], box[1]-28), "demo_object:0.99", fill="red", font=font)
    return im

if run:
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        # Save uploaded image to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.getbuffer())
        tfile.flush()
        tfile.close()

        result_image = None
        # Try to load user-provided model (if any)
        model = None
        if model_file is not None:
            # save model file to temp and try to load with ultralytics
            model_path = os.path.join(tempfile.gettempdir(), model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())
            model = load_ultralytics_model(model_path)

        if model is not None:
            st.info("Running inference with Ultralytics YOLO model...")
            try:
                # Use the model to predict; results[0].plot() returns an annotated image (numpy array)
                results = model.predict(source=tfile.name, conf=conf, save=False, verbose=False)
                # results might be a list-like, take first
                r = results[0]
                try:
                    annotated = r.plot()
                    # convert to PIL
                    import numpy as np
                    annotated_pil = Image.fromarray(annotated)
                    result_image = annotated_pil
                except Exception as ex:
                    st.warning(f"Could not use result.plot(): {ex}")
                    st.write(r.boxes)  # debug info
            except Exception as exc:
                st.error(f"Model inference failed: {exc}")
                result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
        else:
            st.info("No compatible model loaded — running demo (dummy) inference.")
            result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))

        if result_image is not None:
            st.image(result_image, caption="Detection / Segmentation result", use_column_width=True)

st.markdown("---")
st.markdown("### How to adapt this app for your notebook/model")
st.markdown("""
1. If your notebook contains custom loading/predict functions (e.g. `load_model()`, `run_inference()`), copy them into `app.py`.
2. Replace the `load_ultralytics_model` and inference block with your model's loading and prediction code.
3. For segmentation masks you can overlay masks on the image using the mask arrays (numpy) and `PIL.Image` or OpenCV.
4. Commit `app.py` and `requirements.txt` to your repo and deploy to Streamlit Cloud.
""")
