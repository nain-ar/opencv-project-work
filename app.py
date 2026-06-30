import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# ===============================
# Import Your Modules
# ===============================

from modules.gaussian_blur import GaussianBlurTool
from modules.median_blur import MedianBlurTool
from modules.edge import EdgeDetector
from modules.contour_detection import ContourDetector
from modules.motion_detection import MotionDetector
from modules.face_eye_detection import FaceEyeDetector

from datetime import datetime
import pandas as pd


#load css file
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ===============================
# Page Configuration
# ===============================


st.set_page_config(
    page_title="OpenCV Vision Studio",
    page_icon="🎯",
    layout="wide"
)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("static/style.css")
# ===============================
# Session State
# ===============================


if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# History Function
# ===============================

def add_history(operation, filename, status="Success"):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Operation": operation,
        "File": filename,
        "Status": status
    })


# ===============================
# Sidebar
# ===============================

st.sidebar.title("🖥 OpenCV Vision Studio")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📷 Image Processing",
        "🎥 Video Processing",
        "📜 History",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Professional OpenCV Toolkit"
)

# ===============================
# HOME PAGE
# ===============================

if page == "🏠 Home":

    st.title("🎯 OpenCV Vision Studio")

    st.markdown("""
    ## Welcome 👋

    This application is developed using **Python, OpenCV, and Streamlit**.

    It provides powerful Image and Video Processing tools through a simple web interface.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
### 📷 Image Processing

✔ Gaussian Blur

✔ Median Blur

✔ Edge Detection

✔ Contour Detection
""")

    with col2:
        st.success("""
### 🎥 Video Processing

✔ Motion Detection

✔ Face Detection

✔ Eye Detection

✔ Video Recording
""")

    st.divider()

    st.subheader("📊 Features")

    c1, c2, c3 = st.columns(3)

    c1.metric("Image Filters", "4")

    c2.metric("Video Tools", "3")

    c3.metric("Framework", "Streamlit")
# ===============================
# IMAGE PROCESSING PAGE
# ===============================

elif page == "📷 Image Processing":

    st.title("📷 Image Processing")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Original Image",
            use_container_width=True
        )

        operation = st.selectbox(
            "Choose Operation",
            [
                "Gaussian Blur",
                "Median Blur",
                "Edge Detection",
                "Contour Detection"
            ]
        )

        # ==========================================
        # Gaussian Blur
        # ==========================================

        if operation == "Gaussian Blur":

            kernel = st.slider(
                "Kernel Size",
                3,
                25,
                5,
                step=2
            )

            sigma = st.slider(
                "Sigma",
                0,
                10,
                0
            )

            if st.button("Apply Gaussian Blur"):

                tool = GaussianBlurTool()

                tool.load_image(image)

                result = tool.blur_image(
                    kernel,
                    sigma
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        cv2.cvtColor(
                            image,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Original",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        cv2.cvtColor(
                            result,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Gaussian Blur",
                        use_container_width=True
                    )

                _, buffer = cv2.imencode(
                    ".png",
                    result
                )

                st.download_button(
                    "📥 Download",
                    buffer.tobytes(),
                    "gaussian.png",
                    "image/png"
                )
                add_history(
    "Gaussian Blur",
    uploaded_image.name
)

        # ==========================================
        # Median Blur
        # ==========================================

        elif operation == "Median Blur":

            kernel = st.slider(
                "Kernel Size",
                3,
                25,
                5,
                step=2
            )

            if st.button("Apply Median Blur"):

                tool = MedianBlurTool()

                tool.load_image(image)

                result = tool.blur_image(kernel)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        cv2.cvtColor(
                            image,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Original",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        cv2.cvtColor(
                            result,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Median Blur",
                        use_container_width=True
                    )

                _, buffer = cv2.imencode(
                    ".png",
                    result
                )

                st.download_button(
                    "📥 Download",
                    buffer.tobytes(),
                    "median.png",
                    "image/png"
                )
                add_history(
    "Median Blur",
    uploaded_image.name
)

        # ==========================================
        # Edge Detection
        # ==========================================

        elif operation == "Edge Detection":

            t1 = st.slider(
                "Lower Threshold",
                0,
                255,
                100
            )

            t2 = st.slider(
                "Upper Threshold",
                0,
                255,
                200
            )

            if st.button("Detect Edges"):

                detector = EdgeDetector()

                detector.load_image(image)

                edge = detector.detect_edges(
                    t1,
                    t2
                )

                st.image(
                    edge,
                    caption="Edge Detection",
                    use_container_width=True
                )

                _, buffer = cv2.imencode(
                    ".png",
                    edge
                )

                st.download_button(
                    "📥 Download",
                    buffer.tobytes(),
                    "edge.png",
                    "image/png"
                )
                add_history(
    "Edge Detection",
    uploaded_image.name
)

        # ==========================================
        # Contour Detection
        # ==========================================

        elif operation == "Contour Detection":

            threshold = st.slider(
                "Threshold",
                0,
                255,
                200
            )

            if st.button("Find Contours"):

                detector = ContourDetector()

                detector.load_image(image)

                result, thresh, total = detector.process(
                    threshold
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.image(
                        thresh,
                        caption="Threshold Image",
                        use_container_width=True
                    )

                with c2:

                    st.image(
                        cv2.cvtColor(
                            result,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption=f"Contours : {total}",
                        use_container_width=True
                    )

                _, buffer = cv2.imencode(
                    ".png",
                    result
                )

                st.download_button(
                    "📥 Download",
                    buffer.tobytes(),
                    "contours.png",
                    "image/png"
                )
                add_history(
    "Contour Detection",
    uploaded_image.name
)

# ===============================
# VIDEO PAGE
# ===============================

elif page == "🎥 Video Processing":

    st.title("🎥 Video Processing")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.success("Video uploaded successfully.")

        st.video(temp_video.name)

        operation = st.selectbox(
            "Choose Operation",
            (
                "Motion Detection",
                "Face & Eye Detection",
                "Video Recorder"
            )
        )

        output_video = os.path.join(
            "outputs",
            "processed_video.mp4"
        )

        os.makedirs(
            "outputs",
            exist_ok=True
        )


        # ======================================
        # Motion Detection
        # ======================================

        if operation == "Motion Detection":

            st.subheader("Motion Detection")

            threshold = st.slider(
                "Threshold",
                10,
                100,
                20
            )

            blur_kernel = st.select_slider(
                "Blur Kernel",
                options=[3,5,7,9],
                value=5
            )

            min_area = st.slider(
                "Minimum Motion Area",
                500,
                5000,
                1000
            )

            if st.button("Process Video"):

                with st.spinner(
                    "Detecting Motion..."
                ):

                    detector = MotionDetector()

                    detector.load_video(
                        temp_video.name
                    )

                    detector.process_video(
                        output_video,
                        min_area,
                        threshold,
                        blur_kernel
                    )

                st.success(
                    "Processing Complete!"
                )

                st.video(
                    output_video
                )

                with open(
                    output_video,
                    "rb"
                ) as file:

                    st.download_button(
                        "📥 Download Processed Video",
                        file,
                        file_name="motion_detection.mp4"
                    )
                    add_history(
    "Motion Detection",
    uploaded_video.name
)
        # ======================================
        # Face & Eye Detection
        # ======================================

        elif operation == "Face & Eye Detection":

            st.subheader("Face & Eye Detection")

            scale = st.slider(
                "Scale Factor",
                1.1,
                2.0,
                1.2,
                0.1
            )

            neighbors = st.slider(
                "Minimum Neighbors",
                3,
                10,
                5
            )

            if st.button("Detect Faces"):

                with st.spinner("Processing Video..."):

                    detector = FaceEyeDetector()

                    detector.load_video(
                        temp_video.name
                    )

                    detector.process_video(
                        output_video,
                        scale,
                        neighbors
                    )

                st.success("Processing Complete!")

                st.video(output_video)

                with open(
                    output_video,
                    "rb"
                ) as file:

                    st.download_button(
                        "📥 Download Processed Video",
                        file,
                        file_name="face_eye_detection.mp4"
                    )
                                                
                    add_history(
    "Face & Eye Detection",
    uploaded_video.name
)
            elif operation == "Video Recorder":

                st.subheader("🎥 Record Video From Webcam")

                filename = st.text_input(
                    "Output File Name",
                    "recorded_video.avi"
                )

                if st.button("Start Recording"):

                    recorder = VideoRecorder(filename)

                    recorder.record_video()

                    add_history(
                        "Video Recorder",
                        filename
                    )

                    st.success("Recording Completed!")

                    if os.path.exists(filename):

                        with open(filename, "rb") as file:

                            st.download_button(
                                "📥 Download Recorded Video",
                                file,
                                file_name=filename
                            )
                            add_history(
    "Face & Eye Detection",
    uploaded_video.name
)




# HISTORY PAGE
# ===============================

elif page == "📜 History":

    st.title("📜 Processing History")

    if len(st.session_state.history) == 0:

        st.warning("No processing history found.")

    else:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.info(f"Total Operations : {len(df)}")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download History",
            csv,
            "history.csv",
            "text/csv"
        )

        if st.button("🗑 Clear History"):

            st.session_state.history.clear()

            st.success("History Cleared!")

            st.rerun()

# ===============================
# ABOUT PAGE
# ===============================
elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
# OpenCV Vision Studio

This project was developed using

- Python
- OpenCV
- Streamlit
- NumPy

---

## Available Modules

✔ Gaussian Blur

✔ Median Blur

✔ Edge Detection

✔ Contour Detection

✔ Motion Detection

✔ Face Detection

✔ Eye Detection

✔ Video Recorder

---

### Developed For

Computer Vision & Image Processing Project

Made with  using OpenCV
""")