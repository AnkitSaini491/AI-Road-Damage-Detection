
import streamlit as st
import cv2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ AI Road Damage Detection System")
st.write("Detect potholes and road cracks using Artificial Intelligence.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Damage classes
damage_classes = ["Pothole", "Crack"]

# Upload file
uploaded_file = st.file_uploader(
    "Upload Road Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type

    st.subheader("Detection Results")

    # Image detection
    if "image" in file_type:

        file_bytes = uploaded_file.read()

        image = cv2.imdecode(
            __import__("numpy").frombuffer(
                file_bytes, dtype="uint8"
            ),
            cv2.IMREAD_COLOR
        )

        results = model(image, conf=0.25)

        detected_image = results[0].plot()

        detected_image = cv2.cvtColor(
            detected_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            detected_image,
            caption="Road Damage Detection",
            use_container_width=True
        )

        counts = Counter()

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            counts[class_name] += 1

    # Video detection
    else:

        tfile = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        )

        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)

        video_display = st.empty()
        counts = Counter()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame, conf=0.25)

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                counts[class_name] += 1

            annotated = results[0].plot()

            annotated = cv2.cvtColor(
                annotated, cv2.COLOR_BGR2RGB
            )

            video_display.image(
                annotated,
                channels="RGB"
            )

        cap.release()

    # Display statistics
    st.subheader("Road Damage Statistics")

    col1, col2, col3 = st.columns(3)

    total = sum(counts.values())

    col1.metric("Total Detections", total)

    col2.metric(
        "Potholes",
        counts.get("Pothole", 0)
    )

    col3.metric(
        "Cracks",
        counts.get("Crack", 0)
    )

    # Results dataframe
    df = pd.DataFrame(
        list(counts.items()),
        columns=["Damage Type", "Count"]
    )

    st.subheader("Damage Analysis")

    st.dataframe(df, use_container_width=True)

    # Chart
    if not df.empty:

        fig, ax = plt.subplots()

        ax.bar(
            df["Damage Type"],
            df["Count"]
        )

        ax.set_title("Road Damage Detection Analysis")
        ax.set_xlabel("Damage Type")
        ax.set_ylabel("Number of Detections")

        st.pyplot(fig)

        # Download report
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Damage Report",
            csv,
            "road_damage_report.csv",
            "text/csv"
        )

    st.success("Road Damage Analysis Completed!")

else:
    st.info("Upload a road image or video to start detection.")

