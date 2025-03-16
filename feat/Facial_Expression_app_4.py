import streamlit as st
from feat import Detector
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

# Initialize the detector
detector = Detector()

# Dictionary to map AU codes to their names
au_names = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU11": "Nasolabial Deepener",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU24": "Lip Pressor",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU28": "Lip Suck",
    "AU43": "Eyes Closed",
}

# Function to process the video
def process_video(input_video_path, output_video_path, confidence_threshold, selected_aus):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width + 400, height))  # Add space for the side panel

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=["Frame", "Emotion", "AU", "Confidence"])

    # Placeholder for real-time video display
    video_placeholder = st.empty()

    # Placeholder for cropped face display
    cropped_face_placeholder = st.empty()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Convert the OpenCV frame (NumPy array) to a PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Save the PIL image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_image_path = temp_file.name
            pil_image.save(temp_image_path, format="JPEG")

        # Detect facial expressions in the current frame
        results = detector.detect_image([temp_image_path])

        # Delete the temporary file
        os.remove(temp_image_path)

        # Overlay detected AUs and emotions on the frame
        if not results.empty:
            # Extract the first row (assuming only one face is detected)
            face = results.iloc[0]

            # Draw facial landmarks (if available)
            if "landmarks" in results.columns:
                landmarks = face["landmarks"]
                for (x, y) in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green dots for landmarks

            # Extract face bounding box coordinates
            if "FaceRectX" in results.columns and "FaceRectY" in results.columns and "FaceRectWidth" in results.columns and "FaceRectHeight" in results.columns:
                x = face["FaceRectX"]
                y = face["FaceRectY"]
                w = face["FaceRectWidth"]
                h = face["FaceRectHeight"]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)  # Blue rectangle for face bounding box

                # Crop the face region
                cropped_face = frame[int(y):int(y + h), int(x):int(x + w)]

                # Display the cropped face
                cropped_face_placeholder.image(cropped_face, channels="BGR", use_column_width=True)

            # Display face recognition confidence interval in the upper right corner
            if "confidence" in results.columns:
                confidence = face["confidence"]
                if confidence >= confidence_threshold:
                    confidence_text = f"Confidence: {confidence:.2f}"
                    text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = width - text_size[0] - 10  # Position text in the upper right corner
                    text_y = text_size[1] + 10
                    cv2.putText(frame, confidence_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Create a side panel for visualization
            side_panel = np.zeros((height, 400, 3), dtype=np.uint8)  # 400px wide side panel
            side_panel.fill(255)  # White background

            # Plot emotions as a horizontal bar chart
            emotion_columns = ["happiness", "sadness", "anger", "surprise", "fear", "disgust", "neutral"]
            emotion_values = [face[col] for col in emotion_columns if col in results.columns]

            plt.figure(figsize=(4, 4))
            plt.barh(emotion_columns, emotion_values, color=['green', 'blue', 'red', 'purple', 'orange', 'brown', 'gray'])
            plt.title("Emotions")
            plt.xlim(0, 1)  # Set x-axis limit to 1 (max intensity)
            plt.tight_layout()

            # Save the bar chart to a temporary image
            temp_chart_path = "temp_chart.png"
            plt.savefig(temp_chart_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # Load the bar chart image and overlay it on the side panel
            chart_image = cv2.imread(temp_chart_path)
            chart_height, chart_width, _ = chart_image.shape
            side_panel[10:10 + chart_height, 10:10 + chart_width] = chart_image

            # Plot selected AUs as a horizontal bar chart with names
            au_columns = [col for col in results.columns if col.startswith("AU") and col in selected_aus]
            au_values = [face[col] for col in au_columns]
            au_labels = [f"{col} - {au_names.get(col, 'Unknown')}" for col in au_columns]  # Add AU names

            plt.figure(figsize=(4, 4))
            plt.barh(au_labels, au_values, color='cyan')
            plt.title("Action Units")
            plt.xlim(0, 1)  # Set x-axis limit to 1 (max intensity)
            plt.tight_layout()

            # Save the bar chart to a temporary image
            temp_chart_path = "temp_chart.png"
            plt.savefig(temp_chart_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # Load the bar chart image and overlay it on the side panel
            chart_image = cv2.imread(temp_chart_path)
            chart_height, chart_width, _ = chart_image.shape
            side_panel[height // 2 + 10:height // 2 + 10 + chart_height, 10:10 + chart_width] = chart_image

            # Combine the video frame and side panel
            combined_frame = np.hstack((frame, side_panel))

            # Write the combined frame to the output video
            out.write(combined_frame)

            # Display the combined frame in real-time
            video_placeholder.image(combined_frame, channels="BGR", use_column_width=True)

            # Clean up temporary chart image
            os.remove(temp_chart_path)

            # Append results to DataFrame using pd.concat
            for au in au_columns:
                new_row = pd.DataFrame({
                    "Frame": [frame_count],
                    "Emotion": [emotion_columns[np.argmax(emotion_values)]],
                    "AU": [au],
                    "Confidence": [face[au]],
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update progress bar
        progress_bar.progress(min((frame_count + 1) / total_frames, 1.0))
        status_text.text(f"Processing frame {frame_count + 1} of {total_frames}")

        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    return results_df

# Streamlit UI
st.title("Facial Expression Analysis")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        input_video_path = temp_file.name

    # Specify output video path
    output_video_path = "output_video.mp4"

    # Customizable parameters
    st.sidebar.header("Customizable Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    selected_aus = st.sidebar.multiselect("Select Action Units (AUs)", list(au_names.keys()), default=list(au_names.keys()))

    # Process the video
    if st.button("Process Video"):
        st.write("Processing video...")
        results_df = process_video(input_video_path, output_video_path, confidence_threshold, selected_aus)
        st.write("Processing complete!")

        # Display the output video
        st.video(output_video_path)

        # Add a download button for the processed video
        with open(output_video_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

        # Export results as CSV
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Export Results as CSV",
            data=csv,
            file_name="facial_expression_results.csv",
            mime="text/csv",
        )

    # Clean up temporary files
    os.remove(input_video_path)