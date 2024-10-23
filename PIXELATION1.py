import streamlit as st
import cv2
import yt_dlp

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from roboflow import Roboflow

# Initialize Roboflow with your API key and project/model details
rf = Roboflow(api_key="ZnJzXaB86Tol6zubMIPd")  # Your Roboflow API key
project = rf.workspace().project("pixelation_mk")  # Your project ID
model = project.version(1).model  # Model version

# Set up the Streamlit app
st.title("YouTube Video Object Detection and Pixelation")
st.write("Detect and pixelate 'pixel' regions using the Roboflow model.")

# Input for YouTube URL
youtube_url = st.text_input("YouTube Video URL", placeholder="Enter a valid YouTube URL")

# Function to extract the video stream URL using yt-dlp
def get_video_stream(url):
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# Function to pixelate a specific region of an image
def pixelate_region(image, x0, y0, x1, y1, pixelation_level=15):
    roi = image[y0:y1, x0:x1]  # Extract the region of interest (ROI)
    roi = cv2.resize(roi, (pixelation_level, pixelation_level), interpolation=cv2.INTER_LINEAR)
    roi = cv2.resize(roi, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)  # Scale it back
    image[y0:y1, x0:x1] = roi  # Replace original region with pixelated version
    return image

# Video Transformer class for Streamlit WebRTC
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model  # Load Roboflow model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to ndarray

        # Perform object detection using Roboflow
        result = self.model.predict(img, confidence=40, overlap=30).json()

        # Loop through predictions and draw bounding boxes/pixelate
        for prediction in result['predictions']:
            x0 = int(prediction['x'] - prediction['width'] / 2)
            y0 = int(prediction['y'] - prediction['height'] / 2)
            x1 = int(prediction['x'] + prediction['width'] / 2)
            y1 = int(prediction['y'] + prediction['height'] / 2)
            class_name = prediction['class']

            # Draw a bounding box with label
            color = (0, 255, 0) if class_name == "pixel" else (255, 0, 0)  # Green for pixel, blue otherwise
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)  # Draw bounding box
            cv2.putText(img, class_name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Add label

            # Apply pixelation if the class is "pixel"
            if class_name == "pixel":
                img = pixelate_region(img, x0, y0, x1, y1)

        return img

# Check if a valid YouTube URL is provided
if youtube_url:
    try:
        # Get the video stream URL using yt-dlp
        stream_url = get_video_stream(youtube_url)

        # Display the video in the Streamlit app
        st.video(stream_url)

        # Start real-time detection and pixelation using WebRTC
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    except Exception as e:
        st.error(f"Failed to load video: {str(e)}")
