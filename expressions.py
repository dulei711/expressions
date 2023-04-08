import streamlit as st
import cv2
from deepface import DeepFace

# Define a function to capture video from webcam and analyze sentiment
def detect_sentiment():
    # Set up video capture from default camera
    cap = cv2.VideoCapture(0)

    # Define emotions to be analyzed
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Run face analysis on the current frame
        result = DeepFace.analyze(frame, actions=emotions)

        # Get the predicted emotion and its probability
        predicted_emotion = max(result["emotion"].items(), key=lambda x: x[1])[0]
        probability = result["emotion"][predicted_emotion]

        # Display the current frame with the predicted emotion and its probability
        cv2.putText(frame, predicted_emotion + " ({:.2f}%)".format(probability*100),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Sentiment Analysis', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Create the Streamlit web app
st.title("Webcam Sentiment Analysis")
st.write("This app uses the DeepFace library to analyze sentiment from webcam expressions.")

# Add a button to start the sentiment analysis
if st.button("Start Sentiment Analysis"):
    detect_sentiment()
