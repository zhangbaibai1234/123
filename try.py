import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
import imutils
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


class EmotionDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("1200x800")

        # Create a frame for the video and result display
        self.display_frame = tk.Frame(self.root, width=900, height=800)
        self.display_frame.grid(row=0, column=0, padx=10, pady=10)

        self.video_frame = tk.Label(self.display_frame)
        self.video_frame.pack()

        self.canvas = tk.Label(self.display_frame)
        self.canvas.pack()

        # Create a frame for the control buttons
        self.control_frame = tk.Frame(self.root, width=300, height=800)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10)

        self.start_button = tk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack(pady=5)

        self.upload_button = tk.Button(self.control_frame, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=5)

        self.stop_upload_button = tk.Button(self.control_frame, text="Stop Video", command=self.stop_video)
        self.stop_upload_button.pack(pady=5)

        self.camera = None
        self.running = False
        self.video_stream = None

    def start_camera(self):
        if not self.running:
            self.camera = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            self.camera.release()
            self.video_frame.config(image='')

    def upload_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_stream = cv2.VideoCapture(file_path)
            self.running = True
            self.update_frame()

    def stop_video(self):
        if self.video_stream:
            self.video_stream.release()
            self.video_stream = None
            self.running = False
            self.video_frame.config(image='')
            self.canvas.config(image='')

    def update_frame(self):
        if self.running:
            if self.camera:
                ret, frame = self.camera.read()
            elif self.video_stream:
                ret, frame = self.video_stream.read()
            else:
                ret = False

            if ret:
                self.detect_emotion(frame)
                self.root.after(10, self.update_frame)

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        frameClone = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frameClone)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)

        canvas_img = Image.fromarray(canvas)
        canvas_imgtk = ImageTk.PhotoImage(image=canvas_img)
        self.canvas.imgtk = canvas_imgtk
        self.canvas.config(image=canvas_imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetector(root)
    root.mainloop()
