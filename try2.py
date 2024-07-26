import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, \
    QSpacerItem, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
import imutils


class EmotionDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        self.emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.camera = cv2.VideoCapture(0)
        self.video_path = None
        self.video_capture = None

    def initUI(self):
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setFixedSize(300, 250)  # 固定情感分析结果窗口大小
        self.result_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton('Start Camera', self)
        self.stop_button = QPushButton('Stop Camera', self)
        self.upload_button = QPushButton('Upload Video', self)
        self.pause_button = QPushButton('Pause Recognition', self)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.upload_button.clicked.connect(self.upload_video)
        self.pause_button.clicked.connect(self.pause_recognition)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.result_label)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.upload_button)
        control_layout.addWidget(self.pause_button)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('Emotion Detector')
        self.setFixedSize(1280, 768)  # 固定总窗口大小

    def start_camera(self):
        if self.video_capture:
            self.video_capture.release()
        self.camera = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.camera.isOpened():
            self.camera.release()
        if self.video_capture:
            self.video_capture.release()

    def upload_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4 *.avi);;All Files (*)",
                                                   options=options)
        if file_name:
            self.video_path = file_name
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.timer.start(30)

    def pause_recognition(self):
        self.timer.stop()

    def update_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                self.video_capture.release()
                self.timer.stop()
                return
        else:
            ret, frame = self.camera.read()
            if not ret:
                return

        # 获取播放区域的大小
        display_width = self.video_label.width()
        display_height = self.video_label.height()

        # 按比例缩放视频帧
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        if display_width / display_height > aspect_ratio:
            new_height = display_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = display_width
            new_height = int(new_width / aspect_ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 创建黑色背景
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # 将缩放后的帧放置在黑色背景的中央
        top = (display_height - new_height) // 2
        left = (display_width - new_width) // 2
        canvas[top:top + new_height, left:left + new_width] = resized_frame

        # 进行情感检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

        emotion_canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = self.EMOTIONS[preds.argmax()]

            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(emotion_canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(emotion_canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                            2)

            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # 在视频帧上绘制情感分析结果和人脸框
        canvas[top:top + new_height, left:left + new_width] = cv2.resize(frameClone, (new_width, new_height))

        self.display_video(canvas)
        self.display_result(emotion_canvas)

    def display_video(self, img):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()

        self.video_label.setPixmap(QPixmap.fromImage(img))

    def display_result(self, canvas):
        qformat = QImage.Format_RGB888

        img = QImage(canvas, canvas.shape[1], canvas.shape[0], canvas.strides[0], qformat)
        img = img.rgbSwapped()

        self.result_label.setPixmap(QPixmap.fromImage(img))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionDetector()
    ex.show()
    sys.exit(app.exec_())
