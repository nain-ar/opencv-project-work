import cv2


class FaceEyeDetector:
    """
    Face and Eye Detection Tool
    """

    def __init__(self):
        self.video_path = None
        self.output_path = None

        # Load Haar Cascade files
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_eye.xml"
        )

    def load_video(self, video_path):
        """
        Store uploaded video path.
        """
        self.video_path = video_path

    def process_video(
        self,
        output_path,
        scale_factor=1.2,
        min_neighbors=5
    ):
        """
        Detect faces and eyes and save the processed video.
        """

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError("Unable to open video.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors
            )

            for (x, y, w, h) in faces:

                # Draw face rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=6
                )

                for (ex, ey, ew, eh) in eyes:

                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (0, 255, 0),
                        2
                    )

            writer.write(frame)

        cap.release()
        writer.release()

        return output_path