import cv2


class MotionDetector:
    """
    Motion Detection Tool
    """

    def __init__(self):
        self.video_path = None
        self.output_path = None

    def load_video(self, video_path):
        """
        Store video path.
        """
        self.video_path = video_path

    def process_video(
        self,
        output_path,
        min_area=1000,
        threshold_value=20,
        blur_kernel=5
    ):
        """
        Detect motion and save processed video.
        """

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError("Unable to open video.")

        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        if not ret1 or not ret2:
            cap.release()
            raise ValueError("Video contains insufficient frames.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while cap.isOpened():

            diff = cv2.absdiff(frame1, frame2)

            gray = cv2.cvtColor(
                diff,
                cv2.COLOR_BGR2GRAY
            )

            blur = cv2.GaussianBlur(
                gray,
                (blur_kernel, blur_kernel),
                0
            )

            _, thresh = cv2.threshold(
                blur,
                threshold_value,
                255,
                cv2.THRESH_BINARY
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:

                if cv2.contourArea(contour) < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame1,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame1,
                    "Motion Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            writer.write(frame1)

            frame1 = frame2
            ret, frame2 = cap.read()

            if not ret:
                break

        cap.release()
        writer.release()

        return output_path