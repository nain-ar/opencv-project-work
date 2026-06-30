import cv2
import os


class VideoRecorder:
    """
    Webcam Video Recorder
    """

    def __init__(self):
        self.camera = None
        self.writer = None
        self.output_path = None

    def start_recording(
        self,
        output_path="outputs/recorded_video.mp4",
        fps=20
    ):
        """
        Record video from webcam and save it.
        Press 'Q' to stop recording.
        """

        self.output_path = output_path

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            raise ValueError("Unable to access webcam.")

        width = int(
            self.camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while True:

            ret, frame = self.camera.read()

            if not ret:
                break

            cv2.putText(
                frame,
                "Recording... Press Q to Stop",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            self.writer.write(frame)

            cv2.imshow(
                "Video Recorder",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.release()
        self.writer.release()

        cv2.destroyAllWindows()

        return output_path