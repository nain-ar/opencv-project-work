import cv2


class ContourDetector:
    """
    Contour Detection Tool
    """

    def __init__(self):
        self.original_img = None
        self.gray_img = None
        self.threshold_img = None
        self.result_img = None
        self.contours = None
        self.hierarchy = None

    def load_image(self, image):
        """
        Load image from Streamlit.
        """
        self.original_img = image.copy()

    def convert_to_grayscale(self):
        """
        Convert image to grayscale.
        """

        self.gray_img = cv2.cvtColor(
            self.original_img,
            cv2.COLOR_BGR2GRAY
        )

        return self.gray_img

    def apply_threshold(self, threshold_value=200):
        """
        Apply binary threshold.
        """

        _, self.threshold_img = cv2.threshold(
            self.gray_img,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        return self.threshold_img

    def find_contours(self):
        """
        Detect contours.
        """

        self.contours, self.hierarchy = cv2.findContours(
            self.threshold_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        return self.contours

    def draw_contours(
        self,
        color=(0, 255, 0),
        thickness=2
    ):
        """
        Draw contours on image.
        """

        self.result_img = self.original_img.copy()

        cv2.drawContours(
            self.result_img,
            self.contours,
            -1,
            color,
            thickness
        )

        return self.result_img

    def process(
        self,
        threshold_value=200,
        color=(0, 255, 0),
        thickness=2
    ):
        """
        Complete contour detection pipeline.
        """

        self.convert_to_grayscale()

        self.apply_threshold(
            threshold_value
        )

        self.find_contours()

        result = self.draw_contours(
            color,
            thickness
        )

        return (
            result,
            self.threshold_img,
            len(self.contours)
        )

    def get_threshold_image(self):
        """
        Return threshold image.
        """

        return self.threshold_img

    def get_contour_count(self):
        """
        Return total contours detected.
        """

        if self.contours is None:
            return 0

        return len(self.contours)