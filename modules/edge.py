import cv2


class EdgeDetector:
    """
    Canny Edge Detection Tool
    """

    def __init__(self):
        self.original_img = None
        self.gray_img = None
        self.edge_versions = {}
        self.selected_threshold = None

    def load_image(self, image):
        """
        Load image from Streamlit.
        """

        self.original_img = image.copy()

        self.gray_img = cv2.cvtColor(
            self.original_img,
            cv2.COLOR_BGR2GRAY
        )

        self.edge_versions.clear()

    def detect_edges(self, threshold1, threshold2):
        """
        Apply Canny Edge Detection.
        """

        edge = cv2.Canny(
            self.gray_img,
            threshold1,
            threshold2
        )

        key = f"{threshold1}-{threshold2}"

        self.edge_versions[key] = edge

        return edge

    def generate_multiple(self, threshold_pairs):
        """
        Generate multiple edge images.

        Example:
        [
            (50,150),
            (100,200),
            (150,250)
        ]
        """

        self.edge_versions.clear()

        for t1, t2 in threshold_pairs:

            edge = cv2.Canny(
                self.gray_img,
                t1,
                t2
            )

            key = f"{t1}-{t2}"

            self.edge_versions[key] = edge

        return self.edge_versions

    def get_image(self, threshold1, threshold2):
        """
        Return a generated edge image.
        """

        key = f"{threshold1}-{threshold2}"

        return self.edge_versions.get(key)

    def get_all_images(self):
        """
        Return all generated edge images.
        """

        return self.edge_versions