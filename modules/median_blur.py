import cv2


class MedianBlurTool:
    """
    Median Blur Processing Class
    """

    def __init__(self):
        self.original_img = None
        self.blurred_versions = {}
        self.selected_kernel = None

    def load_image(self, image):
        """
        Load image from Streamlit.
        """
        self.original_img = image
        self.blurred_versions.clear()

    def blur_image(self, kernel_size):
        """
        Apply Median Blur
        """

        # Kernel size must be odd and greater than 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size < 3:
            kernel_size = 3

        blurred = cv2.medianBlur(
            self.original_img,
            kernel_size
        )

        self.blurred_versions[kernel_size] = blurred

        return blurred

    def generate_multiple(self, kernels):
        """
        Generate multiple blurred images.
        """

        self.blurred_versions.clear()

        for kernel in kernels:

            if kernel % 2 == 0:
                kernel += 1

            if kernel < 3:
                kernel = 3

            self.blurred_versions[kernel] = cv2.medianBlur(
                self.original_img,
                kernel
            )

        return self.blurred_versions

    def get_image(self, kernel):
        """
        Return blurred image for selected kernel.
        """

        return self.blurred_versions.get(kernel)

    def get_all_images(self):
        """
        Return all generated blur images.
        """

        return self.blurred_versions