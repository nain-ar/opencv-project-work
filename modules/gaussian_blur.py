import cv2


class GaussianBlurTool:
    """
    Gaussian Blur Processing Class
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

    def blur_image(self, kernel_size, sigma=0):
        """
        Apply Gaussian Blur.
        """

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size < 3:
            kernel_size = 3

        blurred = cv2.GaussianBlur(
            self.original_img,
            (kernel_size, kernel_size),
            sigma
        )

        self.blurred_versions[kernel_size] = blurred

        return blurred

    def generate_multiple(self, kernels, sigma=0):
        """
        Generate multiple blurred images.
        """

        self.blurred_versions.clear()

        for kernel in kernels:

            if kernel % 2 == 0:
                kernel += 1

            if kernel < 3:
                kernel = 3

            self.blurred_versions[kernel] = cv2.GaussianBlur(
                self.original_img,
                (kernel, kernel),
                sigma
            )

        return self.blurred_versions

    def get_image(self, kernel):
        """
        Return one generated image.
        """

        return self.blurred_versions.get(kernel)

    def get_all_images(self):
        """
        Return all generated images.
        """

        return self.blurred_versions