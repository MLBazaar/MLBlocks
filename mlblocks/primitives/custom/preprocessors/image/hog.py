import math

import numpy as np
from skimage.feature import hog


class HOG(object):

    def __init__(self, num_orientations, num_cell_pixels, num_cells_block,
                 img_dimension_x=None, img_dimension_y=None):

        self.num_orientations = num_orientations
        self.num_cell_pixels = num_cell_pixels
        self.num_cells_block = num_cells_block
        self.img_dimension_x = img_dimension_x
        self.img_dimension_y = img_dimension_y

    def make_hog_features(self, X):
        """Call the transform function of the HOG primitive.

        NOTE: Get a "ValueError: Negative dimensions" with some settings
        of the hyperparameters.
        """
        if math.sqrt(X.shape[1]).is_integer():
            # We can set dimensions if the image is square (default).
            img_dim = int(math.sqrt(X.shape[1]))
            self.img_dimension_x = img_dim
            self.img_dimension_y = img_dim
        else:
            if not self.img_dimension_x or not self.img_dimension_y:
                raise Exception(
                    "Must specify image dimensions for non-square image")

        def make_hog(image):
            image = image.reshape((self.img_dimension_x, self.img_dimension_y))
            features = hog(
                image,
                orientations=self.num_orientations,
                pixels_per_cell=(self.num_cell_pixels, self.num_cell_pixels),
                cells_per_block=(self.num_cells_block, self.num_cells_block),
                block_norm='L2-Hys',
                visualise=False
            )
            return features

        return np.apply_along_axis(lambda x: make_hog(x), axis=1, arr=X)
