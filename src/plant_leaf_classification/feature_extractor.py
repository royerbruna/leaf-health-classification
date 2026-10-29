
import cv2
import numpy as np


class FeatureExtractor:
    """Extract numerical color features from leaf images."""

    def extract_rgb_basic(self, image_path: str) -> dict:
        """Read an image and convert it from BGR to RGB. Calculate the mean and standard deviation of the red, green, and blue channels, producing six numerical features that describe the image colors."""
        
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        means = image.mean(axis=(0, 1))
        stds = image.std(axis=(0, 1))

        return {
            "r_mean": means[0],
            "g_mean": means[1],
            "b_mean": means[2],
            "r_std": stds[0],
            "g_std": stds[1],
            "b_std": stds[2],
        }


    def extract_rgb_percentiles(self, image_path: str) -> dict:
        """Extract the mean, standard deviation, and percentiles (25, 50, and 75) of each RGB channel. The percentiles provide additional information about how pixel intensities are distributed, producing a total of fifteen numerical features."""

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        means = image.mean(axis=(0, 1))
        stds = image.std(axis=(0, 1))
        percentiles = np.percentile(image, [25, 50, 75], axis=(0, 1))

        features = {}

        for index, channel in enumerate(("r", "g", "b")):
            features[f"{channel}_mean"] = means[index]
            features[f"{channel}_std"] = stds[index]

            for p_index, p in enumerate((25, 50, 75)):
                features[f"{channel}_p{p}"] = percentiles[p_index, index]

        return features


    def extract_rgb_histogram(self, image_path: str) -> dict:
        """Extract basic RGB statistics and color histograms from an image. Each color channel is divided into sixteen intensity ranges, and the number of pixels in each range is converted into a relative frequency. Together, these measurements produce fifty-four numerical features."""

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        means = image.mean(axis=(0, 1))
        stds = image.std(axis=(0, 1))

        features = {}

        for index, channel in enumerate(("r", "g", "b")):
            features[f"{channel}_mean"] = means[index]
            features[f"{channel}_std"] = stds[index]

            histogram, _ = np.histogram(
                image[:, :, index], bins=16, range=(0, 256)
            )

            histogram = histogram / (image.shape[0] * image.shape[1])

            for bin_index, value in enumerate(histogram):
                features[f"{channel}_hist_{bin_index}"] = value

        return features
