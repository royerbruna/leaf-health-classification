
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from plant_leaf_classification.config import RANDOM_STATE


class DatasetManager:
    """Load and organize image metadata from the PlantVillage dataset.  """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_images(self) -> pd.DataFrame:
        """Go through the PlantVillage folders and collect information about each image. The species and original class are identified from the folder name, and the image path is saved for later processing."""

        images = []

        for class_dir in self.dataset_path.iterdir():
            if not class_dir.is_dir():
                continue

            species = class_dir.name.split("___")[0]

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    images.append({
                        "image_path": str(image_path),
                        "species": species,
                        "original_class": class_dir.name,
                    })

        return pd.DataFrame(images)

    def map_binary_labels(self, images: pd.DataFrame) -> pd.DataFrame:
        """Convert the original plant classes into two categories: healthy (0) and diseased (1). The classification is based on the original class name, making the dataset suitable for binary classification."""

        images["binary_label"] = (
            ~images["original_class"].str.endswith("___healthy")
        ).astype(int)

        return images


    def balance_binary_classes(self, images: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by keeping the same number of healthy and diseased images. The smaller class determines how many images are selected from each group. Images are chosen randomly, using a fixed seed to ensure the same selection every time."""

        counts = images["binary_label"].value_counts()

        if not {0, 1}.issubset(counts.index):
            raise ValueError("Both classes are required for balancing.")

        min_count = counts.min()

        balanced = images.groupby(
            "binary_label", group_keys=False
        ).sample(n=min_count, random_state=RANDOM_STATE)

        return balanced.sort_values("image_path").reset_index(drop=True)


    def create_stratified_folds(self, images: pd.DataFrame):
            """Split the dataset into five groups (folds) while preserving the proportion of healthy and diseased images. In each round, four groups are used for training and one for validation. A fixed seed ensures the same divisions are used in every experiment."""

            kfold = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            return list(kfold.split(images, images["binary_label"]))
