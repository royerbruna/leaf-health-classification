
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class DatasetManager:
    """Load and organize image metadata from the PlantVillage dataset.  """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_images(self) -> pd.DataFrame:
        """Collect image paths, species, and original class labels."""
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
        """Map healthy images to 0 and diaseased images to 1."""
        images["binary_label"] = (
            ~images["original_class"].str.endswith("___healthy")
        ).astype(int)

        return images


    def create_stratified_folds(self, images: pd.DataFrame):
        """Create five stratified folds with reproducible splits."""
        kfold = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        return list(kfold.split(images, images["binary_label"]))
