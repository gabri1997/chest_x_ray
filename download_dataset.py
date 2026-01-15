import logging

from huggingface_hub import snapshot_download
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def download():
    repo_id = "BahaaEldin0/NIH-Chest-Xray-14"
    local_dir = "data/NIH_ChestXray14"
    logging.info("Starting download for %s into %s", repo_id, local_dir)
    try:
        snapshot_download(
        repo_id="BahaaEldin0/NIH-Chest-Xray-14",
        repo_type="dataset",               
        local_dir="data/NIH_ChestXray14",
        local_dir_use_symlinks=False,
    )
        logging.info("Download completed successfully.")
    except Exception:
        logging.exception("Failed to download the dataset")
        raise

def class_vocabulary():
    classes = set()
    chest_ds = load_dataset (
        "parquet",
        data_files = {"test": "data/NIH_ChestXray14/data/test-*.parquet"},
    )

    print(f"Queste sono le features: ", chest_ds["test"].features)
    for example in chest_ds["test"]:
        for label in example["label"]:
            classes.add(label)
    
    print("Class vocabulary:")
    for cls in sorted(classes):
        print(cls)
    print(f"Total number of classes: {len(classes)}")

    """Class vocabulary:
        Atelectasis
        Cardiomegaly
        Consolidation
        Edema
        Effusion
        Emphysema
        Fibrosis
        Hernia
        Infiltration
        Mass
        No Finding
        Nodule
        Pleural_Thickening
        Pneumonia
        Pneumothorax
        Total number of classes: 15"""
    
    # Ogni immagine → vettore y ∈ {0,1}¹⁵


if __name__ == "__main__":
    #download()
    class_vocabulary()
