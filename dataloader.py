import logging
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from datasets import load_dataset
import numpy as np 
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ChestXrayTorchDataset(Dataset):
    def __init__(self, hf_dataset, class_idx, num_classes, transform=None):
        self.hf_dataset = hf_dataset
        self.class_idx = class_idx
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx): 
        """ Dato un indice idx, restituisco l'immagine e il vettore multi-hot encoded delle label """
        item = self.hf_dataset[idx]
        image = item["image"]
        labels = item["label"]  # Lista di label testuali
        y = np.zeros((self.num_classes,), dtype=np.float32)
        for label in labels:
            label_idx = self.class_idx[label]
            # Multi-hot encoding essendo un problema multi-label più valori possono essere 1
            y[label_idx] = 1
        # Le immagini sono in scala di grigi, le converto in RGB per usare modelli pre-addestrati su ImageNet
        image = image.convert("RGB")  
        if self.transform:
            image = self.transform(image)
        return image, y

class chest_xray_dataset_loader:
    def __init__(self):
        self.repo_id = "BahaaEldin0/NIH-Chest-Xray-14"
        self.local_dir = "data/NIH_ChestXray14"
        self.classes = set()
        self.class_idx = {}
        self.num_classes = 0
        self.chest_ds_train = None
        self.chest_ds_val = None
        self.chest_ds_test = None

    def download(self):
        logging.info("Starting download for %s into %s", self.repo_id, self.local_dir)
        try:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",               
                local_dir=self.local_dir,
                local_dir_use_symlinks=False,
            )
            logging.info("Download completed successfully.")
        except Exception:
            logging.exception("Failed to download the dataset")
            raise

    def load_dataset(self):
        """
        Qui creo i dati Arrow per la cache: 
        I dati Arrow sono tabelle binarie columnar, memory-mapped, 
        che contengono metadata e riferimenti ai dati (immagini incluse), non le immagini decodificate,
        le immagini vengono decodificate solo on-demand (dataset[i]).
        """
        chest_ds = load_dataset (
            "parquet",
            data_files = {
                "train": f"{self.local_dir}/data/train-*.parquet", 
                "validation": f"{self.local_dir}/data/valid-*.parquet", 
                "test": f"{self.local_dir}/data/test-*.parquet"
            },
        )
        self.chest_ds_train = chest_ds["train"]
        self.chest_ds_val = chest_ds["validation"]
        self.chest_ds_test = chest_ds["test"]

        print(f"Queste sono le features: ", self.chest_ds_test.features)
        for example in self.chest_ds_test:
            for label in example["label"]:
                self.classes.add(label)
        
        print("Class vocabulary:")
        for cls in sorted(self.classes):
            print(cls)
        
        self.num_classes = len(self.classes)
        print(f"Total number of classes: {self.num_classes}")
        self.classes = sorted(self.classes)
        self.class_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print("Class to index mapping:", self.class_idx)
        return self.chest_ds_train, self.chest_ds_val, self.chest_ds_test, self.classes, self.class_idx, self.num_classes



# Questa funzione non serve è ridondante con il metodo __getitem__ della classe ChestXrayTorchDatasets
def multi_hot_encode_labels(classes, class_idx, num_classes):
    """Qui vorrei creare una funzione che data una lista di label testuali mi restituisca un vettore one-hot encoded"""
    y = np.zeros((num_classes,), dtype=np.float32)
    for label in classes:
        idx = class_idx[label]
        y[idx] = 1
    print("One-hot encoded vector:", y)
    return y

def compute_mean_and_std(dataset):

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        mean = 0.0
        std = 0.0
        num_images = 0.0

        for images, _ in loader:
            # la shape di images è [B,C,H,W] che è l'output del dataloader
            batch_size = images.size(0)
            # faccio una view così mettendo -1 posso collassare H e W e creare una sola dimensione
            images = images.view(batch_size, images.size(1), -1)
            # ora images è [B, C, H*W], essendo che a me serve la media per ogni canale devo arrivare a ottenere (C,) come shape
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            num_images += batch_size

        final_mean = mean/num_images
        final_std = std/num_images

        return final_mean.numpy(), final_std.numpy()

if __name__ == "__main__":

    dataset = chest_xray_dataset_loader()
    chest_ds_train, chest_ds_val, chest_ds_test, classes, class_idx, num_classes = dataset.load_dataset()
    train_dataset = ChestXrayTorchDataset(chest_ds_train, class_idx, num_classes, transform=transforms.ToTensor())
    final_mean, final_std = compute_mean_and_std(train_dataset)
    print("Computed mean: ", final_mean)
    print("Computed std: ", final_std)
    # Computed mean:  [0.4980974 0.4980974 0.4980974]
    # Computed std:  [0.22967155 0.22967155 0.22967155]
    # Transform per il training
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=final_mean, std=final_mean)
    ])

    # Transform per validation e test
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=final_mean, std=final_std),
    ])
    train_dataset = ChestXrayTorchDataset(chest_ds_train, class_idx, num_classes, transform=transform_train)
    val_dataset = ChestXrayTorchDataset(chest_ds_val, class_idx, num_classes, transform=transform_val)
    test_dataset = ChestXrayTorchDataset(chest_ds_test, class_idx, num_classes, transform=transform_val)
    print(f"Number of training samples: {len(train_dataset)}")
    # Qui hardcodiamo le 15 classi per non fare ogni volta il parsing del dataset
    classes = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]
    k = len(classes)
    print(f"Classes: {classes}, Total: {k}")
    class_to_idx = {cls:i for i, cls in enumerate(classes)}
    print(f"Number of training samples: {len(train_dataset)}")  
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    # Esempio di utilizzo del dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break  # Solo per dimostrazione, rimuovere in allenamento reale
    
    