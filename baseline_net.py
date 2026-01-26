import os
import json
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from vit import SimpleViT
from datetime import datetime
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import  ChestXrayTorchDataset
from dataloader import chest_xray_dataset_loader
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 42
set_seed(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # oppure ":16:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

g = torch.Generator()
g.manual_seed(seed)

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "no_slurm_id")
print("SLURM Job ID:", SLURM_JOB_ID)


class Densenet(nn.Module):

    def __init__(self, num_classes):
        super(Densenet, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier.in_features
        # il Linear è una logistic regression che mappa le feature estratte dalla densenet alle classi di output
        # Logistic regression è un modello per classificazione che trasforma una combinazione lineare delle feature in una probabilità.
        # ho un vettore di feature x in ingresso , pesi e bias b, e voglio predire la probabilità di ciascuna classe, genero lo score lineare z = w⋅x + b
        # poi uso la sigmoide per ottenre una probabilità p = sigmoid(z) = 1 / (1 + exp(−z)) intervallo [0, 1], intepretabile come probabilità della classe positiva dato x
        # si allena con la loss, in particolare la binary cross entropy tra p e la label vera y (0 o 1), essendo un problema multilabel uso BCEWithLogitsLoss che combina sigmoid + BCE in modo numericamente stabile
        # e avendo 15 classi ne fccio 15 in parallelo, quindi logistic regression = Linear + BCE, fatte sulle features della densenet. 
        self.model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

class myNet():
    def __init__(self, description, report_path, num_classes, num_epochs, num_workers, device, learning_rate=1e-4, weight_decay=1e-5, momentum=0.9, step_size=7, gamma=0.1, net_type='densenet', batch_size=32, loss_function='focal'):
        self.description = description
        self.report_path = report_path
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.device = device
        self.learning_rate = learning_rate
        # il weight decay è una forma di regolarizzazione L2 che aiuta a prevenire l'overfitting tenendo i pesi bassi 
        # altrimenti il modello potrebbe adattarsi troppo ai dati di training e non generalizzare bene sui dati nuovi
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.net_type = net_type
        self.batch_size = batch_size
        self.loss_function = loss_function

        # wandb
        wandb.init(project="Chest X-ray Classification", config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "step_size": step_size,
            "gamma": gamma,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "net_type": net_type
        })

        # questi valori vengono calcolati prima dalla classe dataloader.py compute_mean_and_std
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
        ])

        # Transform per validation e test
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
        ])

        if net_type == 'densenet':
            print("Inizializzo il modello Densenet...")
            self.model = Densenet(num_classes=num_classes).to(self.device)
        elif net_type == 'vit':
            print("Inizializzo il modello ViT...")
            self.model = SimpleViT(num_classes=num_classes, img_size=224, patch_size=16, in_channels=3,
                                embed_dim=256, num_heads=4, depth=4, mlp_dim=512).to(self.device)

        dataset = chest_xray_dataset_loader()
        chest_ds_train, chest_ds_val, chest_ds_test, classes, class_idx, num_classes = dataset.load_dataset()
        self.train_dataset = ChestXrayTorchDataset(chest_ds_train, class_idx, num_classes, transform=transform_train)
        self.val_dataset = ChestXrayTorchDataset(chest_ds_val, class_idx, num_classes, transform=transform_val)
        self.test_dataset = ChestXrayTorchDataset(chest_ds_test, class_idx, num_classes, transform=transform_val)
        print(f"Number of training samples: {len(self.train_dataset)}")

        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True, generator=g)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True, generator=g)
        self.test_loader = DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)

        # qui sto allenando tutti i parametri del modello (densenet + classifier) perchè self.model.parameters() include tutti i parametri della rete, quindi sto facendo fine tuning completo
        if net_type == 'densenet':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif net_type == 'vit':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # qua definisco la loss function per il problema di classificazione multilabel
        # BCE(p,y)=−[y⋅log(p)+(1−y)⋅log(1−p)]
        # se una classe è molto rara, la BCE potrebbe non dare abbastanza peso agli errori su quella classe, non penalizza abbastanza il modello
        if self.loss_function == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = None  # useremo self.focal_loss


        # la focal loss agiunge un fattore modulante alla BCE per dare più peso agli esempi difficili e rari, FL(p,y)=−α(1−p)γylog(p)−αpγ(1−y)log(1−p)
        # lo schedueler per il learning rate, che lo riduce ogni step_size epoch moltiplicandolo per gamma
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
   
        K = self.num_classes  # 15
        self.auroc = MultilabelAUROC(num_labels=K, average="macro").to(self.device)

        # usando micro averaging, calcolo le metriche globalmente sommando i veri positivi, falsi positivi e falsi negativi attraverso tutte le classi
        # in questo caso se una classe è molto comune cioè ha molti esempi, influenzerà di più la metrica complessiva rispetto a una classe rara, tirando la metrica verso le performance sulle classi più frequenti

        self.acc_micro = MultilabelAccuracy(num_labels=K, threshold=0.5, average="micro").to(self.device)
        self.prec_micro = MultilabelPrecision(num_labels=K, threshold=0.5, average="micro").to(self.device)
        self.rec_micro = MultilabelRecall(num_labels=K, threshold=0.5, average="micro").to(self.device)
        self.f1_micro = MultilabelF1Score(num_labels=K, threshold=0.5, average="micro").to(self.device)
        self.f1_macro = MultilabelF1Score(num_labels=K, threshold=0.5, average="macro").to(self.device)

    # alla fine è una BCE con un fattore in più che dà più peso agli esempi difficili
    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1 - p_t) ** gamma
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = modulating_factor * alpha_factor * bce_loss
        return focal_loss.mean()

    def mixup_data(self, x, y, alpha=0.2, prob=0.5):
        # se faccio più del 50% di probabilità allora non applico mixup
        if np.random.rand() > prob:
            return x, y
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y

    def write_report_results(self, metrics, saving_epoch):
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_id")

        LOG = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_job_id": slurm_job_id,
            "description": self.description,
            "num_classes": self.num_classes,
            "num_epochs": self.num_epochs,
            "num_workers": self.num_workers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "net_type": self.net_type,
            "optimizer": str(self.optimizer),
            "scheduler": str(self.scheduler),
            "loss_function": str(self.loss_function),
            "saving_epoch_best_model": saving_epoch,
            "results_metrics": {
                "Accuracy(micro)" : metrics['accuracy_micro'],
                "Precision(micro)" : metrics['precision_micro'],
                "Recall(micro)": metrics['recall_micro'],
                "Val loss" : metrics['val_loss'],
                "AUROC(macro)" : metrics['auroc_macro'],
                "F1(micro)" : metrics['f1_micro'],
                "F1(macro)" : metrics['f1_macro'],
                "Threshold" : metrics['threshold'],
            }
        }

        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        # Sovrascrive sempre il file con l'ultimo esperimento
        with open(self.report_path, "w") as f:
            json.dump(LOG, f, indent=4)

    def train(self):
        patience = 10
        best_auroc = 0.0
        current_waiting_time = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            # monitorando l'AUROC per implementare l'early stopping
            current_loss_value = 0.0
            for image, lbl in self.train_loader:
               
               # non_blocking=True per migliorare le prestazioni con pin_memory=True nel DataLoader, rende la copia in GPU asincrona cosi la CPU non aspetta
               # in questo punto il batch dalla RAM viene copiato nella VRAM della GPU e diventa un tensor Cuda, tutto il batch non l'intero dataset chiaramente 
               image = image.to(self.device, non_blocking=True)

               # se uso mixup augmentation, perchè la loss diventa continua essendo il target continuo tra 0 e 1:  
               lbl = lbl.to(self.device, non_blocking=True).float()
               image,lbl = self.mixup_data(image, lbl, alpha=0.2, prob=0.5)
               # se non uso mixup:
               # lbl = lbl.to(self.device, non_blocking=True)


               # azzero i gradienti perchè Pytorch accumula i gradienti ad ogni backward() per default
               # mi serve per cambiare i pesi ad ogni batch
               self.optimizer.zero_grad()
               outputs = self.model(image)
               # calcolo della loss
               if self.loss_function == "bce":
                    loss = self.criterion(outputs, lbl.float())
               else:
                    loss = self.focal_loss(outputs, lbl.float())

               # calcolo dei gradienti, per ogni parametri w, ottengo la derivata della loss rispetto a w, questi gradienti vengono salvati in w.grad
               loss.backward()
               # update dei pesi con regola w ← w − (lr⋅∇w​loss) esempio per SGD base
               # con il training, calcolo come cambiare w per ridurre la loss, dove ∇w​loss è il gradiente e mi dice in che direzione modificare i pesi
               self.optimizer.step()
               current_loss_value += loss.item()
            # aggiorna il learning rate secondo lo scheduler durante l'allenamento
            # aggiorno l'iperparametro learning rate ogni epoch, spesso conviene partire con lr alto per fare i primi progressi rapidi
            # e poi 'rifinire' la soluizione con lr più bassi senza oscillare nel intorno al minimo
            # quindi cambio il learning rate per guidare l'optimizer
            self.scheduler.step()  
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {current_loss_value/len(self.train_loader)}")

            print("Validating...")
            metrics = self.evaluate()
            print("Ho terminato l'evaluation")
            current_auroc_value = metrics["auroc_macro"]
            if current_auroc_value > best_auroc:
                best_auroc = current_auroc_value
                current_waiting_time = 0
                torch.save(self.model.state_dict(), f"/work/grana_far2023_fomo/ChestXray/models/best_model_{SLURM_JOB_ID}.pth")
                # scrivo in un file txt le metriche
                print("Scrivo le metriche del best model su file...")
                with open(f"/work/grana_far2023_fomo/ChestXray/models/best_model_metrics_{SLURM_JOB_ID}.txt", "w") as f:
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"epochs: {epoch+1}\n")
                print(f"Best model saved with AUROC(macro): {best_auroc:.4f}")
                self.write_report_results(metrics, saving_epoch=epoch+1)
            else:
                current_waiting_time += 1
                if current_waiting_time >= patience:
                    print("Early stopping triggered.")
                    break
        wandb.finish()


    @torch.no_grad()
    def evaluate(self):

        self.auroc.reset()
        self.acc_micro.reset()
        self.prec_micro.reset()
        self.rec_micro.reset()
        self.f1_micro.reset()
        self.f1_macro.reset()

        self.model.eval()

        metrics = {}
        tot_loss = 0
        tot_samples = 0
        threshold = 0.5

        # non aggiorno i gradienti durante la validazione per risparmiare memoria e calcolo
        with torch.no_grad():

            # per ciascun batch nel validation set
            for image, lbl in self.val_loader:

                # sposto i tensori in GPU
                image = image.to(self.device, non_blocking=True)
                lbl = lbl.to(self.device, non_blocking=True)

                n_samples = image.size(0)
                tot_samples += n_samples

                # calcolo il mio output
                outputs = self.model(image)
            
                loss = self.criterion(outputs, lbl.float())
                tot_loss += loss.item()*n_samples

                # converto gli output in probabilità con la sigmoid, essendo problema multilabel un sample può avere più classi -> non uso softmax che forza la somma delle probabilità a 1            
                probs = torch.sigmoid(outputs)                        

                # AUROC: usa probs (non binarie)
                self.auroc.update(probs, lbl.int())

                # Accuracy/Precision/Recall/F1: torchmetrics binarizza con threshold interno
                self.acc_micro.update(probs, lbl.int())
                self.prec_micro.update(probs, lbl.int())
                self.rec_micro.update(probs, lbl.int())
                self.f1_micro.update(probs, lbl.int())
                self.f1_macro.update(probs, lbl.int())

            avg_val_loss = tot_loss / max(tot_samples, 1)

            metrics = {
                "val_loss": avg_val_loss,
                "auroc_macro": self.auroc.compute().item(),
                "accuracy_micro": self.acc_micro.compute().item(),
                "precision_micro": self.prec_micro.compute().item(),
                "recall_micro": self.rec_micro.compute().item(),
                "f1_micro": self.f1_micro.compute().item(),
                "f1_macro": self.f1_macro.compute().item(),
                "threshold": threshold,
            }

            print(
                f"Accuracy(micro): {metrics['accuracy_micro']:.4f} | "
                f"Precision(micro): {metrics['precision_micro']:.4f} | "
                f"Recall(micro): {metrics['recall_micro']:.4f} | "
                f"Val loss: {metrics['val_loss']:.4f} | "
                f"AUROC(macro): {metrics['auroc_macro']:.4f} | "
                f"F1(micro): {metrics['f1_micro']:.4f} | "
                f"F1(macro): {metrics['f1_macro']:.4f}"
            )

            wandb.log({
                "Validation/Loss": metrics["val_loss"],
                "Validation/AUROC_macro": metrics["auroc_macro"],
                "Validation/Accuracy_micro": metrics["accuracy_micro"],
                "Validation/Precision_micro": metrics["precision_micro"],
                "Validation/Recall_micro": metrics["recall_micro"],
                "Validation/F1_micro": metrics["f1_micro"],
                "Validation/F1_macro": metrics["f1_macro"],
                "Validation/Threshold": threshold,
            })

            return metrics
        
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg           

if __name__ == '__main__':


    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--net_type', type=str, default='vit', choices=['densenet', 'vit'], help='Type of network to use: densenet or vit')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization) for the optimizer')
    parser.add_argument('--loss_function', type=str, default='focal', choices=['bce', 'focal'], help='Loss function to use: bce or focal')
    parser.add_argument('--description', type=str, default="Chest X-ray classification model", help='Description of the experiment')
    args = parser.parse_args()
    
    
    num_classes = 15
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_id")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_path = f"/work/grana_far2023_fomo/ChestXray/results_report/report_{slurm_job_id}.json"

    trainer = myNet(description=args.description, report_path = report_path, num_classes=num_classes, num_epochs=args.num_epochs, num_workers=4, device=device, net_type=args.net_type, batch_size=args.batch_size, loss_function=args.loss_function)
    trainer.train()


    # RIPASSO DEL GRADIENTE E DEL BACKPROPAGATION
    # suppongo che la loss dipenda da un singolo peso w : loss = f(w)
    # per aggiornare w, devo sapere come cambiare w per ridurre la loss
    # il gradiente ∇w​loss = d(loss)/d(w) mi dice come varia la loss al variare di w
    # se il gradiente è positivo, significa che aumentando w, la loss aumenta
    # quindi per ridurre la loss, devo diminuire w
    # se il gradiente è negativo, significa che aumentando w, la loss diminuisce
    # quindi per ridurre la loss, devo aumentare w
    # la regola di aggiornamento w ← w − (lr⋅∇w​loss) combina queste informazioni
    # lr è il learning rate, un iperparametro che controlla la dimensione del passo di aggiornamento
    # moltiplicando il gradiente per lr, controllo quanto grande sarà il cambiamento di w
    # sottraendo lr⋅∇w​loss da w, aggiorno w nella direzione che riduce la loss
    # in sintesi, il gradiente mi dice in che direzione e di quanto cambiare w per minimizzare la loss durante l'allenamento del modello
    # I pesi sono un vettore multidimensionale enorme ∇w​L=(∂w1​∂L​,∂w2​∂L​,...,∂wn​∂L​), ogni componente mi dice "se aumento quel peso un pochino, la loss sale o scende?" 
    # quindi il gradiente è anch'esso un vettore multidimensionale, il segno di ciascuna componente mi dice se aumentare o diminuire quel peso per ridurre la loss, (su/giu), la magnitude mi dice di quanto, se quel peso influenza molto la Loss
    # ----> loss.backward(), è la backpropagation,calcola tutti questi gradienti, cioè tutte le derivate parziali della loss rispetto ai pesi e li salva in w.grad per ogni peso w nel modello
    # ----> self.optimizer.step() usa questi gradienti per aggiornare tutti i pesi
    


    # Quando calcolo la loss con criterion(logits, y) ottengo un valore che misura l’errore del modello sul batch.
    # La loss è definita confrontando, per ogni sample e per ogni classe, le predizioni (logits/sigmoid) con le label (multi-hot).
    # I gradienti rispetto ai pesi non vengono calcolati dalla loss da sola: li ottengo con loss.backward(), che calcola 
    # ∂loss/∂w per ogni parametro del modello.
    # Questi gradienti indicano quanto ogni peso influenza la loss e in che direzione va aggiornato per ridurla.
    # I pesi sono condivisi tra tutti i sample: non esistono “pesi per sample”, ma output per sample e gradienti globali accumulati dal batch.

    # Ogni peso è una dimensione
    # Immagina che il tuo modello abbia n pesi. Possiamo pensare a ognuno come a una coordinata in uno spazio n-dimensionale:
    # w=[w1​,w2​,...,wn​]
    # Ogni punto in questo spazio rappresenta uno stato del modello, cioè un set completo di pesi.
    # Il gradiente indica una direzione
    # Quando calcoli loss.backward(), ottieni un gradiente per ciascun peso:
    # g=∂w∂loss​=[∂w1​∂L​,∂w2​∂L​,...,∂wn​∂L​]
    # Questo vettore g punta nella direzione dove la loss cresce più rapidamente.
    # Se vuoi ridurre la loss, ti muovi nella direzione opposta, aggiorno i pesi nella direzione opposta al gradiente:
    # w←w−ηg
    # Penalizzare un esempio difficile
    # La Focal Loss fa in modo che gli esempi difficili generino gradienti più grandi:
    # Per un esempio facile: gradiente piccolo → piccolo spostamento del vettore dei pesi.
    # Per un esempio difficile: gradiente grande → grande spostamento del vettore dei pesi.
    # In altre parole, la magnitudine del passo nello spazio dei pesi è modulata dalla difficoltà dell’esempio.
    
    
    