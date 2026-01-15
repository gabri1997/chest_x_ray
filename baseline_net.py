import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import  ChestXrayTorchDataset
from dataloader import chest_xray_dataset_loader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



class Densenet(nn.Module):

    def __init__(self, num_classes):
        super(Densenet, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

class myNet():
    def __init__(self, num_classes, num_epochs, num_workers, device, learning_rate=1e-4, weight_decay=1e-5, momentum=0.9, step_size=7, gamma=0.1, net_type='densenet', batch_size=32):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.net_type = net_type
        self.batch_size = batch_size
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Transform per validation e test
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.model = Densenet(num_classes=num_classes).to(self.device)

        dataset = chest_xray_dataset_loader()
        chest_ds_train, chest_ds_val, chest_ds_test, classes, class_idx, num_classes = dataset.load_dataset()
        self.train_dataset = ChestXrayTorchDataset(chest_ds_train, class_idx, num_classes, transform=transform_train)
        self.val_dataset = ChestXrayTorchDataset(chest_ds_val, class_idx, num_classes, transform=transform_val)
        self.test_dataset = ChestXrayTorchDataset(chest_ds_test, class_idx, num_classes, transform=transform_val)
        print(f"Number of training samples: {len(self.train_dataset)}")

        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def train(self):
        for epoch in range(self.num_epochs):
            current_loss_value = 0.0
            for image, lbl in self.train_loader:
               # non_blocking=True per migliorare le prestazioni con pin_memory=True nel DataLoader, rende la copia in GPU asincrona cosi la CPU non aspetta
               image = image.to(self.device, non_blocking=True)
               lbl = lbl.to(self.device, non_blocking=True)
               # azzero i gradienti perchè Pytorch accumula i gradienti ad ogni backward() per default
               # mi serve per cambiare i pesi ad ogni batch
               self.optimizer.zero_grad()
               outputs = self.model(image)
               # calcolo della loss
               loss = self.criterion(outputs, lbl.float())
               # calcolo dei gradienti, per ogni parametri w, ottengo la derivata della loss rispetto a w, questi gradienti vengono salvati in w.grad
               loss.backward()
               # update dei pesi con regola w ← w − (lr⋅∇w​loss) esempio per SGD base
               # con il training, calcolo come cambiare w per ridurre la loss, dove ∇w​loss è il gradiente e mi dice in che direzione modificare i pesi
               self.optimizer.step()
               current_loss_value += loss.item()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {current_loss_value/len(self.train_loader)}")

            # aggiorna il learning rate secondo lo scheduler durante l'allenamento
            # aggiorno l'iperparametro learning rate ogni epoch, spesso conviene partire con lr alto per fare i primi progressi rapidi
            # e poi 'rifinire' la soluizione con lr più bassi senza oscillare nel intorno al minimo
            # quindi cambio il learning rate per guidare l'optimizer
            self.scheduler.step()   




if __name__ == '__main__':

    num_classes = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Densenet(num_classes=num_classes)
    trainer = myNet(num_classes=num_classes, num_epochs=50, num_workers=4, device=device)
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
    # ----> loss.backward(), è la backpropagation,git calcola tutti questi gradienti, cioè tutte le derivate parziali della loss rispetto ai pesi e li salva in w.grad per ogni peso w nel modello
    # ----> self.optimizer.step() usa questi gradienti per aggiornare tutti i pesi
    
    
    
    