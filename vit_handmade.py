import torch
import torch.nn as nn
import torch.nn.functional as F

def num_patches(img_size, patch_size):

    num_patches_per_size = img_size // patch_size
    total_patches = num_patches_per_size**2

    return num_patches_per_size, total_patches


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_paches = (self.img_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_paches, self.embed_dim, kernel_size=self.patch_size, stride=patch_size)

    def forward(self, x):
        x_flattened = self.projection(x)
        x_out = x.flatten(1)
        x_out = x.transpose(1, 2)
        return x_out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # questi 3 layer lineari fanno la proiezione in Q,K,V, ogni patch diventa 3 vettori diversi 
        # i pesi W dei linear sono chiaramente diversi per Q,K,V, e vengono addestrati durante il training conforntando le label
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        # definisci i layer Q,K,V

    def forward(self, x):
        # x: [B, n_patches, embed_dim]
        # calcola Q,K,V
        Q = self.Q(x)  # shape [B, n_patches, embed_dim]
        K = self.K(x)  # shape [B, n_patches, embed_dim]
        V = self.V(x)  # shape [B, n_patches, embed_dim]
        # calcola matrice di attenzione e moltiplica per V
        # attention score[b,i,j], dove l'elemento i indica quanto il patch i guardi il patch j, cioè quanto è affine nello spazio proiettato
        # faccio la trasposta dell'ultimo e del penultimo asse
        attention_score = Q @ K.transpose(-2,-1)
        attention_score = attention_score / (self.embed_dim)**0.5
        attention_probs = torch.softmax(attention_score)
        return attention_probs  # shape [B, n_patches, embed_dim]


class FeedFoward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        # super serve perchè ogni volta che eredito dalla classe padre come nn.Module, 
        # chiamo il costruttore della superclasse, cosi quando inserisco attributi nella sottoclasse nn.Module sa come gestirli e inizializzarli
        super().__init__()
        # sequential serve solo per evitare di dover scrivere tutti i layer nel forward ma li impacchetto dentro a Sequential
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)  
            )
        
    
    def forward(self, x):
        return self.net(x)

# il transformer è composto da piu blocchi identici del TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.attention_mechanism = SelfAttention(embed_dim)
        self.fforw = FeedFoward(embed_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # meccanismo di attenzione
        x = x + self.attention_mechanism(self.ln1(x))
        # feed forward
        # x + ... → Residual / Skip Connection, evitare il problema del vanishing gradient o exploding gradient
        # che si genera perchè il gradiente calcolato durante il backpropagation può diventare troppo piccolo o troppo grande essendo 
        # un prodotto di derivate costruito con la chain rule 
        # con le normalizzazioni layer norm prima di ogni blocco, con le skip connections, con le ttivazioni relu o gelu
        # si possono usare anche ottimizzatori come adam che adattano il learning rate durante il training
        # Somma l’output dell’attenzione con l’input originale x
        # Questo aiuta a:
        # - Stabilizzare il gradiente
        # - Conservare l’informazione originale
        # - Rendere il training più veloce e profondo
        x = x + self.fforw(self.ln2(x))

        return x

if __name__ == "__main__":
    print(num_patches(224, 16))  # Expected output: 196