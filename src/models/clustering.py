"""
MLP classifiers used for clustering. The 3 classes proceed in order 
- MLPClassifer: Exactly the name, specify the input and output dimensions and hidden blocks
- MLP_dict: For specifying a grid or cube of MLPs, specified as hyperparameters.
- Connected_Clusterer: For connecting the MLP dict to the representation backbone 
"""
import torch
import torch.nn as nn
from itertools import product

#### Classification head models ####

class MlpClassifier(nn.Module):
    """
    This class defines the standard MLP classifier that will be used as a 
    clustering head. The default is a 3 layer MLP with input dim = 512 and 
    hidden dim = 512. 
    """
    def __init__(self,input_dim=512,output_dim=10,
                 hidden_dim=512,num_hidden_blocks=1,
                 dropout_rate=0.25):
        super(MlpClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_blocks = num_hidden_blocks

        self.input_block = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)        
        )

        if num_hidden_blocks != 0:
            self.hidden_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ) for _ in range(num_hidden_blocks)
            ])

        self.num_hidden_blocks = num_hidden_blocks

        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim,output_dim),
        )
        
    def forward(self,x):

        if len(x.shape) > 2:
            x = x.reshape(x.size(0),-1)

        x = self.input_block(x)

        if self.num_hidden_blocks != 0:
            for block in self.hidden_blocks:
                x = block(x)

        x = self.output_block(x)

        return x
    

class MLP_dict(nn.Module):
    """
    This class is for generating a dictionary of MLPs, one for each 'head'. 
    We use this to specify a grid or cube (copy dimension) of MLPs based on a linear 
    range for lambdas and Ks. The keys for the dictionary are then tuples 
    lambda factor, we input it as 'lamb_factor' such that the training lambda is known by 
    lamb_factor ** key[0]. The output dimension of each MLP is key[1]. 
    of the form (lambda,K,copy_idx). So that we don't lose the 
    nn.ModuleDict is used requires strings as keys, 
    so we convert the keys to strings, but save them
    as a tuple in the keys variable. Optionally the user can specify desired 
    keys on runtime, but the default return is a dictionary of the same form
    as the heads but containing the predicted logits.
    """
    def __init__(self,
                 K_range,
                 lamb_range,
                 num_copies=1,
                 input_dim=512,
                 hidden_dim=512,
                 num_hidden_blocks=1,
                 dropout_rate=0.25,
                 lamb_factor=1.1):
        
        super(MLP_dict,self).__init__()
        
        self.K_range = K_range
        self.lamb_range = lamb_range
        self.num_copies = num_copies
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_blocks = num_hidden_blocks
        self.dropout_rate = dropout_rate
        self.lamb_factor = lamb_factor
        
        keys = list(product(lamb_range,K_range,range(num_copies)))
        self.keys = keys

        self.mlp_dict = nn.ModuleDict({
            str(key): MlpClassifier(input_dim=input_dim,
                                    output_dim=key[1],
                                    hidden_dim=hidden_dim,
                                    num_hidden_blocks=num_hidden_blocks,
                                    dropout_rate=dropout_rate) for key in keys
        })
        
        
    def forward(self,x,keys=None):
        
        if keys is None:
            p_dict = {key:self.mlp_dict[str(key)](x) for key in self.keys}
        else:
            p_dict = {key:self.mlp_dict[str(key)](x) for key in keys}
        
        return p_dict
    
class Connected_Clusterer(nn.Module):

    def __init__(self, backbone, clusterer):
        super(Connected_Clusterer, self).__init__()
        self.backbone = backbone
        self.clusterer = clusterer  
        self.keys = clusterer.keys
        self.lamb_factor = clusterer.lamb_factor
        
        self.backbone.eval()

    def forward(self, x,backbone_grads=False):

        if backbone_grads:
            z = self.backbone(x)
        else:
            with torch.no_grad():
                z = self.backbone(x)

        p = self.clusterer(z)
        
        return p


def _create_temporal_embeddings(self, max_seq_len, hidden_dim):
    """Create sinusoidal temporal positional embeddings"""
    pe = torch.zeros(max_seq_len, hidden_dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # Add batch dimension

class TransformerClassifier(nn.Module):

    def __init__(self,
                 input_dim=768,
                 hidden_dim=768,
                 dim_feedforward=1048,
                 num_layers=2,
                 num_classes=10,
                 dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()

        self.clf_token = nn.Parameter(torch.randn(1,1,hidden_dim))

        # Create temporal positional embeddings
        max_seq_len = 100  # Maximum sequence length
        self.register_buffer("temporal_embeddings", self._create_temporal_embeddings(max_seq_len, hidden_dim))
        
        self.projection_layer = nn.Linear(input_dim,hidden_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                            nhead=8,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=dropout_rate)
        
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer,
                                                         num_layers=num_layers,
                                                         batch_first=True
                                                         )
        
        self.classifier = nn.Linear(hidden_dim,num_classes)
        
    def forward(self,x):

       #project the input #potentially to use lower/higher dims
        x = self.projection_layer(x)

        #add the temporal embeddings to the input
        x = x + self.temporal_embeddings[:,:x.shape[1],:]

        #concat the clf token to the input
        x = torch.cat([self.clf_token.expand(x.shape[0],-1,-1),x],dim=1)

        #pass through the transformer
        x = self.transformer_encoder(x)

        predictions = self.classifier(x[:,0,:])

        return predictions
    

class Transformer_dict(nn.Module):
    def __init__(self,
                 K_range,
                 lamb_range,
                 num_copies=1,
                 input_dim=768,
                 hidden_dim=768,
                 dim_feedforward=1048,
                 num_layers=2,
                 dropout_rate=0.1):
        super(Transformer_dict, self).__init__()

        keys = list(product(lamb_range,K_range,range(num_copies)))
        self.keys = keys


        
        
        self.transformer_dict = nn.ModuleDict({
            str(key): TransformerClassifier(input_dim=input_dim,
                                            hidden_dim=hidden_dim,
                                            dim_feedforward=dim_feedforward,
                                            num_layers=num_layers,
                                            num_classes=key[1],
                                            dropout_rate=dropout_rate) for key in keys
        })

    def forward(self,x,keys=None):

        if keys is None:
            p_dict = {key:self.transformer_dict[str(key)](x) for key in self.keys}
        else:
            p_dict = {key:self.transformer_dict[str(key)](x) for key in keys}
        
        return p_dict
    
