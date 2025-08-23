import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecEncoder(nn.Module):
    """
    A simple wrapper for the Wav2Vec2Model that returns only the encoder's
    last hidden state. This is designed for feature extraction.
    """
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

    def forward(self, waveform):
    
        # The model returns a tuple of outputs, we only want the last hidden state
        outputs = self.wav2vec2(waveform)
        return outputs.last_hidden_state


class Wav2VecClsPooler(nn.Module):
    """
    A pooling module that uses a CLS token and a shallow transformer encoder
    to learn a fixed-size representation from Wav2Vec2 encoder outputs.

    Args:
        input_size (int): The hidden size of the Wav2Vec2 encoder output (e.g., 768).
        num_layers (int): The number of transformer encoder layers.
        nhead (int): The number of attention heads in the transformer.
    """
    def __init__(self, hidden_dim=768, num_layers=2, nhead=8,projection_dim=128):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))

        self.Wav2VecEncoder = Wav2VecEncoder()
        #turn these gradients off
        for param in self.Wav2VecEncoder.parameters():
            param.requires_grad = False

        transformer_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=1536,
            dropout=0.1,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        self.projection_head = nn.Sequential(nn.Linear(hidden_dim,hidden_dim*2),
                                             nn.BatchNorm1d(hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim*2,hidden_dim*2),
                                             nn.BatchNorm1d(hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim*2,projection_dim))

    def forward(self,x,return_features=False):
        
        x = self.Wav2VecEncoder(x)
        x = self.transformer_encoder(torch.cat([self.cls_token.expand(x.shape[0],-1,-1),x],dim=1))        
        
        if return_features:
            return x[:,0],self.projection_head(x[:,0])
        
        else:
            return self.projection_head(x[:,0])
