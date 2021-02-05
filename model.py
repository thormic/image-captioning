import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Embedding:
        torch.nn.init.kaiming_uniform_(m.weight)
    else:
        pass

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.num_layers, batch_first = True, dropout = 0.3)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        self.apply(init_weights)
    


            
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        
        captions = self.embed(captions)
        
        features = features.unsqueeze(1)
        
        inputs = torch.cat((features, captions), 1)
        outputs, _ = self.gru(inputs)
        
        outputs = self.fc(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        
        while len(output) <= max_len:
            
            outputs, states = self.gru(inputs, states)
            
            outputs = self.fc(outputs)
            
            prob, ind = torch.max(outputs, dim=2)
            
            word_ind = ind.item()
            output.append(word_ind)
            
            inputs = self.embed(ind)
            
            if word_ind == 1:
                break
            
        
        return output