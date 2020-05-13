import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        
        # Set the hidden size for init_hidden
        self.hidden_size = hidden_size
        self.device = device
        # Embedded layer
        self.embed_captions = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first= True,
                            dropout = 0)
        
        # Fully Connected layer
        self.hidden2caption = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))
    
    def forward(self, features, captions):
        
        # Initialize the hidden state
        self.hidden = self.init_hidden(features.shape[0])# features is of shape (batch_size, embed_size)
        
        
        # Embedding the captions
        embedded_captions = self.embed_captions(captions[:,:-1])
       
        features_embedded_captions = torch.cat((features.unsqueeze(1), embedded_captions), dim=1)
        
        
        # LSTM
        lstm_out, self.hidden = self.lstm(features_embedded_captions, self.hidden)
        
        # Functional component
        out = self.hidden2caption(lstm_out)
        return out

      
                                                          


      
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption=[]
        for i in range(max_len):
            lstm_out,states = self.lstm(inputs,states)
            predctions=self.hidden2caption(lstm_out.squeeze(1))
            _,predected=predctions.max(1)
            caption.append(predected.item())
            inputs = self.embed_captions(predected)
            inputs = inputs.unsqueeze(1)
            
        return caption
            
            