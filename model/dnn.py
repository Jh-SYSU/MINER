import torch.nn as nn

class DNN(nn.Module):
    
    def __init__(self, input_dim, emb_dim, num_layer, output_dim, drop_ratio):
        
        super(DNN, self).__init__()
        self.input_layer = nn.Sequential(
                                    nn.BatchNorm1d(input_dim), 
                                    nn.Linear(input_dim, emb_dim),
                                    nn.LeakyReLU()
                                    )
        
        self.hidden_block = []
        for i in range(num_layer-1):
            self.hidden_block += [
                                nn.BatchNorm1d(emb_dim),
                                nn.Dropout(drop_ratio), 
                                nn.Linear(emb_dim, emb_dim), 
                                nn.LeakyReLU()
                            ]   
        
            
        self.hidden_layer = nn.Sequential(*self.hidden_block)
        
        self.output_layer = nn.Sequential(
                                    nn.BatchNorm1d(emb_dim),
                                    nn.Dropout(drop_ratio),
                                    nn.Linear(emb_dim, output_dim)
                                    )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

