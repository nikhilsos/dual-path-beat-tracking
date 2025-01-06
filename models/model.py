import torch
import torch.nn as nn
from ablation_models.frontend import BeatThis


class residual_block(nn.Module):
    def __init__(self, i, in_channels, num_filter, kernel_size, dropout):
        super(residual_block, self).__init__()
        self.res = nn.Conv1d(in_channels=in_channels, out_channels=num_filter, kernel_size=1, padding='same')
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=num_filter, kernel_size=kernel_size, dilation=i, padding='same')
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=num_filter, kernel_size=kernel_size, dilation=i*2, padding='same')
        self.elu = nn.ELU()
        self.spatial_dropout = nn.Dropout2d(p=dropout)
        self.conv_final = nn.Conv1d(in_channels=num_filter * 2, out_channels=num_filter, kernel_size=1, padding='same')

    def forward(self, x):
        #x: (B, F, T)
        x_res = self.res(x)
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.elu(x).unsqueeze(-1) #(B, F, T, 1)
        x = self.spatial_dropout(x).squeeze(-1) #(B, F, T)
        x = self.conv_final(x)
        return x + x_res, x


class TCN(nn.Module):
    def __init__(self, num_layers=11, dropout=.1, kernel_size=5, n_token=2):
        super(TCN, self).__init__()
        self.nlayers = num_layers

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=(2, 0))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 8), padding=(0, 0))  
        self.dropout3 = nn.Dropout(p=dropout)

        self.tcn_layers = nn.ModuleDict({})
        for layer in range(num_layers):
            self.tcn_layers[f'TCN_layer_{layer}'] = residual_block(i=2**layer, in_channels=128, num_filter=128, kernel_size=kernel_size, dropout=dropout)

        self.out_linear = nn.Linear(128, n_token)
        # self.out_linear_2 = nn.Linear(128, n_token)

        self.dropout_t = nn.Dropout(p=.5)
        self.out_linear_t = nn.Linear(128, 300)
        self.alpha = nn.Parameter(torch.tensor(0.5))


        self.beatthis = BeatThis(transformer_dim=128, ff_mult=4, n_layers=6, head_dim=8, dropout={"frontend" :0.1, "transformer": 0.2})
                                
    def forward(self, x):
        # x: spectrogram of size (B, T, mel_bin)
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = nn.ELU()(x)
        x = self.dropout1(x)
        
    
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = nn.ELU()(x)
        x = self.dropout2(x)
        
        

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = nn.ELU()(x)

        x = x.squeeze(-1) #(B, 20, T)
        print('after convo', x.shape)
       

        beatthisres = self.beatthis(x.transpose(1,2))
        

        t = []
        for layer in range(self.nlayers):
            x, skip = self.tcn_layers[f'TCN_layer_{layer}'](x)  #x: B, 20, T; skip: B, 20, T
            t.append(skip)

        x = torch.relu(x).transpose(-2, -1)

     
       
        x_combined = self.alpha * x + (1 - self.alpha) * beatthisres 
        # on hindsight, maybe hadamard product would've been a better feature aggregator
        x_combined = x_combined.unsqueeze(0) 

        x = self.out_linear(x_combined)
     

    

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=-1) #(batch, 20)
        t = self.out_linear_t(t)

        return x, t
    

