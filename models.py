import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path



class NodeP(BasicModule):
    def __init__(self, batch_size, lstm_hid_dim, n_classes, label_embed):
        super(NodeP, self).__init__()
        self.n_classes = n_classes
        self.lstm_hid_dim = lstm_hid_dim
        self.first_layer = nn.Linear(self.lstm_hid_dim*2, self.lstm_hid_dim)
        self.label_embed = self.load_labelembedd(label_embed)
        self.lstm = torch.nn.LSTM(self.lstm_hid_dim*2, hidden_size=lstm_hid_dim, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.output_layer = torch.nn.Linear(lstm_hid_dim * 2, n_classes)
        self.emdropout = torch.nn.Dropout(p=0.3)
        self.convs1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.batch_size = batch_size


        self.nei_att=nn.Linear(self.lstm_hid_dim*2,self.lstm_hid_dim)
        self.label_layer = nn.Linear(50, self.lstm_hid_dim)
        self.final_layer = nn.Linear(self.lstm_hid_dim, 75)
        self.output_layer = nn.Linear(75, self.n_classes)
        self.edropout = torch.nn.Dropout(p=0.3)

    def load_labelembedd(self, label_embed):
        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.lstm_hid_dim).cuda(),
                torch.randn(2, self.batch_size, self.lstm_hid_dim).cuda())
    def forward(self,x,neighbor_emb):
        x=self.edropout(x)
        neighbor_emb=self.edropout(neighbor_emb)
        label = self.label_layer(self.label_embed.weight.data)
        embs_batch = self.first_layer(x)
        embs1 = embs_batch.unsqueeze(-1).transpose(1, 2)
        embs_batch = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), embs1.transpose(1, 2))
        embs_batch = torch.bmm(embs_batch, embs1)
        hidden_state = self.init_hidden()
        outputs, hidden_state = self.lstm(neighbor_emb, hidden_state)
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :,self.lstm_hid_dim:]
        m1 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        label_att = torch.cat((torch.bmm(m1,h1),torch.bmm(m2,h2)),2)
        neigh =self.nei_att(label_att)
        node=torch.cat((neigh,embs_batch),2)
        node = self.convs1(node).squeeze(1)
        embs = F.relu(self.final_layer(node), inplace=True)
        output = torch.sigmoid(self.output_layer(embs).squeeze(-1))
        return output


