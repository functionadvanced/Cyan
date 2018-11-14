import torch
import torch.utils.data
import MidiPoint

class LSTMpredictor(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, note_num, target_dim):
        super(LSTMpredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.note_embedding = torch.nn.Embedding(note_num, embedding_dim)
        self.hidden2note = torch.nn.Linear(hidden_dim, target_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, note_seq):
        embeds = self.note_embedding(note_seq)
        lstm_out, self.hidden = self.lstm(embeds.view(len(note_seq), 1, -1), self.hidden)
        note_space = self.hidden2note(lstm_out.view(len(note_seq), -1))
        note_scores = torch.nn.functional.log_softmax(note_space, dim=1)
        return note_scores

class DataSet(torch.utils.data.Dataset):
    def __init__(self, mode, isTrain=False):
        self.isTrain = isTrain
        if mode == 0: # both hands
            self.pointList = MidiPoint.PointList('1-l.mid', '1-r.mid').list
        elif mode == 1: # left hand
            self.pointList = MidiPoint.PointList('1-l.mid').list
        else:
            self.pointList = MidiPoint.PointList('1-r.mid').list
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        offset = 0
        total_len = int(len(self.pointList) * 0.8) - 1
        if not self.isTrain:
            offset = int(len(self.pointList) * 0.8)
            total_len = int(len(self.pointList) * 0.2) - 1
        result = []
        label = []
        for i in range(total_len):
            temp = offset+i
            result.append((self.pointList[temp].note, self.pointList[temp].delta_time))
            label.append((self.pointList[temp+1].note, self.pointList[temp+1].delta_time))
        
        return (result, label)

train_dataset = DataSet(0,isTrain=True)
train_loader = torch.utils.data.DataLoader(train_dataset)
for (i, j) in train_loader:
    print(i[10])
    break