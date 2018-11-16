import torch
import torch.utils.data
import MidiPoint

torch.manual_seed(1)

class LSTMpredictor(torch.nn.Module):
    def __init__(self, hidden_dim, note_num):
        super(LSTMpredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(2, hidden_dim)
        self.hidden2note = torch.nn.Linear(hidden_dim, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, note_seq):
        lstm_out, self.hidden = self.lstm(note_seq.view(len(note_seq), 1, 2), self.hidden)
        note_space = self.hidden2note(lstm_out.view(len(note_seq), -1))
        note_scores = torch.nn.functional.log_softmax(note_space, dim=1)
        return note_space

class DataSet(torch.utils.data.Dataset):
    def normalize(self):
        # find the min note & max delta_time, then normalize
        self.min_note = 1000
        self.max_note = 0
        self.max_delta_time = 0
        for i in range(len(self.pointList)):
            if self.pointList[i].note < self.min_note:
                self.min_note = self.pointList[i].note
            if self.pointList[i].note > self.max_note:
                self.max_note = self.pointList[i].note
            if self.pointList[i].delta_time > self.max_delta_time:
                self.max_delta_time = self.pointList[i].delta_time
        
        for i in range(len(self.pointList)):
            self.pointList[i].note -= self.min_note
            self.pointList[i].delta_time *= 3
    
    def recover(self, aim): # aim should be a n*2 tensor
        aim[:, 0] += self.min_note
        aim[:, 1] /= 3
        return aim

    def __init__(self, mode, isTrain=False):
        self.isTrain = isTrain
        if mode == 0: # both hands
            self.pointList = MidiPoint.PointList('2-l.mid', '2-r.mid').list
        elif mode == 1: # left hand
            self.pointList = MidiPoint.PointList('2-l.mid').list
        else:
            self.pointList = MidiPoint.PointList('2-r.mid').list

        self.normalize()

        offset = 0
        total_len = 40
        if not self.isTrain:
            offset = int(len(self.pointList) * 0.8)
            total_len = int(len(self.pointList) * 0.2) - 1
        self.result = []
        self.label = []

        

        for i in range(total_len):
            temp = offset+i
            self.result.append([self.pointList[temp].note, self.pointList[temp].delta_time])
            self.label.append([self.pointList[temp+1].note, self.pointList[temp+1].delta_time])


    def __len__(self):
        return 1
    def __getitem__(self, idx):       
        return (self.result, self.label)


def PlayResult(re):
    pl = []
    for i in re:
        if i[1] < 0:
            i[1] = 0
        pl.append(MidiPoint.Point(int(torch.round(i[0])), 100, 0, 0, i[1]))
    MidiPoint.PlayList(pl)

train_dataset = DataSet(2,isTrain=True)
myLstm = LSTMpredictor(64, len(train_dataset.result))
loss_func = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(myLstm.parameters(), lr=0.001)
data = torch.tensor(train_dataset.result)
target = torch.tensor(train_dataset.label)

for epoch in range(10000):
    myLstm.zero_grad()
    myLstm.hidden = myLstm.init_hidden()
    predict = myLstm(data)
    loss = loss_func(predict, target)
    print(loss)
    loss.backward(retain_graph=True)
    optimizer.step()
    # print(myLstm(data))
with torch.no_grad():
    # print(train_dataset.recover(target))
    myLstm.hidden = myLstm.init_hidden()
    
    re = train_dataset.recover(myLstm(data))
    print(re)
    # PlayResult(train_dataset.recover(target))
    PlayResult(re)

