import torch
import torch.utils.data
import MidiPoint
import os

torch.manual_seed(1)

class LSTMpredictor(torch.nn.Module):
    def __init__(self, hidden_dim):
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

    def loadModel(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        savedModel_name = "LSTM.model"
        model_path = os.path.join(dir_path, savedModel_name)
        if os.path.isfile(model_path):
            savedModel = torch.load(model_path)
            self.load_state_dict(savedModel['state_dict'])
            self.min_loss = savedModel['MSEloss'] * savedModel['total_len'] / TOTAL_LEN
            self.min_note = savedModel['min_note']

    def predictFromOne(self):
        all_re = []
        re = torch.tensor([[73.0, 0.05]])
        self.loadModel()
        self.hidden = myLstm.init_hidden()
        with torch.no_grad():
            for ii in range(TOTAL_LEN):                                
                re = self.forward(re)
                s_re = re.view(2).tolist()
                s_re[1] = round(s_re[1]*1.5, 0) / 1.5
                all_re.append(s_re)
        all_re = torch.tensor(all_re)
        all_re[:, 0] += self.min_note
        all_re[:, 1] /= 3

        print(all_re)
        PlayResult(all_re)

    def train(self, train_dataset):                
        loss_func = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(myLstm.parameters(), lr=0.0001)

        self.min_loss = 1000000
        self.loadModel()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        savedModel_name = "LSTM.model"
        model_path = os.path.join(dir_path, savedModel_name)

        for epoch in range(5000):
            sum_loss = 0
            for songIdx in range(train_dataset.numSong):
                self.zero_grad()
                self.hidden = myLstm.init_hidden()
                data = torch.tensor(train_dataset.allResults[songIdx])
                target = torch.tensor(train_dataset.allLabels[songIdx])
                predict = myLstm(data)
                loss = loss_func(predict, target)
                sum_loss += loss
                # print(loss)
                
                
                loss.backward(retain_graph=True)
                optimizer.step()
            avg_loss = sum_loss / train_dataset.numSong
            print(avg_loss)
            # save
            # if avg_loss < self.min_loss and avg_loss < 100:
            if avg_loss < 0.3:
                torch.save(
                            {
                                'MSEloss': avg_loss / TOTAL_LEN,
                                'state_dict': self.state_dict(),
                                'total_len': TOTAL_LEN,
                                'min_note': train_dataset.min_note,
                            }, model_path
                        )
                self.min_loss = avg_loss
                print('save model')
                break



        with torch.no_grad():
            self.hidden = self.init_hidden()    
            re = train_dataset.recover(self.forward(data))
            print(re)            
            PlayResult(re)


TOTAL_LEN = 500

class DataSet(torch.utils.data.Dataset):
    def normalize(self):
        # find the min note & max delta_time, then normalize
        self.min_note = 1000
        self.max_note = 0
        self.max_delta_time = 0
        for j in range(self.numSong):
            for i in range(len(self.allLists[j])):
                if self.allLists[j][i].note < self.min_note:
                    self.min_note = self.allLists[j][i].note
                if self.allLists[j][i].note > self.max_note:
                    self.max_note = self.allLists[j][i].note
                if self.allLists[j][i].delta_time > self.max_delta_time:
                    self.max_delta_time = self.allLists[j][i].delta_time
        for j in range(self.numSong):
            for i in range(len(self.allLists[j])):
                self.allLists[j][i].note -= self.min_note
                self.allLists[j][i].delta_time *= 3
    
    def recover(self, aim): # aim should be a n*2 tensor
        aim[:, 0] += self.min_note
        aim[:, 1] /= 3
        return aim

    def __init__(self, mode, numSong=1, isTrain=False):
        self.isTrain = isTrain
        self.numSong = numSong
        self.allLists = []
        self.allResults = []
        self.allLabels = []
        for ii in range(self.numSong):
            idx = str(ii+1)
            if mode == 0: # both hands
                pointList = MidiPoint.PointList(idx+'-l.mid', idx+'-r.mid').list
            elif mode == 1: # left hand
                pointList = MidiPoint.PointList(idx+'-l.mid').list
            else:
                pointList = MidiPoint.PointList(idx+'-r.mid').list
            
            
            
            self.allLists.append(pointList)
        self.normalize()
        for ii in range(self.numSong):
            offset = 0
            total_len = TOTAL_LEN
            if not self.isTrain:
                offset = int(len(pointList) * 0.8)
                total_len = int(len(pointList) * 0.2) - 1
            result = []
            label = []
            for i in range(total_len):
                temp = offset+i
                result.append([pointList[temp].note, pointList[temp].delta_time])
                label.append([pointList[temp+1].note, pointList[temp+1].delta_time])
            self.allResults.append(result)
            self.allLabels.append(label)

        


    def __len__(self):
        return self.numSong
    def __getitem__(self, idx):       
        return (self.result[idx], self.label[idx])

def PlayResult(re):
    pl = []
    for i in re:
        if i[1] < 0:
            i[1] = 0
        pl.append(MidiPoint.Point(int(torch.round(i[0])), 100, 0, 0, i[1]))
    MidiPoint.PlayList(pl)

train_dataset = DataSet(2,numSong=18,isTrain=True)
myLstm = LSTMpredictor(64)
# myLstm.train(train_dataset)
myLstm.predictFromOne()