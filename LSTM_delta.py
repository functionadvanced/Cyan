import torch
import torch.utils.data
import MidiPoint
import os


class LSTMpredictor(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMpredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(2, hidden_dim)
        self.hidden2note = torch.nn.Linear(hidden_dim, 2)
        self.hidden = self.init_hidden()
        self.min_note = 0

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
        savedModel_name = "LSTM_delta_DT.model"
        model_path = os.path.join(dir_path, savedModel_name)
        if os.path.isfile(model_path):
            savedModel = torch.load(model_path)
            self.load_state_dict(savedModel['state_dict'])
            self.min_loss = savedModel['MSEloss']

    def predictFromMultiple(self, LEN, start_notes, min_time):
        '''
        start_notes should be a n*2 array
        '''
        all_re = []
        self.loadModel()
        self.hidden = self.init_hidden()
        base_note = start_notes[0][0]
        start_notes = torch.tensor(start_notes)
        
        for jj in range(len(start_notes)-1):
            ii = len(start_notes) - jj -1
            start_notes[ii, 0] -= start_notes[ii-1, 0]
        start_notes[0, 0] = 0
        
        with torch.no_grad():
            for ii in range(len(start_notes)):
                re = self.forward(start_notes[ii, :].view(1,1,2))
            for ii in range(LEN):                                
                re = self.forward(re)
                s_re = re.view(2).tolist()
                s_re[0] = round(s_re[0])
                s_re[1] = round(s_re[1])
                if s_re[1] <= 0:
                    s_re[1] = min_time
                all_re.append(s_re)
                re = torch.tensor([s_re], dtype=torch.float)
        all_re = torch.tensor(all_re)   
        all_re = recover(all_re, base_note)
        return all_re

    def train(self, train_dataset):                
        loss_func = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(myLstm.parameters(), lr=0.001)


        self.loadModel()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        savedModel_name = "LSTM_delta_DT.model"
        model_path = os.path.join(dir_path, savedModel_name)

        test_song_num = int(train_dataset.numSong * 0.2)

        for epoch in range(50000):
            sum_loss = 0
            for songIdx in range(train_dataset.numSong - test_song_num):
                self.zero_grad()
                self.hidden = self.init_hidden()
                data = torch.tensor(train_dataset.allResults[songIdx], dtype=torch.float)
                target = torch.tensor(train_dataset.allLabels[songIdx], dtype=torch.float)
                predict = self.forward(data)
                loss = loss_func(predict, target)
                sum_loss += loss
                loss.backward(retain_graph=True)
                optimizer.step()
            with torch.no_grad():
                train_loss = sum_loss / (train_dataset.numSong - test_song_num) / TOTAL_LEN
                for ii in range(test_song_num):
                    self.hidden = self.init_hidden()
                    songIdx = train_dataset.numSong - test_song_num - 1 + ii
                    data = torch.tensor(train_dataset.allResults[songIdx], dtype=torch.float)
                    target = torch.tensor(train_dataset.allLabels[songIdx], dtype=torch.float)
                    predict = self.forward(data)
                    loss = loss_func(predict, target)
                    sum_loss += loss
                test_loss  = sum_loss / test_song_num / TOTAL_LEN
                print("train loss: {:.1f}, test loss: {:.1f}".format(train_loss, test_loss))
                # save
                if test_loss < 15.5:
                    torch.save(
                                {
                                    'MSEloss': test_loss,
                                    'state_dict': self.state_dict(),
                                }, model_path
                            )
                    print('save model')
                    break

        with torch.no_grad():
            self.hidden = self.init_hidden()
            data = torch.tensor(train_dataset.allResults[16], dtype=torch.float)
            re = recover(self.forward(data), 70)
            print(re)            
            PlayResult(re)


TOTAL_LEN = 150

class DataSet(torch.utils.data.Dataset):
    def normalize(self):
        for j in range(self.numSong):
            current_note = self.allLists[j][0].note
            self.allLists[j][0].note = 0          
            for i in range(len(self.allLists[j])-1):
                temp = self.allLists[j][i+1].note
                self.allLists[j][i+1].note -= current_note
                current_note = temp
        
        # quantize delta time
        for j in range(self.numSong):
            min_interval = 100
            for ii in range(len(self.allLists[j])):
                tt = self.allLists[j][ii].delta_time
                if tt > 0.15 and tt < min_interval:        
                    min_interval = tt
            # print(min_interval)
            for ii in range(len(self.allLists[j])):
                tt = self.allLists[j][ii].delta_time
                if tt <= 0.15:
                    self.allLists[j][ii].delta_time = 0
                else:
                    self.allLists[j][ii].delta_time = round(tt / min_interval)
    


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
            else: # right hand
                pointList = MidiPoint.PointList(idx+'-r.mid').list
            
            
            
            self.allLists.append(pointList)
        self.normalize()
        for j in range(self.numSong):
            offset = 0
            total_len = TOTAL_LEN
            if not self.isTrain:
                offset = int(len(self.allLists[j]) * 0.8)
                total_len = int(len(self.allLists[j]) * 0.2) - 1
            result = []
            label = []
            for i in range(total_len):
                temp = offset+i
                result.append([self.allLists[j][temp].note, self.allLists[j][temp].delta_time])
                label.append([self.allLists[j][temp+1].note, self.allLists[j][temp+1].delta_time])
            self.allResults.append(result)
            self.allLabels.append(label)
    def __len__(self):
        return self.numSong
    def __getitem__(self, idx):       
        return (self.result[idx], self.label[idx])

def PlayResult(re):
    pl = []
    # UI.AutoPlay(re)
    # return
    for i in re:
        if i[1] < 0:
            i[1] = 0
        # print(i[1].double() / 50.0)
        pl.append(MidiPoint.Point(int(i[0]), 100, 0, 0, i[1].double()))
    MidiPoint.PlayList(pl)

def recover(aim, current_note): # aim should be a n*2 tensor
    for j in range(len(aim)):
        aim[j, 0] += current_note
        aim[j, 1] *= 0.5
        current_note = aim[j, 0]
    return aim

# train_dataset = DataSet(2,numSong=18,isTrain=True)
# # # PlayResult(recover(torch.tensor(train_dataset.allResults[0]), 70))
# myLstm = LSTMpredictor(128)
# myLstm.train(train_dataset)