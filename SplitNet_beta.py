import MidiPoint
import torch
import torch.utils.data
import os


'''
Example:
myNet = SplitNet(num_notes=6)
# Train for 10000 epoches, need ctrl^c to end
myNet.train()
# forward: indicate whether the key at int(num_notes/2) is left or right
# Assum num_notes = 6, then
argmax(myNet.forward([n_0, n_2, ..., n_5]))
# 0: n_3 is right hand
# 1: n_3 is left hand
'''


class SplitNet(torch.nn.Module):
    def __init__(self, num_notes=6):
        super(SplitNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_notes,  60),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(60,  120),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(120,  60),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(60,  30),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(30,  2),
            torch.nn.ReLU(inplace=True),
        )
        self.num_notes = num_notes
        self.loadModel()

    def forward(self, x):
        a = self.model(x)
        return a

    def Split(self, midi_name):
        # do something to split the midi file
        plist = MidiPoint.PointList(midi_name)
        total_len = len(plist.list)
        for i in range(total_len):
            begin_pos = i - int(self.num_notes/2)
            segment = []
            for j in range(self.num_notes):
                t_pos = begin_pos + j
                if t_pos >= total_len:
                    t_pos -= total_len
                segment.append(plist.list[t_pos].note)
            segment = torch.tensor(segment, dtype=torch.float)
            plist.list[i].isLeft = self.forward(segment).argmax()
        plist.saveAsMidi('temp.mid', play=True, mode=2) # play right hand
        

    def loadModel(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        savedModel_name = "SplitModel.model"
        model_path = os.path.join(dir_path, savedModel_name)
        self.best_accuracy = 0
        if os.path.isfile(model_path):
            savedModel = torch.load(model_path)
            if savedModel['num_notes'] == self.num_notes:
                self.load_state_dict(savedModel['state_dict'])
                self.best_accuracy = savedModel['accuracy']

    def train(self):
        val_dataset = DataSet(num_notes=self.num_notes)
        train_dataset = DataSet(num_notes=self.num_notes,isTrain=True)
        lossFunc = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0005)

        best_accuracy = self.best_accuracy
        dir_path = os.path.dirname(os.path.realpath(__file__))
        savedModel_name = "SplitModel.model"
        model_path = os.path.join(dir_path, savedModel_name)
        if os.path.isfile(model_path):
            savedModel = torch.load(model_path)
            if savedModel['num_notes'] == self.num_notes:
                self.load_state_dict(savedModel['state_dict'])
                best_accuracy = savedModel['accuracy']

        for epoch in range(10000):
            # validate
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
            wrong_num = 0
            total_num = 0
            for _, (v_data, v_label) in enumerate(val_loader):
                re = self.forward(v_data[0])
                a = int(re.argmax(0))
                b = int(v_label[0].argmax(0))
                if a != b:
                    wrong_num += 1
                total_num += 1
            val_loss = wrong_num / total_num
            # train
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            wrong_num = 0
            total_num = 0
            for _, (t_data, t_label) in enumerate(train_loader):
                re = self.forward(t_data[0])
                loss = lossFunc(re, t_label[0])
                self.zero_grad()
                loss.backward()
                optimizer.step()
                a = int(re.argmax(0))
                b = int(t_label[0].argmax(0))
                if a != b:
                    wrong_num += 1
                total_num += 1    
            train_loss = wrong_num / total_num
            if epoch % 1 == 0:
                print("Epoch {}".format(epoch))
                print("validation loss: {0:.4f}%".format(val_loss * 100))
                print("training loss: {0:.4f}%".format(train_loss * 100))

            accuracy = 1 - val_loss
            if accuracy > best_accuracy:
                torch.save(
                    {
                        'accuracy': 1-val_loss,
                        'state_dict': self.state_dict(),
                        'num_notes': self.num_notes,
                    }, model_path
                )
                best_accuracy = accuracy
                print("save model")

class DataSet(torch.utils.data.Dataset):
    def __init__(self, num_notes=6, isTrain=False):
        self.isTrain = isTrain
        self.num_notes = num_notes
        self.pointList = MidiPoint.PointList('1-l.mid', '1-r.mid').list
        self.pointList.extend(MidiPoint.PointList('2-l.mid', '2-r.mid').list)
        self.pointList.extend(MidiPoint.PointList('3-l.mid', '3-r.mid').list)
        self.pointList.extend(MidiPoint.PointList('4-l.mid', '4-r.mid').list)
        self.pointList.extend(MidiPoint.PointList('5-l.mid', '5-r.mid').list)
    def __len__(self):
        if self.isTrain:
            return int(len(self.pointList) * 0.8)-self.num_notes
        else:
            return int(len(self.pointList) * 0.2)-self.num_notes
    def __getitem__(self, idx):
        offset = 0
        if not self.isTrain:
            offset = int(len(self.pointList) * 0.8)
        result = torch.zeros(self.num_notes)
        for i in range(self.num_notes):
            temp = offset+idx+i
            result[i]   = self.pointList[temp].note
        label = torch.zeros(2)
        target_pos = offset+idx+int(self.num_notes/2)
        if self.pointList[target_pos].isLeft:
            label[1] = 1
        else:
            label[0] = 1
        return (result, label)

myNet = SplitNet(num_notes=6)
# myNet.train()
myNet.Split('3-whole.mid')