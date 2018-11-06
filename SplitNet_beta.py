import MidiPoint
import torch
import torch.utils.data
import os

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
    def forward(self, x):
        x = self.model(x)
        return x

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

num_notes = 6 # train this number of notes together

myNet = SplitNet(num_notes = num_notes)
val_dataset = DataSet(num_notes=num_notes)
train_dataset = DataSet(num_notes=num_notes,isTrain=True)
lossFunc = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(myNet.parameters(), lr=0.0005)

best_accuracy = 0
dir_path = os.path.dirname(os.path.realpath(__file__))
savedModel_name = "SplitModel.model"
model_path = os.path.join(dir_path, savedModel_name)
if os.path.isfile(model_path):
    savedModel = torch.load(model_path)
    if savedModel['num_notes'] == num_notes:
        myNet.load_state_dict(savedModel['state_dict'])
        best_accuracy = savedModel['accuracy']

for epoch in range(10000):
    # validate
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    wrong_num = 0
    total_num = 0
    for _, (v_data, v_label) in enumerate(val_loader):
        re = myNet.forward(v_data[0])
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
        re = myNet.forward(t_data[0])
        loss = lossFunc(re, t_label[0])
        myNet.zero_grad()
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
                'state_dict': myNet.state_dict(),
                'num_notes': num_notes,
            }, model_path
        )
        best_accuracy = accuracy
        print("save model")