import MidiPoint
import torch
import torch.utils.data

class SplitNet(torch.nn.Module):
    def __init__(self, num_notes=6):
        super(SplitNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_notes*3,  1<<num_notes),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1<<num_notes, (1<<num_notes) * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((1<<num_notes) * 2, (1<<num_notes) * 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((1<<num_notes) * 3, (1<<num_notes) * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((1<<num_notes) * 4, (1<<num_notes) * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((1<<num_notes) * 2, 1<<num_notes),
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
    def __len__(self):
        if self.isTrain:
            return int(len(self.pointList) * 0.8 / self.num_notes)
        else:
            return int(len(self.pointList) * 0.2 / self.num_notes)
    def __getitem__(self, idx):
        offset = 0
        if not self.isTrain:
            offset = int(len(self.pointList) * 0.8)
        result = torch.zeros(self.num_notes*3)        
        corr_index = 0
        time_offset = self.pointList[offset+idx*self.num_notes].time
        for i in range(self.num_notes):
            result[i*3]   = self.pointList[offset+idx*self.num_notes+i].note
            result[i*3+1] = self.pointList[offset+idx*self.num_notes+i].velocity
            result[i*3+2] = self.pointList[offset+idx*self.num_notes+i].time - time_offset
            corr_index << 1
            if self.pointList[idx*self.num_notes+i].isLeft:                    
                corr_index += 1
        label = torch.zeros(1<<self.num_notes)
        label[corr_index] = 1
        return (result, label)


# Function to get no of set bits in binary 
# representation of positive integer n */ 
def  countSetBits(n): 
    count = 0
    while (n): 
        count += n & 1
        n >>= 1
    return count 

num_notes = 7 # train this number of notes together

myNet = SplitNet(num_notes = num_notes)
val_dataset = DataSet(num_notes=num_notes)
train_dataset = DataSet(num_notes=num_notes,isTrain=True)
lossFunc = torch.nn.MSELoss()
optimizer = torch.optim.SGD(myNet.parameters(), lr=0.001)

for epoch in range(10000):
    # validate
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    wrong_num = 0
    total_num = 0
    for _, (v_data, v_label) in enumerate(val_loader):
        re = myNet.forward(v_data[0])
        a = int(re.argmax(0))
        b = int(v_label[0].argmax(0))
        n = a ^ b
        wrong_num += countSetBits(n)
        total_num += num_notes
    val_loss = wrong_num / total_num
    # train
    train_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    wrong_num = 0
    total_num = 0
    for _, (t_data, t_label) in enumerate(train_loader):
        re = myNet.forward(t_data[0])
        a = int(re.argmax(0))
        b = int(t_label[0].argmax(0))
        n = a ^ b
        wrong_num += countSetBits(n)
        total_num += num_notes    
        loss = lossFunc(re, t_label[0])
        myNet.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = wrong_num / total_num
    if epoch % 100 == 1:
        print("Epoch {}".format(epoch))
        print("validation loss: {0:.2f}%".format(val_loss * 100))
        print("training loss: {0:.2f}%".format(train_loss * 100))

