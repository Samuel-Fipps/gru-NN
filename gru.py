from sqlalchemy import true
import torch.cuda
import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from distutils import util
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time

#-------------------------------------------------------------------------
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(threshold=sys.maxsize) 
torch.set_printoptions(threshold=10_000)
#-------------------------------------------------------------------------

#input_data = torch.Tensor(np.load("1inputData.npy", allow_pickle=True))
#predict_data = torch.Tensor(np.load("1predict.npy", allow_pickle=True))

input_data = torch.Tensor(np.load("biginputdata.npy", allow_pickle=True))
predict_data = torch.Tensor(np.load("bigpredictdata.npy", allow_pickle=True))

#testingdata_x = torch.Tensor(np.load("1testingdata_x.npy", allow_pickle=True))
#testingdata_y = torch.Tensor(np.load("1testingdata_y.npy", allow_pickle=True))

#testingdata_x = testingdata_x.type(torch.FloatTensor)
#testingdata_y = testingdata_y.type(torch.LongTensor)
input_data = input_data.type(torch.FloatTensor)
predict_data = predict_data.type(torch.LongTensor)



#testingdata_x = torch.Tensor(np.load("1inputData.npy", allow_pickle=True))
#testingdata_y = torch.Tensor(np.load("1predict.npy", allow_pickle=True))

#testingdata_x = testingdata_x.type(torch.FloatTensor)
#testingdata_y = testingdata_y.type(torch.LongTensor)

#print(predict_data)

#print(type(input_data))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)     
        self.fc = nn.Linear(hidden_size, num_classes)                                                
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        out, _ = self.gru(x, (h0))
        out = out[:, -1, :]
         
        out = self.fc(out)

        return out

learning_rate = 0.001######################################################################################
input_size = 248
num_layers = 2
hidden_size = 248
num_classes = 2

lr = .001
wd = 0
# 3 lrs
# 1 wd
# 3 eps
# 4 betas
wd  = [ 1e-11, 1e-14, 1e-15]
eps = [0.000005, 0.000001, 0.0000005]  # 
lr  = [0.0002, 0.0005, 0.0009]
betasleft = [0.65, 0.7, 0.75]
betasright = [0.9999999, 0.9999, 0.999 ]

best = np.float64([99]) #antioverfit   

# hint just take the most wining and put parameters on both sides of it
  
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss() 



optimizer = torch.optim.RAdam(model.parameters(), lr=0.0005, betas=(0.8, 0.999), eps=1e-07, weight_decay=1e-11)
BATCH_SIZE = 100
num_epochs = 5000000
print_interval = 3000
a = np.float64([99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99]) #antioverfit  
testing_loss = 0.0
model.train()

model.to(device)
input_data.to(device)
predict_data.to(device)

PATH = "model.pt"
torch.save({
            'model_state_dict': model.state_dict(),
            }, PATH)


start_time = time.time()

counter = 0
for WD in wd:
    for EPS in eps:
        for LR in lr:
            for BETASRIGHT in betasright:
                for BETASLEFT in betasleft:
                    counter = counter + 1

                    checkpoint = torch.load(PATH)
                    model.load_state_dict(checkpoint['model_state_dict'])

                    optimizer = torch.optim.RAdam(model.parameters(), lr=LR, betas=(BETASLEFT, BETASRIGHT), eps=EPS, weight_decay=WD  )
                    a = np.float64([99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99]) #antioverfit     
                    testing_loss = 0.0
                    model.train()

                    for epoch in range(num_epochs):
                        start_time = time.time()

                        if(testing_loss < a[4]): # part of anti overfit
                            train_loss = 0.0        
                            testing_loss = 0.0

                            model.train()
                            for i in (range(0, len(input_data), BATCH_SIZE)):
                                batch_X = input_data[i:i+BATCH_SIZE]
                                batch_y = predict_data[i:i+BATCH_SIZE]

                                batch_X = batch_X.to(device) #gpu                        # input data here!!!!!!!!!!!!!!!!!!!!!!!!!!
                                batch_y = batch_y.to(device) #gpu                    # larget data here!!!!!!!!!!!!!!!!!!!!!!!!!!

                                batch_X = batch_X.reshape(-1, 1, input_size).to(device)
                                output = model(batch_X)
                                
                                loss = criterion(output, batch_y).to(device)

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                
                            
                            print(
                                f"Epoch [{epoch + 1}/{num_epochs}], "
                                f"Step [{i + 1}/{len(input_data)}], "
                                f"Loss: {loss.item():.4f}"
                            )
                            secondTime = time. time()
                            print("total time for 1 epoch: ", secondTime-start_time)


                            if(epoch%10 ==0):
                                #val loss calc below
                                model.eval()

                                correctCount =0 
                                wrongCount =0

                                with torch.no_grad():

                                    for i in (range(0, len(input_data), BATCH_SIZE)):
                                        batch_X = input_data[i:i+BATCH_SIZE]
                                        batch_y = predict_data[i:i+BATCH_SIZE]

                                        batch_X = batch_X.to(device) #gpu                        # input data here!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        batch_y = batch_y.to(device) #gpu                    # larget data here!!!!!!!!!!!!!!!!!!!!!!!!!!      

                                        batch_X = batch_X.reshape(-1, 1, input_size).to(device)
                                        output = model(batch_X)

                                        _, pred = torch.max(output, dim=1)
                                        correct = np.squeeze(pred.eq(batch_y.data.view_as(pred)))
                                        for i in correct:
                                            if i == True:
                                                correctCount = correctCount+1
                                            else:
                                                wrongCount = wrongCount +1


                                #print(f"Accuracy: {wrongCount / 90000 * 100:.10f}%")

                                accuracy = wrongCount / (correctCount+wrongCount) * 100

                                a = np.insert(a,0,accuracy) # part of anti overfit         
                                a = np.delete(a,22)      
                                testing_loss = accuracy
                                print("Accuracy: " ,accuracy)
                                secondTime = time. time()
                                print("total time for 1 epoch: ", secondTime-start_time)
                                

                    torch.save(model, "models/GRUModel.pth")
                    print(optimizer)
                    print("lr= ", LR, "betaright= ", BETASRIGHT, "betaleft= ", BETASLEFT, " wd= ", WD, "eps= ", EPS)
                    print("round: ", counter, " out of 243")
                    best = np.append(best, accuracy)
print(best)
                    


