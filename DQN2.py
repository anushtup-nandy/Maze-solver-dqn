'''
This is my own simple implementation of the DQN algorithm. 
Using the pytorch library
'''

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as ag


#NN design:
class Net(nn.Module):
    def __init__(self, nIn, nOut, nNeuron):
        super(Net, self).__init__()
        self.nIn=nIn
        self.nOut=nOut
        self.fc1=nn.Linear(nIn, nNeuron)
        self.fc2=nn.Linear(nNeuron, nOut)

    def forward(self, st):
        x=F.relu(self.fc1(st))
        Q_val=self.fc2(x)
        return Q_val

#Replay memory:
class Replay(object):
    def __init__(self, capacity):
        self.capacity=capacity
        self.memory=[]

    def push(self, event): #pushing data into the memory of the agent
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]  #kind of like FIFO algo

    def sample(self, batchSize):
        samples=zip(*random.sample(self.memory, batchSize))
        return map(lambda x: ag.Variable(torch.cat(x,0)))

#the DQN:
class DQN():
    def __init__(self, settings):
        self.settings = settings
        self.gamma = settings["gamma"]
        self.rewardWindow = []
        self.model = Net(settings["nInputs"], settings["nOutputs"],settings["nNeurons"])
        self.memory = Replay(settings["memoryCapacity"])
        self.optimizer = optim.Adam(self.model.parameters(), lr = settings["learningRate"])
        self.lastState = torch.Tensor(settings["nInputs"]).unsqueeze(0)
        self.lastAction = 0
        self.lastRewad = 0

    def selectAct(self, st):
        if(len(self.memory.memory) < self.settings["learningIterations"]):
            with torch.no_grad():
                probs = F.softmax(self.model(ag.Variable(st))*self.settings["softmaxTemperature"], dim=0)
        else:            
            with torch.no_grad():
                action = np.argmax(self.model(ag.Variable(st)).numpy(),1)
                return action[0]
        action = probs.multinomial(1)
        return int(action.data[0,0])

    def learn(self, batchst, batchnxtst, batchrew, batchact):
        outputs=self.model(batchst).gather(1, batchact.unsqueeze(1)).squeeze(1)
        nxtOutputs=self.model(batchnxtst).detach().max(1)[0]
        target=self.gammma*nxtOutputs + batchrew #this is the value function
        TDLLOSS=F.smooth_l1_loss(outputs,target)
        self.optimizer.zero_grad()
        TDLLOSS.backward()
        self.optimizer.step()  #performs a single optimization step.

    def update(self, reward, newSig):
        newSt=torch.Tensor(newSig).float().unsqueeze()
        self.memory.push((self.lastState, newSt, torch.LongTensor([int(self.lastAction)]), torch.Tensor([self.lastRewad])))
        action=self.selectAct(newSt)
        
        if len(self.memory.memory)>self.settings["batchSize"]:
              batchst, batchnxtst, batchact, batchrew= self.memory.sample(self.settings['batchSize'])
              self.learn(batchst, batchnxtst, batchrew, batchact)
        self.lastAction=action
        self.lastState=newSt
        self.lastRewad=reward
        return action
    
    def score(self):
        return sum(self.rewardWindow)/(len(self.rewardWindow)+1.)
    
    def save(self):
        torch.save({'state_dictionary': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("Loading...")
            check=torch.load('last_brain.path')
            self.model.load_state_dict(check['state_dict'])
            self.optimizer.load_state_dict(check['optimizer'])
            print('DONE')
        else:
            print('nothing here')