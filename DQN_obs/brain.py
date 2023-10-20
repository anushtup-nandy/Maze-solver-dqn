import torch
import numpy as np
import time 
from datetime import datetime
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag

np.random.seed(0)

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
        
        self.cost_his = []
        self.cost_time=[]
        self.cost_time2=[]
        self.sum=[]
        self.sum2=[]
        self.episode=0
        self.time=time.time()

    def selectAct(self, st, pa):
        if(len(self.memory.memory) < self.settings["learningIterations"]):
            with torch.no_grad():
                probs = F.softmax(self.model(ag.Variable(st))*self.settings["softmaxTemperature"], dim=0)
        else:            
            with torch.no_grad():
                action = np.argmax(self.model(ag.Variable(st)).numpy(),1)
                return action[0]
        action = probs.multinomial(1)
        return int(action.data[0,0])

    def learn(self, batchst, batchnxtst, batchrew, batchact, max_episodes):
        outputs=self.model(batchst).gather(1, batchact.unsqueeze(1)).squeeze(1)
        nxtOutputs=self.model(batchnxtst).detach().max(1)[0]
        target=(self.gamma)*nxtOutputs + batchrew #this is the value function
        TDLLOSS=F.smooth_l1_loss(outputs,target)
        self.optimizer.zero_grad()
        TDLLOSS.backward()
        self.optimizer.step()  #performs a single optimization step.
        
        if self.episode!=max_episodes:
            self.cost_his.append(TDLLOSS)
            t1=time.time()
            self.cost_time.append(t1-self.time)
            self.cost_time2.append(t1)
            
            self.time=t1
            self.episode=max_episodes
            self.sum.append(np.sum(target))

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
        
    def save(self):
        torch.save({'state_dictionary': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
    
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.figure(2) 
        plt.plot(np.arange(len(self.sum)), self.sum,'o')
        plt.figure(3) 
        plt.plot(np.arange(len(self.cost_time)), self.cost_time)
        plt.figure(4) 
        plt.plot(np.arange(len(self.cost_time2)), self.cost_time2)
        plt.show()
        

    
    