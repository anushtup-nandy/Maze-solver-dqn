import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd 
from collections import namedtuple
import copy

class DeepQNetwork(nn.Module):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            hidden_layers=[10, 10],
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None
    ):
        super(DeepQNetwork, self).__init__()

        self.episode=0
        self.time=time.time()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.hidden_layers = hidden_layers
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.cost_his = []
        self.cost_time=[]
        self.cost_time2=[]
        self.sum=[]
        self.sum2=[]
        self.episode=0
        self.time=time.time()

        self.pastop=[1,0,3,2]
        self.arcount=0
        self.ascount=0

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory =  pd.DataFrame(np.zeros((self.memory_size, n_features*2+2)))
        #self.memory=torch.zeros((self.memory_size, n_features*2+2))

        self._build_net()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.target_net = copy.deepcopy(self.eval_net)
        
    def _build_net(self):

        self.eval_net = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[1], self.n_actions)
        )

    def forward(self, x):
        return self.eval_net(x)
    
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        #print(transition.shape) #(6,)

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1
    
    def choose_action(self, observation,pa):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value).item()

        else:

            while True:
                r=np.random.uniform()
                if r < 0.5 :
                    action=3
                else:
                    action=1
                self.ascount=self.ascount+1


                if pa==-1:
                    break

                elif self.pastop[pa]!=action:
                    break

                elif self.pastop[pa]==action:
                    self.arcount=self.arcount+1
                    continue

        return action

    def choose_action2(self, observation,pa):
        # Convert observation to tensor
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        observation = observation[np.newaxis, :]
        # Forward pass to get action values
        actions_value = self.eval_net(observation)
        # Get the action with the highest value
        action=torch.argmax(actions_value).item()
        
        return action


    def _replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


    def learn(self, episode):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)
        #print(batch_memory.shape) #(32,6)

        q_next, q_eval = self.eval_net(batch_memory[:, -self.n_features:]), self.eval_net(batch_memory[:, self.n_features:]) 

        # change q_target w.r.t q_eval's action
        q_target = copy.deepcopy(q_eval).detach()
        q_target[range(self.batch_size), batch_memory[:, self.n_features].long()] = \
            batch_memory[:, self.n_features+1] + self.gamma * q_next.max(1)[0]        

        # train eval network
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_target, q_eval)
        loss.backward()
        self.optimizer.step()


        if self.episode!=episode:
            self.cost_his.append(self.cost)
            t1=time.time()
            self.cost_time.append(t1-self.time)
            self.cost_time2.append(t1)

            self.time=t1
            self.episode=episode
            self.sum.append(np.sum(q_target))



        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

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

