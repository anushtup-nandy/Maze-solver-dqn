from maze import Maze
from dqn_new import DeepQNetwork
import numpy as np
import time


def run_maze():
    step = 0
    tt1=RL.time
    n1=0
    n2=0
    n3=0
    n4=0
    for episode in range(10200):
        # initial observation
        observation = env.reset()
        pa=-1

        # env.render()
        while True:
            # fresh env
            # env.render()##########################################

            # RL choose action based on observation
            action = RL.choose_action(observation,pa)
            pa=action

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)


            if reward ==-1:
                if n1%5==0:
                    RL.store_transition(observation, action, reward, observation_)
                n1+=1

            elif reward ==0:
                if n2%5==0:
                    RL.store_transition(observation, action, reward, observation_)
                n2+=1

            elif reward ==-1:        
                if n3%2==0:
                    RL.store_transition(observation, action, reward, observation_)
                n3+=1

            elif reward ==5:
                if n4%2==0:
                    RL.store_transition(observation, action, reward, observation_)
                n4+=1

            # RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step %5== 0):
            #if (step > 200) :
                RL.learn(episode)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done==1:
                break
            # if done==-1:
            #     #episode=episode-1
            #     break
            step += 1

    # end of game
    tt2=RL.time
    print(tt2-tt1)
    print('game over')
    print(n1)
    print(n2)
    print(n3)
    print(n4)


    # search path
    observation = env.reset()
    while True: 
        action = RL.choose_action2(observation)
        observation_, reward, done = env.step(action)
        print(observation, observation_)
        env.path(observation,observation_)
        observation = observation_
        if done:
            break
    time.sleep(5)
    env.destroy()

if __name__ == "__main__":
    # maze game
    pits=np.array([[6,3],[2,6]])
    agentXY=[0,0]
    goalXY=[8,4]

    env = Maze(agentXY, goalXY, pits)

    RL = DeepQNetwork(env.num_act, env.num_feat,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,#0.7 nono
                      hidden_layers=[10,10],#[10,10]
                      replace_target_iter=500,#500
                      memory_size=5000,#5000
                      )
    
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
    