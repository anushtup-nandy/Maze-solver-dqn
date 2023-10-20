import numpy as np
import tkinter as tk 
import time as t

#--SIZE--
PIXELS=50 #number of pixels per cell 
MAZE_HEIGHT=10
MAZE_WIDTH=10

origin=np.array([PIXELS/2, PIXELS/2])

class Maze(tk.Tk):
    def __init__(self,agentXY, goalXY, pits=[]):
        super(Maze, self).__init__()
        self.action_space=['u','d','l','r']
        self.num_act=len(self.action_space)
        self.num_feat=2
        self.pitblocks=[]
        self.PIXELS=50
        self.MAZE_HEIGHT=10
        self.MAZE_WIDTH=10
        self.title('MAZE')
        self.geometry('{0}x{1}'.format(MAZE_HEIGHT * PIXELS, MAZE_WIDTH * PIXELS))
        self.build_maze(agentXY, goalXY, pits)
    
    '''
        Goals, pitts, agent
    '''
    def add_pitt(self,x,y):
        pit_center = origin + np.array([PIXELS * x, PIXELS*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='black'))
    
    def add_agent(self, x=0, y=0):
        agent_center = origin + np.array([PIXELS * x, PIXELS*y])
        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='blue')
    
    def add_goal(self, x=10, y=10):
        goal_center=origin+np.array([PIXELS*x, PIXELS*y])
        self.goal= self.canvas.create_oval(goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='red')

    def build_maze(self, agentXY, goalXY, pits):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_HEIGHT * PIXELS,
                           width=MAZE_WIDTH * PIXELS)

        #grids
        for c in range(0, MAZE_WIDTH * PIXELS, PIXELS):
            x0, y0, x1, y1 = c, 0, c, MAZE_HEIGHT * PIXELS
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_HEIGHT * PIXELS, PIXELS):
            x0, y0, x1, y1 = 0, r, MAZE_HEIGHT * PIXELS, r
            self.canvas.create_line(x0, y0, x1, y1) 
        
        for a,b in pits:
            self.add_pitt(a,b)
        
        self.add_goal(goalXY[0],goalXY[1])
        self.add_agent(agentXY[0],agentXY[1])
        self.canvas.pack()

    def reset(self, value = 1, resetAgent=True):
        self.update()
        #time.sleep(0.2)
        if(value == 0):
            return self.canvas.coords(self.agent)
        else:
            #Reset Agent
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                origin[0] + 15, origin[1] + 15,
                fill='red')
    
            return self.canvas.coords(self.agent)
    
    def step(self,action):
        reward=0
        s=self.canvas.coords(self.agent)
        base_act=np.array([0,0])
        
        #setting up what the action must be, and what the reward will be to take it:
        if action == 0:   # up
            if s[1] > PIXELS:
                base_act[1] -= PIXELS
            else:
                reward=-1
        elif action == 1:   # down
            if s[1] < (MAZE_HEIGHT - 1) * PIXELS:
                base_act[1] += PIXELS
            else:
                reward=-1
        elif action == 2:   # right
            if s[0] < (MAZE_WIDTH - 1) * PIXELS:
                base_act[0] += PIXELS
            else:
                reward=-1
        elif action == 3:   # left
            if s[0] > PIXELS:
                base_act[0] -= PIXELS
            else:
                reward=-1

        self.canvas.move(self.agent, base_act[0], base_act[1])  # move agent

        next_coords = self.canvas.coords(self.agent)  # next state

        if reward==-1:
            done=0
        elif next_coords==self.canvas.coords(self.goal):
            reward=5
            done=1
        elif next_coords in [self.canvas.coords(i) for i in self.pitblocks]:
            reward=-1
            done=0
        else:
            reward=0
            done=0
        
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.goal)[:2]))/(MAZE_HEIGHT*PIXELS)
        return s_, reward, done

    def render(self):
        self.update()

    def path(self, s, s_): 
        self.update()
        s0=s[0]*MAZE_HEIGHT*PIXELS+self.canvas.coords(self.oval)[:1]
        s1=s[1]*MAZE_HEIGHT*PIXELS+self.canvas.coords(self.oval)[1:2]

        s_0=s_[0]*MAZE_HEIGHT*PIXELS+self.canvas.coords(self.oval)[:1]
        s_1=s_[1]*MAZE_HEIGHT*PIXELS+self.canvas.coords(self.oval)[1:2]

        # print(s0,s1,s_0,s_1)
        self.canvas.create_line(int(s0)+15, int(s1)+15, int(s_0)+15, int(s_1)+15)
        t.sleep(1)
