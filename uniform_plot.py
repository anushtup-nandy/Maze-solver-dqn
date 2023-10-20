import numpy as np
import matplotlib.pyplot as plt

y=0
l=[]
for i in range(200):
    y=np.random.uniform(0.05, 0.95)
    l.append(y)
    
x=[i for i in range(len(l))]
plt.scatter(x,l)
plt.show()