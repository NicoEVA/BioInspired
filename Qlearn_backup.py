from math import ceil
from numpy.lib.function_base import _gradient_dispatcher
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#The agent can go in 4 directions
#up right down left
rows =24
cols = 24
stacks = 10 #attempt 3D
q_vals = np.zeros((rows,cols,stacks,6))
choices = np.zeros((rows,cols,stacks))

alt_rest1 = 7
restricted_airspace1 = [0,int(rows/2),0,int(cols/2),alt_rest1]
thermals1  = [10,20,7,18]



actions = ['up','right','down','left',"3ddown","3dup"]
rewards = np.full((rows,cols,stacks),-1)
for z in range(stacks-1):
    for x in range(cols-1):
        rewards[x,0,z]=-100
        rewards[x,rows-1,z]=-100       
    for y in range(rows-1):
        rewards[0,y,z]=-100
        rewards[cols-1,y,z]=-100
    for x in range(thermals1[0],thermals1[1]):
        for y in range(thermals1[2],thermals1[3]):
            rewards[x,y,z] = -0.2
for x in range(cols-1):
    for y in range(rows-1):
        for z in [0,stacks-1]:
            rewards[x,y,z] = -100

# establish restricted airspace
for z in range(alt_rest1,stacks):
    for x in range(int(restricted_airspace1[0]),restricted_airspace1[1]):
        for y in range(restricted_airspace1[2],restricted_airspace1[3]):
            rewards[x,y,z] = -100

#Defining where the goal/target point is
rewards[2,2,3]=100            














def terminal(row,col,stack):
    if rewards[row,col,stack]==-100:
        return True
    elif rewards[row,col,stack]==100:
        return True
    else:
        return False

def startloc():
    row = np.random.randint(rows)
    col = np.random.randint(cols)
    stack = np.random.randint(stacks)
    while terminal(row,col,stack):
        row = np.random.randint(rows)
        col = np.random.randint(cols)
        stack = np.random.randint(stacks)
    return row,col,stack

def findact(row,col,stack,eps):
    if np.random.random() < eps:
        return np.argmax(q_vals[row,col,stack])
    else: 
        return np.random.randint(6)

def newloc(row,col,stack,act):
    newrow = row
    newcol = col
    newstack = stack
    if actions[act]=="up" and row > 0:
        newrow -= 1
    elif actions[act] == 'right' and col < cols-1:
        newcol += 1
    elif actions[act] == 'down' and row < rows-1:
        newrow += 1
    elif actions[act] == 'left' and col > 0:
        newcol -= 1
    elif actions[act] == "3ddown" and stack > 0:
        newstack -=1
    elif actions[act] == "3dup" and stack < stacks -1:
        newstack += 1
    return newrow,newcol,newstack

def extrsp(srow,scol,sstack):
    if terminal(srow,scol,sstack):
        return []

    else: 
        row, col, stack = srow, scol, sstack
        sp = []
        sp.append([row,col,stack])
        print(sp)
        while not terminal(row,col,stack):
            aind = findact(row,col,stack,1.)
            
            row, col, stack = newloc(row,col,stack,aind)
            sp.append([row,col,stack])
            
        return sp

def generatechoices(q_vals,choices):
    for i in range(np.shape(q_vals)[0]):
        for j in range(np.shape(q_vals)[1]):
            for k in range(np.shape(q_vals)[2]):
                choice = np.argmax(q_vals[i,j,k])
                choices[i,j,k] = choice
    return choices




epsilon = 0.3
discount = 0.8
lrate    = 0.85
ep = 0
for episode in range(25000):
    row,col,stack = startloc()
    tries = 0 
    while not terminal(row,col,stack):
        aind = findact(row,col,stack,epsilon)
        orow, ocol, ostack = row,col,stack
        row, col, stack = newloc(row,col,stack,aind)

        rew = rewards[row,col,stack]
        oq = q_vals[row,col,stack,aind]
        tempdiff = rew + (discount*np.max(q_vals[row,col,stack]))-oq

        newq = oq +(lrate*tempdiff)
        q_vals[orow,ocol,ostack,aind] = newq
        if tries % 10 == 0: 
            print(tries)
        tries +=1
    print("Reached target in EP " + str(ep+1))
    ep += 1
print("trained successfully")



data = extrsp(20,20,7)


x = [xi[0] for xi in data]
y = [yi[1] for yi in data]
z = [zi[2] for zi in data]


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label='Agent Trace')

ceilingx = np.arange(restricted_airspace1[0],restricted_airspace1[1],1)
ceilingy = np.arange(restricted_airspace1[2],restricted_airspace1[3],1)

wally1 = np.zeros_like(ceilingx)
wally1.fill(restricted_airspace1[2])
wally2 = np.zeros_like(ceilingx)
wally2.fill(restricted_airspace1[3]-1)
w1x, w1y = np.meshgrid(ceilingx,wally1)
w2x, w2y = np.meshgrid(ceilingx,wally2)


Zwallsx = np.zeros((np.shape(ceilingx)[0],np.shape(wally1)[0]))
for len in range(np.shape(Zwallsx)[1]):
    Zwallsx[len][:] = alt_rest1 + len

X,Y = np.meshgrid(ceilingx,ceilingy)
Z = np.ones((np.shape(ceilingx)[0],np.shape(ceilingy)[0]))
Z.fill(alt_rest1)

thermalsx = np.arange(thermals1[0],thermals1[1]+1,1)
thermalsy = np.arange(thermals1[2],thermals1[3]+1,1)

Xt, Yt = np.meshgrid(thermalsx,thermalsy)
Zt = np.zeros((np.shape(thermalsy)[0],np.shape(thermalsx)[0]))
ax.plot_surface(Xt,Yt,Zt,color="Green",alpha=0.4)
ax.plot_surface(X,Y,Z,color="Red",alpha=0.4)
ax.plot_surface(w1x,w1y,Zwallsx,color="Red",alpha=0.4)
ax.plot_surface(w2x,w2y,Zwallsx,color="Red",alpha=0.4)

ax.legend()

plt.show()

# cs = generatechoices(q_vals,choices)
# #Create Trend 
# xp,yp,zp = np.meshgrid(range(rows),range(cols),range(stacks))

# u = np.zeros((rows,cols,stacks))
# v = np.zeros((rows,cols,stacks))
# w = np.zeros((rows,cols,stacks))

# for i in range(rows-1):
#     for j in range(cols-1):
#         for k in range(stacks-1):
#             if cs[i,j,k] ==  0:
#                 u[i,j,k] = 1
#             elif cs[i,j,k] ==  1:
#                 v[i,j,k] = 1
#             elif cs[i,j,k] == 2:
#                 u[i,j,k] =-1
#             elif cs[i,j,k] == 3:
#                 v[i,j,k] = -1
#             elif cs[i,j,k] == 4:
#                 w[i,j,k] = -1
#             elif cs[i,j,k] == 5:
#                 w[i,j,k] = 1
#             # elif rewards[i,j,k]==-100:
#             #     xp[i,j,k]=0
#             #     yp[i,j,k]=0
#             #     zp[i,j,k]=0
# fig = plt.figure(figsize=(30, 30))
# ax = fig.gca(projection='3d')

         
# ax.quiver(xp, yp, zp, u, v, w, length=0.5, normalize=True)

# plt.show()


