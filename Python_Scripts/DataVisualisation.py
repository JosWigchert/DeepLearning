import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

f = open('C:\\Users\\jwi\\Downloads\\SmallDataset.txt', encoding="utf8")
lines = f.readlines()
f.close()

# initializing arrays
x=[]
y=[]
z=[]
time=[]
type=[]
i=0
iWalk = 0
iRun = 0
iCycle = 0
iStairs = 0


# splitting data and adding to arrays
print('Crunching data ...')
for line in lines:
    line = line.replace('\x00', '')
    split = line.split(',')
    if (len(split) >= 4):
        i += 1
        
       # time.append(i)
        type.append(int(split[3]))
        t = int(split[3])
        if (t == 0): 
            iWalk += 1
        elif (t == 1):
            iRun += 1
            time.append(iRun)
            x.append(int(split[0]))
            y.append(int(split[1]))
            z.append(int(split[2]))
        elif (t == 2):
            iCycle += 1
        elif (t == 3):
            iStairs += 1

axis=[[x], [y], [z]]

print("Walk samples:", iWalk)
print("Run samples:", iRun)
print("Cycle samples:", iCycle)
print("Stairs samples:", iStairs)

# plotting data in a graph
plt.plot(time, x, label = "X acceleration")
plt.plot(time, y, label = "Y acceleration")
plt.plot(time, z, label = "Z acceleration")
#plt.plot(time, type, label = "Type")
#plt.xlim(0, 20000)

# naming the x axis 
plt.xlabel('Time (s)') 
# naming the y axis 
plt.ylabel('Sensor Values') 
# giving a title to my graph 
plt.title('acceleration') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 