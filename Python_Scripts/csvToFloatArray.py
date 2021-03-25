f = open('balanced_data_50hz_Walk_Run_Assesmentwk7.csv', encoding="utf8")
lines = f.readlines()
f.close()


walking=[]
running=[]
cycling=[]
stairs=[]


# splitting data and adding to arrays
print('Crunching data ...')
for i,line in enumerate(lines):
    try:

        line = line.replace('\x00', '')

        split = line.split(',')

        if (len(split) >= 4):
            if (int(split[0]) <= 0):
                walking.append(int(split[1]))
                walking.append(int(split[1]))
                walking.append(int(split[1]))
            elif (int(split[0]) <= 1):
                running.append(float(split[1]))
                running.append(float(split[2]))
                running.append(float(split[3]))
            elif (int(split[0]) <= 2):
                cycling.append(float(split[1]))
                cycling.append(float(split[2]))
                cycling.append(float(split[3]))
            elif (int(split[0]) <= 3):
                stairs.append(float(split[1]))
                stairs.append(float(split[2]))
                stairs.append(float(split[3]))
            
    except:
        print('error at line number: ',i )

print("Making arrays")
print("Walking", len(walking))
print("Running", len(running))
print("Cycling", len(cycling))
print("Stairs", len(stairs))

for i in range(int(len(walking)/300)):
    print("Array", i)
    createArray(walking[i*300:300], ("walking_"+i))


def createArray(data, name):
    print(len(data))
    print(data)
    return