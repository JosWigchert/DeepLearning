import pandas as pd

f = open('WalkingRunningNew.txt', encoding="utf8")
lines = f.readlines()
f.close()


# initializing arrays
x=[]
y=[]
z=[]
time=[]
type=[]
i=0


# splitting data and adding to arrays
print('Crunching data ...')
for i,line in enumerate(lines):
    try:
        line = line.replace('\x00', '')
        split = line.split(',')
        if (len(split) >= 4):
            i += 1
            x.append(int(split[0]))
            y.append(int(split[1]))
            z.append(int(split[2]))
            time.append(i)
            type.append(int(split[3])) #* 500)
    except:
        print('error at line number: ',i )

df = pd.DataFrame() 
df['Activity'] = type
#df['time'] = np.arange(0,len(x))
df['X data'] = x
df['Y data'] = y
df['Z data'] = z

#check for null values in dataset
#print(df.isnull().sum())

#Check activity distribution
print(df['Activity'].value_counts())

#Balance the dataset
df['X data'] = df['X data'].astype('float')#Convert X data to float
df['Y data'] = df['Y data'].astype('float')#Convert Y data to float
df['Z data'] = df['Z data'].astype('float')#Convert Z data to float
#df.info()

# time between measurements is 5 ms thus Fs = 200

activities = df['Activity'].value_counts().index

#Switch codes Walking = 00 Running = 01  Cycling = 10 Stairs= 11
#Walking = 0
#Running = 1
#Cycling = 2
#Stairs = 3

#Smallest dataset is 9540 samples(for walking) thus we will only use the same amount of samples for each activity 
lowestSampleCount = min(df['Activity'].value_counts())
print(lowestSampleCount)
Walking = df[df['Activity'] == 0].head(lowestSampleCount).copy()
Running = df[df['Activity'] == 1].head(lowestSampleCount).copy()
Cycling = df[df['Activity'] == 2].head(lowestSampleCount).copy()
Stairs = df[df['Activity'] == 3].head(lowestSampleCount).copy()


NewWalking = Walking.iloc[::4, :]
NewRunning = Running.iloc[::4, :]
NewCycling = Cycling.iloc[::4, :]
NewStairs  =  Stairs.iloc[::4, :]

NewWalking = NewWalking.append(Walking.iloc[1::4, :])
NewRunning = NewRunning.append(Running.iloc[1::4, :])
NewCycling = NewCycling.append(Cycling.iloc[1::4, :])
NewStairs  =  NewStairs.append( Stairs.iloc[1::4, :])

NewWalking = NewWalking.append(Walking.iloc[2::4, :])
NewRunning = NewRunning.append(Running.iloc[2::4, :])
NewCycling = NewCycling.append(Cycling.iloc[2::4, :])
NewStairs  =  NewStairs.append( Stairs.iloc[2::4, :])

NewWalking = NewWalking.append(Walking.iloc[3::4, :])
NewRunning = NewRunning.append(Running.iloc[3::4, :])
NewCycling = NewCycling.append(Cycling.iloc[3::4, :])
NewStairs  =  NewStairs.append( Stairs.iloc[3::4, :])

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([NewWalking, NewRunning, NewCycling, NewStairs])

balanced_data.to_csv(r'balanced_data_50hzV2.csv',index = False)

print(balanced_data['Activity'].value_counts())




