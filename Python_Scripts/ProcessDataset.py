

f = open('SmallDataset.TXT', 'r')
line = f.read()
f.close()

line = line.replace('\x00', '')

f = open('SmallDataset.TXT', 'w')
f.write(line)
f.close()