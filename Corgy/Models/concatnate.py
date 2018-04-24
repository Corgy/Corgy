filename = "MNIST_CNN_fc_weight_dup.txt"

file = open(filename, 'r')

lines = file.readlines()

row, col = [int(x) for x in lines[0].split()]

for i in range(10):
    if lines[i+1][-1] == '\n':
        lines[i+1] = lines[i+1][:-1]

start = row + 2
while start < len(lines):
    for i in range(10):
        lines[i+1] += " " + lines[start+i][:-1]
    start += row + 1

for i in range(10):
    if lines[i+1][-1] != '\n':
        lines[i+1] += '\n'

file = open("output.txt", 'w')
file.writelines(lines[:row+2])