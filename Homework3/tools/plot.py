import matplotlib.pyplot as plt
import csv
import sys

def plot(argv):
    if len(argv) != 2:
        print("Usage: python plot.py filename outputname")
        return
    (filename, outputname) = argv
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # remove the first row
        next(reader)
        x = []
        y = []
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
        plt.figure(figsize=(12, 6.75))
        plt.rcParams['font.size'] = '16'
        plt.plot(x, y, marker='o')
        plt.title("Identification Rate (Rank 1) - PC Curve")
        plt.xlabel("PC")
        plt.ylabel("Identification Rate (Rank 1)")
        plt.xscale('log')
        plt.yticks([i/10 for i in range(11)])
        plt.savefig(outputname)

if __name__ == '__main__':
    plot(sys.argv[1:])