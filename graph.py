import matplotlib.pyplot as plt
import csv

switcher = [
        'blue',
        'orange',
        'green',
        'red',
        'purple',
        'brown',
        'pink',
        'gray',
        'olive',
        'cyan'
    ]

lbl = [str(i) for i in range(0,10)]


name = 'output.txt'
fig, ax = plt.subplots()
scale = 50.0

with open(name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        label, x, y = int(row[0]), float(row[1]), float(row[2])
        ax.scatter(x, y, c=switcher[label], s=scale, label=lbl[label],
                   alpha=0.8, edgecolors='none')



# ax.legend()
ax.grid(True)

plt.show()