import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# plt.style.use('grayscale')
# plt.style.use('despat.mplstyle')
# plt.style.use('despat_dark.mplstyle')

COLOR_1 = "#fde725" #"#3b528b"
COLOR_2 = "#29798e" #"#81d34d"
COLOR_3 = "#81d34d" #"#81d34d"
COLOR_4 = "#666666" #"#81d34d"

INPUT_FILE = "open_images_all_graphs.txt"

MAX_WIDTH = 10
MAX_DEPTH = 10

TOTAL_NUMBER_IMAGES = 41602

global_graphs = {}
graph_families = np.zeros([MAX_WIDTH+1, MAX_DEPTH+1], dtype=np.int)
images_with_false_positives = 0

with open(INPUT_FILE, "r") as f:
    for line in f.readlines():
        if line.endswith(" \n"):
            line = line[:-2]
        elements = line.split(" ")

        if len(elements) > 1:
            valid_false_positive = False




            for e in elements[1:]:

                res = e.split("|")
                graph = res[0]
                width = int(res[1])
                depth = int(res[2])

                if width > MAX_WIDTH:
                    continue
                if depth > MAX_DEPTH:
                    continue

                valid_false_positive = True

                graph_families[width, depth] += 1

                if graph in global_graphs:
                    global_graphs[graph] += 1
                else:
                    global_graphs[graph] = 1

            if valid_false_positive:
                images_with_false_positives += 1

# for graph in global_graphs.keys():
#     print("{} {}".format(graph, global_graphs[graph]))

np.set_printoptions(precision=2, suppress=True)

print(graph_families)

print("total FP: {} | images with FPs: {} / {:5.2f}".format(np.sum(graph_families), images_with_false_positives, (images_with_false_positives/TOTAL_NUMBER_IMAGES)*100))

print("\nFPs per depth:")
print(np.sum(graph_families, axis=0))
print(np.multiply(np.divide(np.sum(graph_families, axis=0), np.sum(graph_families)), 100))

print("\nProbability for FP per depth:")
print(np.multiply(np.divide(np.sum(graph_families, axis=0), TOTAL_NUMBER_IMAGES), 100))
print("\n")

graph_families = np.divide(graph_families, np.sum(graph_families))
# graph_families = np.multiply(graph_families, 100)

# print(graph_families)

# data = graph_families[1:7, 2].tolist()
# data += graph_families[1:7, 3].tolist()


data_depth2 = graph_families[1:11, 2].tolist()
data_depth3 = graph_families[1:11, 3].tolist()
data_depth4 = graph_families[1:11, 4].tolist()

x_pos = np.arange(len(data_depth2))

# Build the plot
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)

# bar = ax.bar(x_pos, data, align='center', color=COLOR_4, edgecolor="#999999")

bar = ax.plot(x_pos, data_depth2, color=COLOR_4, marker="o")
bar = ax.plot(x_pos, data_depth3, color=COLOR_3, marker="s")
bar = ax.plot(x_pos, data_depth4, color=COLOR_2, marker="P")

# bar[0].set_color(COLOR_1)
# bar[1].set_color(COLOR_2)
# bar[2].set_color(COLOR_3)

xlabels = [str(x) for x in range(1, MAX_WIDTH+1)]

ax.set_ylabel('false positives per image', labelpad=5)
ax.set_xlabel('graph width', labelpad=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(xlabels)
ax.set_ylim(0, 0.13)
# plt.yscale("log")
# ax.set_title('phone model and android version')
ax.yaxis.grid(True)

custom_lines = [
    Line2D([], [], color=COLOR_4, marker='o', linestyle='None', markersize=10, label='depth 2'),
    Line2D([], [], color=COLOR_3, marker='s', linestyle='None', markersize=10, label='depth 3'),
    Line2D([], [], color=COLOR_2, marker='P', linestyle='None', markersize=10, label='depth 4'),
]

ax.legend(handles=custom_lines, borderpad=0.75, frameon=True)

# Save the figure and show
plt.tight_layout()
plt.savefig('plot_falsepositives.png', transparent=False)
# plt.show()