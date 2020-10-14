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

INPUT_FILE = "raw_values.tsv"

MARKER_NAMES = ["matrix", "color", "organic", "point", "circle"]
SAMPLE_SIZE = 30

obtrusive = np.zeros([SAMPLE_SIZE, len(MARKER_NAMES)])
aesthetic = np.zeros([SAMPLE_SIZE, len(MARKER_NAMES)])
acceptable = np.zeros([SAMPLE_SIZE, len(MARKER_NAMES)])

line_count = 0
with open(INPUT_FILE, "r") as f:
    for line in f.readlines():

        line_count += 1
        if line_count == 1:
            continue

        if line.endswith(" \n"):
            line = line[:-2]
        elements = line.split("\t")

        obtrusive[line_count-2, ] = [int(x) for x in elements[0:5]]
        aesthetic[line_count-2, ] = [int(x) for x in elements[6:11]]
        acceptable[line_count-2, ] = [int(x) for x in elements[12:17]]

data = np.average(obtrusive, axis=0)
print(data)

# Build the plot
fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(4, 8))

# bar_width = 0.25

# x_pos = list(range(0, len(MARKER_NAMES)))
# x_pos_left = [x-bar_width/2 for x in x_pos]
# x_pos_right = [x+bar_width/2 for x in x_pos]

# bar = axs[0].bar(x_pos_left, np.average(obtrusive, axis=0), bar_width, color=COLOR_2, edgecolor="#999999")
# bar = axs[1].bar(x_pos_left, np.average(aesthetic, axis=0), bar_width, color=COLOR_2, edgecolor="#999999")
# bar = axs[2].bar(x_pos_left, np.average(acceptable, axis=0), bar_width, color=COLOR_2, edgecolor="#999999")

# bar = axs[0].bar(x_pos_right, np.median(obtrusive, axis=0), bar_width, color=COLOR_3, edgecolor="#999999")
# bar = axs[1].bar(x_pos_right, np.median(aesthetic, axis=0), bar_width, color=COLOR_3, edgecolor="#999999")
# bar = axs[2].bar(x_pos_right, np.median(acceptable, axis=0), bar_width, color=COLOR_3, edgecolor="#999999")

boxplot_obtrusive   = axs[0].boxplot(obtrusive, sym="") # sym="" so no outliers as points (fliers) 
boxplot_aesthetic   = axs[1].boxplot(aesthetic, sym="")
boxplot_acceptable  = axs[2].boxplot(acceptable, sym="")

for item in ["boxes", "whiskers", "caps"]:
    plt.setp(boxplot_obtrusive[item], color="black")
    plt.setp(boxplot_aesthetic[item], color="black")
    plt.setp(boxplot_acceptable[item], color="black")

xlabels = MARKER_NAMES

axs[0].set_ylabel("obtrusive", labelpad=5)
axs[1].set_ylabel("aesthetic", labelpad=5)
axs[2].set_ylabel("acceptable", labelpad=5)
# axs[2].set_xlabel('graph width', labelpad=5)
# ax.set_xticks(x_pos)
axs[2].set_xticklabels(xlabels)
axs[0].set_ylim(0, 10)

for ax in axs:
    ax.yaxis.grid(True)

# # plt.yscale("log")
# # ax.set_title('phone model and android version')

# ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('plot_averagemedian.png', transparent=False)
# plt.show()