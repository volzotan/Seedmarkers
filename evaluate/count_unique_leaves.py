import os
import numpy as np

for width in range(2, 10):
    for depth in range(2, 5):

        filename = "all_trees_width_{}_depth_{}.txt".format(width, depth)

        if not os.path.exists(filename):
            continue

        with open(filename, "r") as f:

            num_at_least_four = 0

            unique_leaves = np.zeros([10, 1], dtype=np.int)

            lines = f.readlines()
            for line in lines:
                elements = line[:-1].split(" ")

                unique_leaves[int(elements[1])] += 1

                # lhds = elements[0]
                # for i in range(0, len(lhds)-1):
                #     if int(lhds[i+1])-int(lhds[i]) >= 2:
                #         print(lhds)
                #         exit()

        print("{} {} {}".format(width, depth, unique_leaves.tolist()))