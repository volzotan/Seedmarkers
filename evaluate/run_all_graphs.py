import subprocess
import os
from datetime import datetime

BASE_PATH = "../generate"
SCRIPT = "tmap.py"
INPUT_DXF = "example_shapes/circle.dxf"
OUTPUT_DXF = "allgraphs/{}.dxf"
OUTPUT_PNG = "allgraphs/{}.png"

GRAPH_FILENAME = "all_trees_width_3_depth_4.txt" 
RESULT_FILE = GRAPH_FILENAME + "_results.txt"

invocation = "python3 {} {} {} --output {} --output-image {} > /dev/null"

lines = []
with open(GRAPH_FILENAME, "r") as f:
    lines = f.readlines()

# lines = sorted(lines, reverse=True)

with open(RESULT_FILE, "w") as f:
    os.chdir(BASE_PATH)

    for line in lines:

        timer_start = datetime.now()

        graph = None
        if line[-1] == "\n":
            graph = line[:-1].split(" ")[0]
        else:
            graph = line.split(" ")[0]

        if len(graph) == 0:
            print("empty graph")
            exit(-1)

        output_filename = OUTPUT_DXF.format(graph)
        output_image_filename = OUTPUT_PNG.format(graph)

        result = subprocess.run(invocation.format(SCRIPT, INPUT_DXF, graph, output_filename, output_image_filename), shell=True, check=True)

        timer_diff = datetime.now()-timer_start

        l = "{:40} {:20.3f} {:2}\n".format(graph, timer_diff.total_seconds(), result.returncode)
        f.write(l)
        f.flush()

        print(l, end="")
