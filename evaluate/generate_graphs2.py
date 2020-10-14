import copy
from datetime import datetime

MAX_WIDTH = 3
MAX_DEPTH = 5

class Node(object):

    def __init__(self, parent=None, depth=0):
        self.parent = parent
        self.children = []
        self.depth = depth

    def get_lhds_name(self):
        
        # name = str(self.depth)
        # # children_sorted = self.get_children() #sorted(self.get_children(), reverse=True)
        # children_sorted = sorted(self.children, reverse=True)
        # for c in children_sorted:
        #     name += c.get_lhds_name()
        # return name

        children_names = []
        for c in self.children:
            children_names.append(c.get_lhds_name())

        return_value = str(self.depth)
        for c in sorted(children_names, reverse=True):
            return_value += str(c)
        
        self.lhds_name = int(return_value)
        return self.lhds_name

    def get_leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            leaves = []
            for c in self.children:
                leaves += c.get_leaves()
            return leaves

    def compute_uniques(self, unique):

        self.is_unique = unique

        if not unique:
            for c in self.children:
                c.compute_uniques(False)
        else:
            children_names_hashed = {}
            for i in range(0, len(self.children)):
                c = self.children[i]
                name = c.lhds_name
                if name not in children_names_hashed:
                    children_names_hashed[name] = [i]
                else:
                    children_names_hashed[name] += [i]

            for indices in children_names_hashed.values():
                if len(indices) == 1:
                    self.children[indices[0]].compute_uniques(True)
                else:
                    for i in indices:
                        self.children[i].compute_uniques(False)

    def add_child(self):
        c = Node(parent=self, depth=self.depth+1)
        self.children.append(c)
        return c

    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    def __lt__(self, other):
        return int(self.get_lhds_name()) <= int(other.get_lhds_name())


def build(node):

    global global_index
    global global_count
    global global_last_save

    root = node.get_root()
    name = "0" + str(root.get_lhds_name())

    if name in global_index:
        return

    root.compute_uniques(True)
    leaves = root.get_leaves()

    num_unique_leaves = 0
    for l in leaves:
        if l.is_unique:
            if len(l.parent.children) == 1:
                num_unique_leaves += 1

    global_index[name] = {"unique_leaves": num_unique_leaves}
    global_count += 1

    for l in leaves:

        if l.depth == MAX_DEPTH:
            continue

        old_node = l
        for i in range(0, MAX_WIDTH):
            new_node = copy.deepcopy(old_node)
            del old_node
            c = new_node.add_child() 
            build(c)
            old_node = new_node

    if global_count % 1000 == 0:
        print(global_count)

    if global_count % 10000 == 0 and global_last_save != global_count:
        save()
        global_last_save = global_count

def save():

    FILENAME = "all_trees_width_{}_depth_{}.txt".format(MAX_WIDTH, MAX_DEPTH)

    with open(FILENAME, "w") as f:
        for item in global_index.keys():
            f.write(str(item))
            f.write(" ")
            f.write(str(global_index[item]["unique_leaves"]))
            f.write("\n")

    print("written to file: {}".format(FILENAME))
    print("took: {:5.2f}s".format((datetime.now()-timer_start).total_seconds()))
    print("total trees: {}".format(len(global_index.keys())))

timer_start = datetime.now()

global_index = {}
global_count = 0
global_last_save = 0

root = Node()
child = root.add_child()
# chold = child.add_child()

build(child)

for item in global_index.keys():
    print(item)

save()
