class Node:

    def __init__(self, name, cat):
        self.name = name
        self.cat = []
        for c in cat:
            self.cat.append(c)
        self.ancestor = None
        self.descendants = []
        self.sample = None


def load_params(params):
    """
        Creates the root and adds children recursively .
    """
    root = Node(params[0], params[1])
    add_children(root, params[2])
    return root


def add_children(node, params):
    """
        Adds the children and their children recursively to "node".

    """
    if len(params) == 0:
        return
    num_children = len(params)
    for child in range(0, num_children, 3):
        child_node = Node(params[child], params[child + 1])
        child_node.ancestor = node
        node.descendants.append(child_node)
        add_children(child_node, params[child + 2])


def load_sample(root, sample):
    """
        Loads a sample in our python / Newick format into the tree nodes.
    """
    root.sample = sample[1]
    load_sample_rec(root, sample[2])


def load_sample_rec(node, sample):
    """
        Recursive function to load a sample in our python /
        Newick format into the tree nodes.
    """
    if len(sample) == 0:
        return
    num_children = len(sample)
    c = 0
    for child in range(0, num_children, 3):
        node.descendants[c].sample = sample[child + 1]
        load_sample_rec(node.descendants[c], sample[child + 2])
        c = c + 1


def print_tree(root, print_sample=False):
    """
        Prints tree layer by layer without correct spacing for children.
        print_sample (bool) determines whether we also print the current
        sample.
    """
    curr_layer = [root]
    while curr_layer != []:
        string = ''
        next_layer = []
        for elem in curr_layer:
            if elem.ancestor != None:
                string = string + '{0}_pa{1} '.format(elem.name, elem.ancestor.name)
            else:
                string = string + elem.name + ' '
            if (print_sample and elem.sample != None):
                string = string[:-1] + ':' + str(elem.sample) + ' '
            for child in elem.descendants:
                next_layer.append(child)
        print(string)
        curr_layer = next_layer
