import pickle
from ex_2_3 import Node, load_params, load_sample, print_tree

"""
Assignment 2. 
Problem 2.3
Author: Abgeiba Isunza Navarro
"""


def s(node, level, prob_dictionary):
    """
    Iterative function from eq. 2
    :param node: node object to evaluate
    :param level: level at the tree from root
    :param prob_dictionary: dictionary with node name and its likelihood
    :return: likelihood
    """
    prob = str(node.name) + "-" + str(level)  # this is the key to save the value of the node in s_vals
    if not node.descendants:  # if it is a leaf node
        if node.sample == level:
            return 1
        else:
            return 0
    if prob in prob_dictionary:  # if the value is already saved. The node has been already visited
        return prob_dictionary[prob]
    else:
        conditional_l = 1.0
        marginal = 0.0
        for child in node.descendants:
            for index_of_cat in range(len(child.cat[level])):  # gets the likelihood conditioned to parent_index = i
                s_ = s(child, index_of_cat, prob_dictionary)
                marginal += child.cat[level][index_of_cat] * s_
            conditional_l *= marginal
            prob_dictionary[prob] = conditional_l
            marginal = 0.0
    return conditional_l


def likelihood(root):
    tot_sum = 0.0  # result
    prob_dictionary = dict()  # values of the tree probabilities this saves for improving computation
    for i in range(len(root.cat[0])):
        root_in_i = s(root, i, prob_dictionary) * root.cat[0][i]
        tot_sum += root_in_i
    return tot_sum


# -- main:

my_data_path = ''  # os.path.dirname(os.path.realpath(__file__)) + '\\'

for i in range(26):  # because we have 27 trees
    with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
        params = pickle.load(handle, encoding='latin1')

    with open(my_data_path + 'tree_samples.pickle', 'rb') as handle:
        samples = pickle.load(handle, encoding='latin1')

    params_name = list(params.keys())[i]
    params = params[params_name]
    root = load_params(params)
    print("Tree #" + str(i+1))
    for k in range(3):
        """
                    Load a matching sample into the tree.
        """
        samples_name = params_name + '_sample_' + str(k + 1)  # load 1-3 samples of each tree
        sample = samples[samples_name]
        load_sample(root, sample)

        print("sample "+str(k+1)+": "+str(likelihood(root)))
