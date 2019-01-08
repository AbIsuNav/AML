import pickle
import random
from ex_2_5 import Node, load_params, load_sample, print_tree
import numpy as np
"""
Assignment 2. 
Problem 2.5
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


def likelihood(root, sample):
    """
    Gets the likelihood at node root
    :param root: object node to evaluate
    :param sample: used category for the parent
    :return: float total likelihood
    """
    tot_sum = 0.0  # result
    prob_dictionary = dict()  # values of the tree probabilities this saves for improving computation
    for i in range(len(root.cat[sample])):
        root_in_i = s(root, i, prob_dictionary) * root.cat[sample][i]
        tot_sum += root_in_i
    return tot_sum


def random_sample(node, index):
    """
    the sampling function for each node from the posterior
    :param node: object node, node to evaluate
    :param index: the category from the parent
    :return: (int) the chosen category, (float) the posterior value
    """
    c = random.randint(0,1)
    phy = node.cat[index][c] * likelihood(node, index)
    if phy > random.random():
        print_L.append(likelihood(node, index))
        return c, phy
    else:
        if c==0:
            c=1
        else: c=0
        print_L.append(likelihood(node, index))
        return c, node.cat[index][c] * likelihood(node, index)


def Sampling_posterior(node, cat_vector, dictionary):
    """
    Iterative function starting from node to the leaves and calling random_sample to obtain the posterior and
    final samples of the chosen categories for each node
    :param node: object node, node from which we get the posterior
    :param cat_vector: array to save the used category at node
    :param dictionary: dictionary having the name of the node and its posterior
    :return: dictionary and cat_vector
    """
    key = str(node.name)
    if not node.descendants: #leaf
        index = int(cat_vector[int(node.ancestor.name) - 1])
        phy = node.cat[index][node.sample]
        # if phy <= random.random():
        #     cat_vector[int(node.name) - 1] = 0
        # else:
        #     cat_vector[int(node.name) - 1] = 1
        #     phy = node.cat[index][1]
        dictionary[str(node.name)] = phy  # saves the posterior of the node
    elif node.ancestor: # vertex
        index = int(cat_vector[int(node.name) - 1])  # the parent value used: 1 or 0
        for child in node.descendants:
            cat_vector[int(child.name) - 1], dictionary[str(child.name)] = random_sample(child, index)  # saves the
            # chosen sample and the posterior of the node
            Sampling_posterior(child, cat_vector, dictionary)

    else:  # this is just for the root
        index = random.randint(0, 1)  # choose random sample from root
        cat_vector[int(node.name) - 1] = index
        dictionary[key] = node.cat[0][index] * likelihood(node, 0)  # this is root
        print_L.append(likelihood(node, 0))
        for child in node.descendants:
            cat_vector[int(child.name) - 1], dictionary[str(child.name)] = random_sample(child, index)  # saves the
            # chosen sample and the posterior of the node
            Sampling_posterior(child, cat_vector, dictionary)
    return dictionary, cat_vector


# --- main --- #
my_data_path = ''  # os.path.dirname(os.path.realpath(__file__)) + '\\'
with open(my_data_path + 'tree_with_CPD.pkl', 'rb') as handle:
    params = pickle.load(handle, encoding='latin1')

with open(my_data_path + 'tree_with_leaf_samples.pkl', 'rb') as handle:
    samples = pickle.load(handle, encoding='latin1')
print_L =[]
print(params)  # conditional probabilities
root = load_params(params)
load_sample(root, samples)
dictionary = dict()
binary_vector = np.zeros(54, dtype=int)  # total elements in the tree
posterior_dictionary, samples_vector = Sampling_posterior(root, binary_vector, dictionary)
print("Node ", "Likelihood     ", "  Posterior")

for i,val in posterior_dictionary.items():
    print(i,"& ",print_L[int(i)-1],"& ", val," \ \ ")
print(samples_vector)





