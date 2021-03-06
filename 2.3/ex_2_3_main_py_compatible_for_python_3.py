import pickle
from ex_2_3 import Node, load_params, load_sample, print_tree
from ex_2_3_tree_helper import Tree
import os

"""
    The data is stored in Newick form:
        [A,B,[C,D]E]F;
        
           ________A 
          |
        F |________B
          |          ________C
          |         |
          |________E
                    |________D
                    
    But we are working in python. So we will work with lists that look
    the following way:
        
    Tree  = ['F', pF, ['A', pA, [], 'B', pB, [], 'E', pE, ['C', pC, [], 'D', pD, []]]]
        
    Each variable has a name (in string format), a list of categorical parameters,
    e.g. pF = [0.3, 0.2, 0.5], and a list of child nodes.
"""

"""
  Load the parameters of each tree and three samples from it.
  The data is stored in a dictionary. 
  
  The key indicates
      k := the number of categorical variables
      md:= maximum depth of the generated tree
      mb:= maximum branching factor of each node
      alpha:= the alpha values used to draw the parameters from a Dirichlet
  
  None of these numbers is important for your implementation but cab be 
  used for interpretation.
"""

my_data_path = '' # os.path.dirname(os.path.realpath(__file__)) + '\\'

with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
    params = pickle.load(handle, encoding='latin1')

with open(my_data_path + 'tree_samples.pickle', 'rb') as handle:
    samples = pickle.load(handle, encoding='latin1')

"""
    Construct a tree with parameters from the loaded parameter dict.
"""
params_name = list(params)[0]

#for i in range(len(params_name)):
print("len: ", params_name)
params = params[params_name]
root = load_params(params)
print("root", root.cat[:])
#for k in range(1,4):
"""
            Load a matching sample into the tree.
"""
samples_name = params_name + '_sample_1'
sample = samples[samples_name]
load_sample(root, sample)

"""
Print the tree (not very sophisticated). Structure: nodename_parentname
"""
print("this is the root:  ")
print_tree(root)
print(" end")

"""
Print the tree with sample (not very sophisticated). Structure: nodename_parentname:sample
"""
# print_tree(root, print_sample=True)

"""
Use tree object:
"""

t = Tree()

my_data_path = '' #os.path.dirname(os.path.realpath(__file__)) + '\\'

with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
    params = pickle.load(handle, encoding='latin1')

key = list(params.keys())[0]
print(key)
# """
# Load params into tree
# """
# t.load_params(params[key])
# t.print_tree()
#
# """
# Generate a random tree
# """
# t.create_random_tree(3)
# t.print_tree()
