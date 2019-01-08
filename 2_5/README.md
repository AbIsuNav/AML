
## How to load the tree with leaf samples

Here you are provided a tree in the newick format that was used in 2.3, you are provided the pickled tree object and the tree in raw format(if you wish to parse it manually). The tree has hidden observations for all nodes except the leaf nodes, the nodes with hidden observations has the sampled value replaced with 'Xv' as to represent the stochastic variable notation used in the assignment.

Parameters used to generate the tree:

k = 2, max_depth = 5, max_branching = 4
where
* k := the number of categorical variables
* max_depth:= maximum depth of the generated tree
* max_branching:= maximum branching factor of each node


Similarly as you did in 2.3 you load the tree using pickle
```
with open('tree_with_leaf_samples.pkl', 'rb') as handle:
    params = pickle.load(handle)
```
