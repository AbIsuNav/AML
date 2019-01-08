## Please download the Forward and backward algorithm code from [2.7 Forward and Backward](https://gits-15.sys.kth.se/butepage/advML_HT18/blob/master/2_7/forward_backward.py)


## How to generate the data
Please use generator.py file if you want to generate the data with new parameters. Uncomment line 64     ```save_obj(output_sequences,"sequence_output") ``` to save the data dictionary in pickle format. I have written the script in a way so that it would be eaisier to understand for everyone. You are welcome to optimize or change it according to your requirements.

#### Note: 
The initial distribution when a pair starts playing is 1/4 and 3/4. Where, 1/4 is the probabilty that a pair will start from table_idx = 0 or left side and 3/4 is the probabilty that it will start from table_idx = 1 or right side. After the first step, the probability to be on the same side will be 1/4 and to change the side will be 3/4


## How to load sequence_output.pkl file
Use the following code to load the data file.
```
def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
sequence_outputs = load_obj("sequence_output")
```

The data file contains dictionary of dictionaries. The first level dictionary has keys as pairs e.g. (1,2) represents player 1 and player 2. Then each first level key has R keys which are rounds e.g [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and each r has the output sequence of observation. 
I have kept N = 20, M = 30 and R = 10.
You can see all the first level keys (player pairs)  using ```print sequence_outputs.keys()``` and rounds for pair (1,2) using ```print sequence_outputs[(1,2)].keys()```.

## Linear Solver
Please use the numpy linear solver ```numpy.linalg.solve``` if required.

* [Numpy Linear Solver](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html) - Solver documentation

#### Example: 
See the example code to solve the linear equations ```3 * x0 + x1 = 9``` and ```x0 + 2 * x1 = 8```:

#### Code
```
a = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(a, b)
print 'output = ',x

```
```
output = array([ 2.,  3.])
```
