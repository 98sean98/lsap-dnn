# Deep Neural Network for LSAP

The Linear Sum Assignment Problem (LSAP) is a classical combinatorial optimization problem.
The goal is to allocate a number of items (e.g. jobs) to a number of entities (e.g. people)
in such a way that the total cost of the allocation is minimized, or the total utility of the
allocation is maximized.

The LSAP is a special case of the more general assignment problem, in which the number of
items and the number of entities are equal. It is also a special case of the transportation
problem.

The LSAP is also known as minimum weight matching in bipartite graphs. It can be solved in
polynomial time, for example by the **Hungarian algorithm**, or by specialized algorithms for
bipartite graphs.

This repository contains an implementation of deep neural networks for solving the LSAP.
The implementation is based on the paper [Deep Neural Networks for Linear Sum Assignment Problems](https://ieeexplore.ieee.org/document/8371290), 
authored by [Mengyuan Lee](https://ieeexplore.ieee.org/author/37086553600), 
[Yuanhao Xiong](https://ieeexplore.ieee.org/author/37086552701),
[Guanding Yu](https://ieeexplore.ieee.org/author/37288513300),
and [Geoffrey Ye Li](https://ieeexplore.ieee.org/author/38516728900).

The design of the neural networks are accredited to the authors of the paper.
The original implementation was in Tensorflow 1.0.0,
but the implementation in this repository is a re-implementation in
Pytorch 1.13.1.

## Data Generation

Data generation is done by the script `sampling.py`.
The script generates a number of random LSAP cost matrices,
with individual solutions computed using the Hungarian algorithm.

The LSAP cost matrices, and solutions, are stored in `.h5` files in the `data` directory.

## Training

The script `training.py` trains a neural network for solving the LSAP.
Training is performed on a specified data file in the `data` directory.
The trained model is stored in the `artifacts` directory.

### Network architecture

As pointed out by the authors of the paper, LSAP is solved by
breaking it down into sub-assignment problems. 
Each sub-problem represents an assignment of a single item to a single entity.
That means that the number of entities is equal to the number of neural network models.

The input to the model is the complete LSAP cost matrix,
whereas the output represents which item is assigned to the specified entity.

## Results

During training, the training loss, test loss, and accuracy of each model are recorded. 
And preliminary results show that the implemented `LinearModel` is able to achieve
a test accuracy greater than 85% after 1000 epochs of training.
However, the testing step has not applied the greedy collision-avoidance rule,
which seems to be an important post-processing step to ensure that
the inferred output is adjusted to be a valid solution, i.e. the same job
is not assigned to multiple people.
