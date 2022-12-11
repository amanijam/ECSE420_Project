# ECSE420_Project

## Setup
To run this project on your machine, you will need to install Ray, an open-source unified compute framework. You can follow the instruction on this website https://docs.ray.io/en/latest/ray-overview/installation.html. Note that since Ray is still in Beta for windows support, the code was implemented on MacOs.

## Sequential Solver
To run the sequential program, run the following:

```bash
python n-queens.py --n
```

The argument "n" corresponds to the board size N for an NxN chess board. 

## Parallel Solver
To run the parallel program, run the following:

```bash
python n-queens_parallel.py --n --h
```

The argument "n" corresponds to the board size N for an NxN chess board. In addition, "k" corresponds to the number of parallel processes. 
