import random, math, time
import numpy as np
import ray
from operator import attrgetter
import argparse

# Data structure that holds information about any board state
# positions:    1D size n int array, holding the row of the queen in each of the n columns
# foward_diags: 1D size n int array, holding value of row+col
# back_diags:   1D size n int array, holding value of row-col
# heuristic:    number of constraint violations
class BoardState:
    def __init__(self, positions, forward_diags, back_diags):
        self.positions = positions 
        self.forward_diags = forward_diags
        self.back_diags = back_diags
        self.heur = self.evaluate_state()

    def updateHeuristic(self, h):
        self.heur = h
    
    # Evaluates a state by calculating the number of constraint violations
    # Two queens in the same row, forward, or backward diagonal = constraint violation
    def evaluate_state(self):
        pos_count = {i: self.positions.count(i) for i in self.positions}
        pos_num_duplicates = num_duplicates(pos_count)

        forward_diag_count = {
            i: self.forward_diags.count(i) for i in self.forward_diags
        }
        forward_diag_num_duplicates = num_duplicates(forward_diag_count)

        back_diag_count = {i: self.back_diags.count(i) for i in self.back_diags}
        back_diag_num_dulpicates = num_duplicates(back_diag_count)

        heur = pos_num_duplicates + forward_diag_num_duplicates + back_diag_num_dulpicates
        return heur

# Counts the number of duplicate values in the given dict
def num_duplicates(dict):
    duplicate_count = 0
    for i in dict:
        if dict[i] > 1:
            duplicate_count += dict[i] - 1
    return duplicate_count

# Returns all neighbor states for states[i]
@ray.remote    
def generateNeighbors(states: list[BoardState], i, n):
    s = states[i]
    neighbors = []
    for col in range(n):
        curr_row = s.positions[col]
        for row in range(n):
            if row != curr_row:
                next_pos = s.positions[:]
                next_pos[col] = row

                next_forward_diags = s.forward_diags[:]
                next_forward_diags[col] = row + col

                next_back_diags = s.back_diags[:]
                next_back_diags[col] = row - col

                next_state = BoardState(next_pos, next_forward_diags, next_back_diags)
                neighbors.append(next_state)
    return neighbors

# Returns list of probabilities p = e^(heuristic/temp)
@ray.remote
def listProbs(states: list[BoardState], t, i, elements_per_process):
    p = []
    start_index = i*elements_per_process
    end_index = start_index+elements_per_process
    x = start_index
    len_states = len(states)
    while x < end_index and x < len_states:
        s = states[x]
        p.append(math.exp(s.heur / t))
        x += 1
    return p

# Noramlizes the lsit of probabilites (to sum up to 1)
@ray.remote
def normalizeProbs(probs, normalizer, i, elements_per_process):
    normal_p = []
    start_index = i*elements_per_process
    end_index = start_index+elements_per_process
    x = start_index
    len_probs = len(probs)
    while x < end_index and x < len_probs:
        p = probs[x]
        normal_p.append(p * normalizer)
        x += 1
    return normal_p

# n: Board size (board is nxn)
# k: number of parallel processes
class NQueens_ParallelProblemSolver:
    def __init__(self, n, k=1):
        self.n = n 
        self.k = k

    # Randomly generates a state (N x N board configuration)
    def generate_init_state(self) -> BoardState:
        n = self.n
        positions = []
        forward_diags = []
        back_diags = []

        for i in range(n):
            positions.append(random.randint(0, n - 1))
            forward_diags.append(positions[i] + i)
            back_diags.append(positions[i] - i)
    
        return BoardState(positions, forward_diags, back_diags)

    def solveParallel(self) -> BoardState:
        n = self.n
        k = self.k

        # Define temperature and cooling values
        # Ensure initial temp is large enough to generate a high initial probability (abbout 0.999)
        t_init = 1000 
        alpha = 0.9  # alpha is the factor used to decrement the temperature (exponential cooling)
        t_k = t_init
        t_threshold = 0.0001
        max_step = 1000  # number of trials per temperature step
        step_count = 0

        # List of current k states and k processes
        states = []
        for _ in range(k):
            states.append(self.generate_init_state())
            
        best_state = states[0]

        while t_k > t_threshold and best_state.heur != 0:       
            ray_states = ray.put(states)

            # generate all neighbors for each of the k states (in parallel)
            ray_neighbors2d = [generateNeighbors.remote(ray_states, i, n) for i in range(k)]
            allNeighbors2d = ray.get(ray_neighbors2d)

            # update best state
            for x in range(k):
                bestN = min(allNeighbors2d[x], key=attrgetter('heur'))
                if bestN.heur == 0: return bestN
                if bestN.heur < best_state.heur:
                    best_state = bestN

            allNeighbors = list(np.concatenate(allNeighbors2d).flat)
            ray_neighbors = ray.put(allNeighbors)

            elements_per_process = int(len(allNeighbors)/k) + 1

            # generate a list of probabilities proportional to e^(heuristic/temp) (in parallel)
            ray_allProbs2d = [listProbs.remote(ray_neighbors, t_k, i, elements_per_process) for i in range(k)]
            allProbs2d = ray.get(ray_allProbs2d)
            allProbs = list(np.concatenate(allProbs2d).flat)
            ray_allProbs = ray.put(allProbs)

            # normalize the list of probabilities (in parallel)
            p_sum = sum(allProbs)
            normalizer = 1/p_sum
            ray_normalProbs2d = [normalizeProbs.remote(ray_allProbs, normalizer, i, elements_per_process) for i in range(k)]
            normalProbs2d = ray.get(ray_normalProbs2d)
            normalProbs = list(np.concatenate(normalProbs2d).flat)
        
            # randomly choose k neighbor states with proability proportional to e^(heuristic/temp)
            states = np.random.choice(allNeighbors, size=k, replace=False, p=normalProbs)

            # update best state
            for s in states:
                if s.heur < best_state.heur: 
                    best_state = s

            # Decrease temperature
            step_count += 1
            if step_count >= max_step:
                step_count = 0
                t_k = t_k * alpha

        return best_state

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="size N for the NxN board")
    parser.add_argument("--k", type=int, default=4, help="number of parallel tasks")
    args = parser.parse_args()
    return args


def main():
    ray.init()

    args = get_args()
    start = time.perf_counter()
    solver = NQueens_ParallelProblemSolver(args.n, args.k)
    solution = solver.solveParallel()
    end = time.perf_counter()

    print("Elapsed time = " + str(end - start))
    print("Solution: " + str(solution.positions))
    print("Heuristics: " + str(solution.heur))

if __name__ == "__main__":
    main()
