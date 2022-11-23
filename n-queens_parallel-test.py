import random, math, time
import numpy as np
import multiprocessing as mp
import ray

# Data structure that holds information about any board state
# positions:    1D size n int array, holding the row of the queen in each of the n columns
# foward_diags: 1D size n int array, holding value of row+col
# back_diags:   1D size n int array, holding value of row-col
# heuristic
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

@ray.remote
def generateNeighbors(states: list[BoardState], i: int, n):
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
                #next_state_heur = self.evaluate_state(next_state)
                #next_state.addHeuristic(next_state_heur)
                neighbors.append(next_state)

    return neighbors
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
        t_init = 1000 * (n // 2)
        alpha = 0.9  # alpha is the factor used to decrement the temperature (exponential cooling)
        t_k = t_init
        t_threshold = 0.00001
        max_step = 1000  # number of trials per temperature step
        step_count = 0

        # List of current k states
        states = []
        next_state = self.generate_init_state()
        if(next_state.heur == 0): return next_state
        states.append(next_state)
        best_state = next_state

        for _ in range(k-1):
            next_state = self.generate_init_state()
            if(next_state.heur == 0): return next_state
            if(next_state.heur < best_state.heur):
                best_state = next_state
            states.append(next_state)

        while t_k > t_threshold:            
            ray_states = ray.put(states)
            futures = [generateNeighbors.remote(ray_states, i, self.n) for i in range(k)] # [ [n1], [n2], ..., [nk] ]
            allNeighbors2d = ray.get(futures)

            allNeighbors = list(np.concatenate(allNeighbors2d).flat)
            probabilities = []
            for n in allNeighbors:
                if n.heur == 0: return n
                if(n.heur < best_state.heur):
                    best_state = n
                probabilities.append(math.exp(n.heur / t_k))

            p_sum = sum(probabilities)
            normalizer = 1/p_sum
            normalize_prob = []
            for p in probabilities:
                normalize_prob.append(p * normalizer)

            states = np.random.choice(allNeighbors, size=k, replace=False, p=normalize_prob)

            # Decrease temperature
            step_count += 1
            if step_count >= max_step:
                step_count = 0
                t_k = t_k * alpha

        return best_state

def main():
    ray.init()
    
    start = time.perf_counter()
    s = NQueens_ParallelProblemSolver(10, 20)
    solution = s.solveParallel()
    end = time.perf_counter()

    print("Elapsed time = " + str(end - start))
    print("Heuristics: " + str(solution.heur))

if __name__ == "__main__":
    main()
