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

# Ray actor
@ray.remote
class Process:
    def __init__(self, i, n):
        self.i = i # process index
        self.n = n # size of board
    
    def generateNeighbors(self, states: list[BoardState]):
        n = self.n
        s = states[self.i]
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

    def listProbs(self, states: list[BoardState], t, elements_per_process):
        p = []
        for x in range(elements_per_process):
            s = states[self.i*elements_per_process+x]
            p.append(math.exp(s.heur / t))
        return p

    def normalizeProbs(self, probs, normalizer, elements_per_process):
        normal_p = []
        for x in range(elements_per_process):
            p = probs[self.i*elements_per_process+x]
            normal_p.append(p * normalizer)
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
        actors = []
        for i in range(k):
            states.append(self.generate_init_state())
            actors.append(Process.remote(i))
            
        best_state = states[0]

        while t_k > t_threshold and best_state.heur != 0:            
            ray_states = ray.put(states)

            allNeighbors = []
            for i in range(k):
                ray_ithNeighbors = actors[i].generateNeighbors.remote(ray_states)
                allNeighbors += ray.get(ray_ithNeighbors)
            ray_neighbors = ray.put(allNeighbors)

            elements_per_process = len(allNeighbors)/k

            probs = []
            for i in range(k):
                ray_ithProbs = actors[i].listProbs.remote(ray_neighbors, t_k, elements_per_process)
                probs += ray.get(ray_ithProbs)

            ray_probs = ray.put(probs)

            p_sum = sum(probs)
            normalizer = 1/p_sum
            normalProbs = []
            for i in range(k):
                ray_ithNormalProbs = actors[i].normalizeProbs.remote(ray_probs, normalizer, elements_per_process)
                normalProbs += ray.get(ray_ithNormalProbs)
        
            states = np.random.choice(allNeighbors, size=k, replace=False, p=normalProbs)

            for s in states:
                if s.heur < best_state.heur: 
                    best_state = s

            # Decrease temperature
            step_count += 1
            if step_count >= max_step:
                step_count = 0
                t_k = t_k * alpha

        return best_state

def main():
    ray.init()
    
    start = time.perf_counter()
    s = NQueens_ParallelProblemSolver(8, 100)
    solution = s.solveParallel()
    end = time.perf_counter()

    print("Elapsed time = " + str(end - start))
    print("Heuristics: " + str(solution.heur))

if __name__ == "__main__":
    main()
