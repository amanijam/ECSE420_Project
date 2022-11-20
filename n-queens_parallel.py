import random, math, time
import numpy as np
import multiprocessing as mp

# Author:       Amani Jammoul
# McGill ID:    260381641

# Data structure that holds information about any board state
# positions:    1D size n int array, holding the row of the queen in each of the n columns
# foward_diags: 1D size n int array, holding value of row+col
# back_diags:   1D size n int array, holding value of row-col
class BoardState:
    def __init__(self, positions, forward_diags, back_diags):
        self.positions = positions # 
        self.forward_diags = forward_diags
        self.back_diags = back_diags

# Class that defines an N Queens Problem, with any value N
class NQueens_Problem:
    def __init__(self, N) -> None:
        self.N = N
        self.init_state = self.generate_init_state() # automatically generate inital board state when a problem is initialized
        self.curr_state = self.init_state

    # Randomly generates a state (N x N board configuration)
    def generate_init_state(self) -> BoardState:
        n = self.N
        positions = []
        forward_diags = []
        back_diags = []

        for i in range(n):
            positions.append(random.randint(0, n - 1))
            forward_diags.append(positions[i] + i)
            back_diags.append(positions[i] - i)

        return BoardState(positions, forward_diags, back_diags)

    # Generates a random neighbor of given state
    def random_neighbor(self, state: BoardState):
        # Choose random column
        random_col = random.randint(0, self.N - 1)

        # Choose random row, excluding the row where the queen in column random_col is placed
        random_row = random.choice(
            [i for i in range(self.N) if i != state.positions[random_col]]
        )

        # Make a copy of the state's positions but changing the position of one random queen
        random_pos = state.positions[:]
        random_pos[random_col] = random_row

        forward_diags = state.forward_diags[:]
        forward_diags[random_col] = random_row + random_col

        back_diags = state.back_diags[:]
        back_diags[random_col] = random_row - random_col

        random_neighbor = BoardState(random_pos, forward_diags, back_diags)
        return random_neighbor


# Class that is used to solve a N Queens Problem
class NQueens_ProblemSolver:
    def __init__(self):
        pass

    # Counts the number of duplicate values in the given dict
    def num_duplicates(self, dict):
        duplicate_count = 0
        for i in dict:
            if dict[i] > 1:
                duplicate_count += dict[i] - 1
        return duplicate_count

    # Evaluates a state by calculating the number of constraint violations
    # Two queens in the same row, forward, or backward diagonal = constraint violation
    def evaluate_state(self, state: BoardState):
        pos_count = {i: state.positions.count(i) for i in state.positions}
        pos_num_duplicates = self.num_duplicates(pos_count)

        forward_diag_count = {
            i: state.forward_diags.count(i) for i in state.forward_diags
        }
        forward_diag_num_duplicates = self.num_duplicates(forward_diag_count)

        back_diag_count = {i: state.back_diags.count(i) for i in state.back_diags}
        back_diag_num_dulpicates = self.num_duplicates(back_diag_count)

        return (
            pos_num_duplicates + forward_diag_num_duplicates + back_diag_num_dulpicates
        )

    # Solves the problem using hill climbing
    def hill_climbing(self, problem: NQueens_Problem):
        n = problem.N
        terminate = False
        best_next_state = None
        best_next_state_heur = n**n + 1

        while not terminate:
            # generate all neighbor states
            # pick the one that improves current heur_val and has min heur_val

            found_better_neighbor = False
            curr_state_heur = self.evaluate_state(problem.curr_state)

            for col in range(n):
                curr_row = problem.curr_state.positions[col]
                for row in range(n):
                    if row != curr_row:
                        next_pos = problem.curr_state.positions[:]
                        next_pos[col] = row

                        next_forward_diags = problem.curr_state.forward_diags[:]
                        next_forward_diags[col] = row + col

                        next_back_diags = problem.curr_state.back_diags[:]
                        next_back_diags[col] = row - col

                        next_state = BoardState(
                            next_pos, next_forward_diags, next_back_diags
                        )
                        next_state_heur = self.evaluate_state(next_state)
                        if (
                            next_state_heur < best_next_state_heur
                            and next_state_heur < curr_state_heur
                        ):
                            found_better_neighbor = True
                            best_next_state = next_state
                            best_next_state_heur = next_state_heur

            # If no neighbor can provide any imporvement, that means we are at 
            #   a local optima and can terminate
            if found_better_neighbor == False:
                terminate = True
            else:
                problem.curr_state = best_next_state

        return problem.curr_state.positions

    # Solves the problem using simulated annealing
    def simulated_annealing(self, problem: NQueens_Problem):
        best_state = problem.init_state
        best_heur = self.evaluate_state(best_state)
        curr_heur = best_heur

        # Define temperature and cooling values
        # Ensure initial temp is large enough to generate a high initial probability (abbout 0.999)
        t_init = 1000 * (problem.N // 2)
        alpha = 0.9  # alpha is the factor used to decrement the temperature (exponential cooling)
        t_k = t_init
        t_threshold = 0.00001
        max_step = 1000  # number of trials per temperature step
        step_count = 0

        count = 0

        # Keep looping until temperature is below the threshold, or a global optimum is reached (heuristic value = 0)
        while t_k > t_threshold and best_heur != 0:
            count += 1

            # Pick a random neighbor and evalute its heuristic value
            next_state = problem.random_neighbor(problem.curr_state)
            next_state_heur = self.evaluate_state(next_state)

            if next_state_heur < best_heur:
                best_state = next_state
                best_heur = next_state_heur

            elif next_state_heur < curr_heur:
                problem.curr_state = next_state
                curr_heur = next_state_heur

            else:
                # If neighbr does not improve what we currently have,
                #   still chose to make it our nre current state
                #   with a certain probability, p = e^(heuristic diff/temp)
                p = math.exp((curr_heur - next_state_heur) / t_k)
                select = np.random.choice([0, 1], 1, p=[1 - p, p])[0]
                if select == 1:
                    problem.curr_state = next_state
                    curr_heur = next_state_heur

            # Decrease temperature
            step_count += 1
            if step_count >= max_step:
                step_count = 0
                t_k = t_k * alpha

        return best_state.positions

    def solve(self, problem, local_search_alg="SA"):
        if local_search_alg == "hill climbing":
            return self.hill_climbing(problem)
        elif local_search_alg == "SA":
            return self.simulated_annealing(problem)
        else:
            return None

    def solveParallel(self, n):
        # List of k states and k heuristics

        # Repeat until a state with heuristic 0 is found or temp < threshold
        # Generate k random init states
        # Generate all neighbor states with their heuristic value
        # Select k individuals at random, with probability proportional to e^(-h/T)
        
        pass


def main():
    print("Number of processors: ", mp.cpu_count())
    # problem = NQueens_Problem(8)

    # start = time.perf_counter()
    # solution = NQueens_ProblemSolver().solve(problem, local_search_alg="SA")
    # end = time.perf_counter()

    # print("Solution: " + str(solution))
    # print("Elapsed time = " + str(end - start))


if __name__ == "__main__":
    main()
