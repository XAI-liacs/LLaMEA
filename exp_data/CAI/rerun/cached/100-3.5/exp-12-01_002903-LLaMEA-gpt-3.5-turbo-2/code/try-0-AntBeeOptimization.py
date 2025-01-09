import numpy as np

class AntBeeOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def local_search(current_solution, delta=0.1):
            candidate_solutions = [current_solution + delta * np.random.randn(self.dim) for _ in range(10)]
            best_solution = min(candidate_solutions, key=evaluate_solution)
            return best_solution

        current_solution = 10 * np.random.rand(self.dim) - 5  # Initialize with random solution

        for _ in range(self.budget):
            new_solution = local_search(current_solution)
            if evaluate_solution(new_solution) < evaluate_solution(current_solution):
                current_solution = new_solution

        return current_solution