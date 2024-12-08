import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size
        self.mutation_prob = 0.5  # Initial mutation probability
        self.population_size = 4  # Initial population size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                future_results = [executor.submit(self.mutate_and_evaluate, func, best_solution) for _ in range(self.population_size)]
                results = [future.result() for future in future_results]
                
                for candidate_solution, candidate_fitness in results:
                    if candidate_fitness < best_fitness:
                        best_solution = candidate_solution
                        best_fitness = candidate_fitness
                
                # Adaptive mutation probability adjustment
                self.mutation_prob = max(0.1, min(self.mutation_prob + 0.05 * np.random.uniform(-1, 1), 0.9))
                self.mutation_step *= np.exp(0.1 * np.random.uniform(-1, 1))
                self.mutation_step = max(0.1, min(self.mutation_step, 2.0))
                
                # Dynamic population size adjustment
                self.population_size = max(2, min(self.population_size + np.random.choice([-1, 1]), 8))

        return best_solution

    def mutate_and_evaluate(self, func, solution):
        mutation = self.mutation_step * np.random.uniform(-1, 1, self.dim)
        candidate_solution = np.clip(solution + mutation, -5.0, 5.0)
        candidate_fitness = func(candidate_solution)
        return candidate_solution, candidate_fitness