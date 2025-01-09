import numpy as np

class DESAOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20  # Population size for differential evolution
        self.mutation_factor = 0.8  # Mutation factor for differential evolution
        self.crossover_rate = 0.9  # Crossover rate for differential evolution
        self.initial_temperature = 10.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for simulated annealing

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_index = np.argmin(scores)
        best_solution = population[best_index]
        best_score = scores[best_index]
        evaluations = self.population_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), 
                                 self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial solution
                trial_score = func(trial)
                evaluations += 1
                
                # Selection and simulated annealing-based acceptance
                if trial_score < scores[i] or np.random.rand() < np.exp((scores[i] - trial_score) / temperature):
                    population[i] = trial
                    scores[i] = trial_score
                    if trial_score < best_score:
                        best_solution = trial
                        best_score = trial_score

            # Cooling the temperature
            temperature *= self.cooling_rate

        return best_solution, best_score