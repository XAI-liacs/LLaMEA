import numpy as np

class HybridDESimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # DE parameters
        self.population_size = 50
        self.crossover_rate = 0.9
        self.differential_weight = 0.8
        
        # Simulated Annealing parameters
        self.initial_temperature = 100.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        num_evaluations = 0
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        num_evaluations += self.population_size

        best_index = np.argmin(scores)
        best_solution = population[best_index]
        best_score = scores[best_index]

        temperature = self.initial_temperature

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.differential_weight * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial and simulated annealing acceptance
                trial_score = func(trial)
                num_evaluations += 1
                
                if trial_score < scores[i] or np.exp((scores[i] - trial_score) / temperature) > np.random.rand():
                    population[i] = trial
                    scores[i] = trial_score

                    # Update best solution
                    if trial_score < best_score:
                        best_score = trial_score
                        best_solution = trial

                if num_evaluations >= self.budget:
                    return best_solution

            # Cool down temperature
            temperature *= self.cooling_rate

        return best_solution