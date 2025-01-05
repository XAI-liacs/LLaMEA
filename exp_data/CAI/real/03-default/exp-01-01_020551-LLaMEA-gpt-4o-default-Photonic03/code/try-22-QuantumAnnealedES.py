import numpy as np

class QuantumAnnealedES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.t0 = 100   # Initial temperature for annealing
        self.alpha = 0.95  # Cooling rate
        self.beta = 0.05  # Quantum-inspired learning rate
        self.sigma = 0.1  # Mutation step size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_index = np.argmin(scores)
        best_position = population[best_index].copy()
        best_score = scores[best_index]
        temperature = self.t0
        evaluations = self.population_size

        while evaluations < self.budget:
            # Annealing-inspired random exploration
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                new_candidate = population[i] + np.random.normal(0, self.sigma * (ub - lb), self.dim)
                new_candidate = np.clip(new_candidate, lb, ub)
                
                # Quantum-inspired update
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    new_candidate += q * (best_position - new_candidate)
                
                new_score = func(new_candidate)
                evaluations += 1

                # Accept new candidate based on simulated annealing criteria
                if new_score < scores[i] or np.random.rand() < np.exp((scores[i] - new_score) / temperature):
                    population[i] = new_candidate
                    scores[i] = new_score

                    # Update the best solution found
                    if new_score < best_score:
                        best_position = new_candidate
                        best_score = new_score

            # Cooling schedule
            temperature *= self.alpha

        # Return the best solution found
        return best_position, best_score