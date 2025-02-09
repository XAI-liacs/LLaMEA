import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Differential Evolution parameters
        self.population_size = 30
        self.crossover_rate = 0.7
        self.differential_weight = 0.5

        # Quantum parameters
        self.entanglement_factor = 0.1

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
        best_position = population[best_index]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired entanglement
                E = np.random.uniform(0, self.entanglement_factor, self.dim)
                entangled_vector = (1 - E) * population[i] + E * best_position

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant_vector = a + self.differential_weight * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, entangled_vector)
                
                # Evaluate trial solution
                trial_score = func(trial_vector)
                num_evaluations += 1
                
                # Selection
                if trial_score < scores[i]:
                    population[i] = trial_vector
                    scores[i] = trial_score

                    # Update best solution
                    if trial_score < scores[best_index]:
                        best_index = i
                        best_position = population[best_index]

                if num_evaluations >= self.budget:
                    return best_position

        return best_position