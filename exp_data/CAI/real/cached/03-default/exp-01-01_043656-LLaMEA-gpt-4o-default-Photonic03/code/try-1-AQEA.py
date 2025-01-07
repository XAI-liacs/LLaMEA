import numpy as np

class AQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.alpha = 0.1  # mutation factor
        self.quantum_prob = 0.5  # initial quantum probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            # Quantum-inspired superposition
            q_population = np.random.choice([0, 1], (self.population_size, self.dim), p=[1-self.quantum_prob, self.quantum_prob])
            q_population = np.where(q_population == 1, pop, pop[::-1])
            
            # Mutation and selection
            for i in range(self.population_size):
                mutation_vector = np.random.uniform(-1, 1, self.dim) * self.alpha
                trial = q_population[i] + mutation_vector
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                
                trial_value = func(trial)
                eval_count += 1
                if trial_value < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_value
                
                # Adaptive mutation based on success
                if eval_count % 10 == 0:
                    self.alpha = min(0.5, self.alpha * 1.1) if trial_value < fitness[i] else max(0.05, self.alpha * 0.9)

                if eval_count >= self.budget:
                    break

            # Update quantum probability adaptively
            self.quantum_prob = min(0.9, self.quantum_prob + 0.01) if np.mean(fitness) < np.median(fitness) else max(0.1, self.quantum_prob - 0.01)
        
        best_index = np.argmin(fitness)
        return pop[best_index]