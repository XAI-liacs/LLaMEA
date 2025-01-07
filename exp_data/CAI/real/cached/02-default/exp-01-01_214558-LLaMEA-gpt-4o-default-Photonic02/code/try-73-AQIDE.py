import numpy as np

class AQIDE:
    def __init__(self, budget, dim, population_size=20, F=0.5, CR=0.9, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_global_value = float('inf')
        best_global_position = None
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                # Quantum-inspired perturbation
                if np.random.rand() < self.quantum_prob:
                    trial = self.quantum_perturbation(trial, lb, ub)

                # Selection
                f_trial = func(trial)
                self.evaluations += 1
                f_target = func(population[i])
                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < best_global_value:
                        best_global_value = f_trial
                        best_global_position = trial

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        perturbation = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        q_position = position + perturbation
        return np.clip(q_position, lb, ub)