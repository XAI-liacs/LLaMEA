import numpy as np

class QADE:
    def __init__(self, budget, dim, population_size=20, F_min=0.4, F_max=0.9, CR=0.9, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F_min = F_min
        self.F_max = F_max
        self.CR = CR
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_value = float('inf')
        best_position = None
        
        while self.evaluations < self.budget:
            new_population = []
            F = self.adaptive_scaling_factor()
            
            for i in range(self.population_size):
                x = population[i]
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = a + F * (b - c)
                trial = self.crossover(x, mutant, lb, ub)

                if np.random.rand() < self.quantum_prob:
                    trial = self.quantum_perturbation(trial, lb, ub)

                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < func(x):
                    new_population.append(trial)
                    if trial_value < best_value:
                        best_value = trial_value
                        best_position = trial
                else:
                    new_population.append(x)

                if self.evaluations >= self.budget:
                    break

            population = new_population

        return best_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def adaptive_scaling_factor(self):
        return self.F_min + (self.F_max - self.F_min) * (1 - (self.evaluations / self.budget))

    def crossover(self, target, mutant, lb, ub):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, lb, ub)

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)