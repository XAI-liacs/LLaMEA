import numpy as np

class QGDE:
    def __init__(self, budget, dim, population_size=20, F=0.5, CR=0.7, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.initialize_population(lb, ub)
        best_individual = None
        best_value = float('inf')

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = self.select_three_others(i)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                trial = self.crossover(pop[i], mutant)

                if np.random.rand() < self.quantum_prob:
                    trial = self.quantum_perturbation(trial, lb, ub)
                
                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < self.gradient_guided_update(trial, trial_value, func):
                    pop[i] = trial
                    if trial_value < best_value:
                        best_value = trial_value
                        best_individual = trial

                if self.evaluations >= self.budget:
                    break
        
        return best_individual

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def select_three_others(self, current_index):
        indices = np.array([i for i in range(self.population_size) if i != current_index])
        return np.random.choice(indices, 3, replace=False)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def gradient_guided_update(self, position, value, func):
        grad = self.estimate_gradient(position, func)
        new_position = position - 0.01 * grad
        new_value = func(new_position)
        self.evaluations += 1
        return new_value if new_value < value else value

    def estimate_gradient(self, position, func, epsilon=1e-4):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            delta = np.zeros(self.dim)
            delta[i] = epsilon
            grad[i] = (func(position + delta) - func(position - delta)) / (2 * epsilon)
            self.evaluations += 2  # Two evaluations for each dimension
        return grad