import numpy as np

class Q_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.population = []
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def initialize_population(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.population_size)]

    def quantum_mutation(self, agent, lb, ub):
        alpha = np.random.uniform(0, np.pi)
        beta = np.random.uniform(0, 2 * np.pi)
        q_bit = np.array([np.cos(alpha), np.sin(alpha) * np.exp(1j * beta)])
        q_positions = np.angle(q_bit[1])  # Extract phase information as position
        return lb + (ub - lb) * ((q_positions % (2 * np.pi)) / (2 * np.pi))

    def mutate(self, target_idx, lb, ub):
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        quantum_mutant = self.quantum_mutation(mutant, lb, ub)
        return np.clip(mutant + quantum_mutant, lb, ub)

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.CR:
                trial[i] = mutant[i]
        return trial

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i, lb, ub)
                trial = self.crossover(target, mutant)
                
                trial_value = func(trial)
                evaluations += 1
                
                if trial_value < func(target):
                    self.population[i] = trial
                    target_value = trial_value
                
                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial.copy()

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value