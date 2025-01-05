import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return population

    def quantum_update(self, particle, global_best, lb, ub, alpha):
        mean_best = (particle['best_position'] + global_best) / 2
        phi = 2 * np.pi * np.random.rand(self.dim)  # Random angle
        r = np.random.rand(self.dim)  # Random radius
        particle['position'] = mean_best + alpha * r * np.sin(phi)
        particle['position'] = np.clip(particle['position'], lb, ub)

    def differential_mutation(self, target_idx, lb, ub):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        F = 0.8  # Differential weight
        mutant = self.population[a]['position'] + F * (self.population[b]['position'] - self.population[c]['position'])
        mutant = np.clip(mutant, lb, ub)
        return mutant

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        global_best = None
        global_best_value = float('inf')
        
        while evaluations < self.budget:
            for idx, particle in enumerate(self.population):
                trial_position = self.differential_mutation(idx, lb, ub)
                trial_value = func(trial_position)
                evaluations += 1
                
                if trial_value < particle['best_value']:
                    particle['best_value'] = trial_value
                    particle['best_position'] = trial_position.copy()
                
                if trial_value < global_best_value:
                    global_best_value = trial_value
                    global_best = trial_position.copy()

                if evaluations >= self.budget:
                    break

            alpha = 1.0 - evaluations / self.budget
            for particle in self.population:
                self.quantum_update(particle, global_best, lb, ub, alpha)

        return global_best, global_best_value