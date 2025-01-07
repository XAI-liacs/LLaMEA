import numpy as np

class AEFO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.population = []

    def initialize_population(self, lb, ub):
        return [{'position': lb + (ub - lb) * np.random.rand(self.dim),
                 'charge': np.random.rand()} for _ in range(self.population_size)]

    def update_particle(self, particle, best_position, lb, ub, inertia_factor):
        for i in range(self.dim):
            force = (best_position[i] - particle['position'][i]) * particle['charge']
            direction = np.sign(force)
            step_size = inertia_factor * force * direction
            particle['position'][i] += step_size
            particle['position'][i] = np.clip(particle['position'][i], lb[i], ub[i])

    def adaptive_mutation(self, particle, lb, ub, eval_ratio):
        if np.random.rand() < 0.5 * (1 - eval_ratio):
            mutation_strength = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.05
            particle['position'] += mutation_strength
            particle['position'] = np.clip(particle['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        best_position = None
        best_value = float('inf')

        while evaluations < self.budget:
            for particle in self.population:
                value = func(particle['position'])
                evaluations += 1
                
                if value < best_value:
                    best_value = value
                    best_position = particle['position'].copy()

                if evaluations >= self.budget:
                    break

            inertia_factor = 0.5 + 0.5 * (1 - evaluations / self.budget)
            for particle in self.population:
                self.update_particle(particle, best_position, lb, ub, inertia_factor)
                self.adaptive_mutation(particle, lb, ub, evaluations / self.budget)

        return best_position, best_value