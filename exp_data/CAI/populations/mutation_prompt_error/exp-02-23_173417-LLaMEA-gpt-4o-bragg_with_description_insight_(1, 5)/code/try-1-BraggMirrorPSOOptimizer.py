import numpy as np
from scipy.optimize import minimize

class BraggMirrorPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.pop_size = 20
        self.inertia_weight = 0.5
        self.cognitive_param = 1.5
        self.social_param = 1.5

    def particle_swarm_optimization(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(p) for p in population])
        self.evaluations += self.pop_size

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * np.random.rand() * (personal_best_positions[i] - population[i]) +
                                 self.social_param * np.random.rand() * (global_best_position - population[i]))

                population[i] = np.clip(population[i] + velocities[i], lb, ub)
                score = func(population[i])
                self.evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]

                if score < personal_best_scores[global_best_idx]:
                    global_best_position = population[i]
                    
            # Periodic reinforcement
            self.reinforce_periodicity(population)

        return global_best_position, personal_best_scores[global_best_idx]

    def reinforce_periodicity(self, population):
        period = 2  # Assuming a basic period of 2 layers
        for ind in population:
            for i in range(0, self.dim, period):
                ind[i:i+period] = np.mean(ind[i:i+period])  # Average over each period block

    def local_search(self, func, x0, bounds):
        result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        # Initial global optimization using Particle Swarm Optimization
        best_solution, best_fitness = self.particle_swarm_optimization(func, bounds)

        # Refine the best solution using local search
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)

        return best_solution