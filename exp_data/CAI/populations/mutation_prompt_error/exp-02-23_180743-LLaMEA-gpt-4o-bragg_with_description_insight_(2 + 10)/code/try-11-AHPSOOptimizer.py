import numpy as np
from scipy.optimize import minimize

class AHPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.7   # inertia weight
        self.evaluations = 0
        self.global_best = None

    def periodic_boundary_correction(self, particle, bounds):
        lb, ub = bounds.lb, bounds.ub
        periodic_particle = np.clip(particle, lb, ub)
        return periodic_particle

    def adaptively_mutate_particle(self, particle, best, bounds):
        lb, ub = bounds.lb, bounds.ub
        mutation_vector = np.random.normal(0, 0.1, size=self.dim) * (ub - lb)
        mutated_particle = np.clip(particle + mutation_vector, lb, ub)
        return mutated_particle if np.random.rand() < 0.2 else best

    def particle_swarm_optimization(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        velocity = np.random.rand(self.population_size, self.dim) * (ub - lb)
        swarm = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        self.evaluations += self.population_size

        global_best_index = np.argmin(personal_best_scores)
        global_best_score = personal_best_scores[global_best_index]
        self.global_best = personal_best_positions[global_best_index]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                               self.c2 * r2 * (self.global_best - swarm[i]))

                # Update position
                swarm[i] = self.periodic_boundary_correction(swarm[i] + velocity[i], bounds)
                
                # Adaptive mutation
                swarm[i] = self.adaptively_mutate_particle(swarm[i], self.global_best, bounds)

                # Evaluate new position
                score = func(swarm[i])
                self.evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    self.global_best = swarm[i]

        return self.global_best

    def local_optimization(self, solution, func, bounds):
        result = minimize(func, solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        return result.x if result.success else solution

    def __call__(self, func):
        best_solution = self.particle_swarm_optimization(func, func.bounds)
        best_solution = self.local_optimization(best_solution, func, func.bounds)
        return best_solution