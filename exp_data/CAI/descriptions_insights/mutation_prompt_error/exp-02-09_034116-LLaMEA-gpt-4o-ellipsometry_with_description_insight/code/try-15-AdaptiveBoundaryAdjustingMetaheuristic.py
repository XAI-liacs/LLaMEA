import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveBoundaryAdjustingMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Exploration phase: Sobol sequence for initial sampling
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        num_samples = max(1, self.budget // 10)  # 10% of budget for sampling
        samples = sobol_sampler.random_base2(m=int(np.log2(num_samples)))
        
        for sample in samples:
            sample_scaled = lb + sample * (ub - lb)
            value = func(sample_scaled)
            if value < best_value:
                best_value = value
                best_solution = sample_scaled

        # Dynamic boundary adjustment
        adaptive_lb = np.copy(lb)
        adaptive_ub = np.copy(ub)
        
        def adjust_bounds(solution):
            global best_solution, best_value
            for i in range(self.dim):
                adaptive_lb[i] = max(adaptive_lb[i], solution[i] - 0.2 * (ub[i] - lb[i]))
                adaptive_ub[i] = min(adaptive_ub[i], solution[i] + 0.2 * (ub[i] - lb[i]))

        # Exploitation phase: Particle Swarm Optimization
        num_particles = max(2, self.budget // 20)  # 5% of budget per evaluation
        particle_positions = lb + np.random.rand(num_particles, self.dim) * (ub - lb)
        particle_velocities = np.zeros((num_particles, self.dim))
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(pos) for pos in particle_positions])
        
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()
        
        iterations = (self.budget - num_samples) // num_particles

        for _ in range(iterations):
            for i in range(num_particles):
                # Update velocity
                particle_velocities[i] = 0.5 * particle_velocities[i] + \
                                         0.5 * np.random.rand(self.dim) * (personal_best_positions[i] - particle_positions[i]) + \
                                         0.3 * np.random.rand(self.dim) * (global_best_position - particle_positions[i])
                # Update position
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], adaptive_lb, adaptive_ub)
                
                # Evaluate new position
                value = func(particle_positions[i])
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particle_positions[i]
                    if value < global_best_value:
                        global_best_value = value
                        global_best_position = particle_positions[i]
                        adjust_bounds(global_best_position)
        
        best_solution = global_best_position
        return best_solution