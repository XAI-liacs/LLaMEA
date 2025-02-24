import numpy as np
from scipy.optimize import minimize

class SwarmPeriodicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def particle_swarm_optimization(self, func, bounds, swarm_size=20, inertia=0.5, cognitive=1.5, social=1.5):
        lb, ub = bounds.lb, bounds.ub
        positions = np.random.uniform(lb, ub, (swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (swarm_size, self.dim))
        personal_best_pos = np.copy(positions)
        personal_best_val = np.array([func(ind) for ind in positions])
        self.evaluations += swarm_size

        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(swarm_size):
                if self.evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    inertia * velocities[i] +
                    cognitive * r1 * (personal_best_pos[i] - positions[i]) +
                    social * r2 * (global_best_pos - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                current_val = func(positions[i])
                self.evaluations += 1
                if current_val < personal_best_val[i]:
                    personal_best_val[i] = current_val
                    personal_best_pos[i] = positions[i]

            global_best_idx = np.argmin(personal_best_val)
            global_best_pos = personal_best_pos[global_best_idx]

            # Dynamic inertia adjustment based on evaluations
            inertia = max(0.3, inertia * 0.99 - (0.05 * (self.evaluations / self.budget)))
            cognitive = min(2.0, cognitive * 1.01)
            social = min(2.0, social * 1.01)

            # Enhance periodicity using symmetry detection
            periodic_positions = self.enforce_symmetry(positions)
            for i in range(swarm_size):
                if self.evaluations >= self.budget:
                    break
                periodic_val = func(periodic_positions[i])
                self.evaluations += 1
                if periodic_val < personal_best_val[i]:
                    personal_best_val[i] = periodic_val
                    personal_best_pos[i] = periodic_positions[i]

        best_idx = np.argmin(personal_best_val)
        return personal_best_pos[best_idx], personal_best_val[best_idx]

    def enforce_symmetry(self, positions):
        symmetric_positions = np.copy(positions)
        for ind in symmetric_positions:
            pattern = self.detect_symmetric_pattern(ind)
            if pattern is not None:
                ind[:] = np.tile(pattern, len(ind) // len(pattern))[:len(ind)]
        return symmetric_positions

    def detect_symmetric_pattern(self, sequence):
        length = len(sequence)
        for period in range(1, length // 2 + 1):
            if all(sequence[i] == sequence[i + period] for i in range(length - period)):
                return sequence[:period]
        return None

    def local_search(self, func, x0, bounds):
        std_bounds = [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]
        result = minimize(func, x0, bounds=std_bounds, method='Nelder-Mead')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.particle_swarm_optimization(func, bounds)
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)
        return best_solution