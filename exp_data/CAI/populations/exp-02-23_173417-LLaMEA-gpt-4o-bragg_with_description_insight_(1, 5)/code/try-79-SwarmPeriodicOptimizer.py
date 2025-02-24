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

            # Adaptively adjust swarm parameters
            inertia = max(0.3, inertia * 0.985)
            adaptive_rate = 1.01 + 0.02 * np.cos(self.evaluations / 100)  # Use cosine modulation
            cognitive = min(2.0, cognitive * adaptive_rate)
            social = min(2.0, social * adaptive_rate)

            # Enhance periodicity using custom pattern recognition
            periodic_positions = self.enforce_periodicity(positions)
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

    def enforce_periodicity(self, positions):
        periodic_positions = np.copy(positions)
        for ind in periodic_positions:
            pattern = self.detect_periodic_pattern(ind)
            if pattern is not None:
                ind[:] = np.tile(pattern, len(ind) // len(pattern) + 1)[:len(ind)]
        return periodic_positions

    def detect_periodic_pattern(self, sequence):
        length = len(sequence)
        autocorrelation = np.correlate(sequence, sequence, mode='full')
        autocorrelation = autocorrelation[length-1:]
        peaks = np.where((autocorrelation[1:] < autocorrelation[:-1]) & 
                         (autocorrelation[:-1] > np.mean(autocorrelation) * 1.05))[0]  # Fine-tuned threshold
        if peaks.size > 0:
            period = peaks[0] + 1
            return sequence[:period]
        return None

    def local_search(self, func, x0, bounds):
        std_bounds = [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]
        result = minimize(func, x0, bounds=std_bounds, method='L-BFGS-B', options={'maxiter': 200})
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.particle_swarm_optimization(func, bounds)
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)
        return best_solution