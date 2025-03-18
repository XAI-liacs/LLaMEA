import numpy as np
from scipy.optimize import minimize

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def harmony_search(self, func, bounds, harmony_memory_size=30, harmony_regeneration_rate=0.3, pitch_adjustment_rate=0.4):
        lb, ub = bounds.lb, bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_fitness = np.array([func(harmony) for harmony in harmony_memory])
        self.evaluations += harmony_memory_size

        while self.evaluations < self.budget:
            if np.random.rand() < harmony_regeneration_rate:
                new_harmony = np.random.uniform(lb, ub, self.dim)
            else:
                new_harmony = np.copy(harmony_memory[np.random.choice(harmony_memory_size)])
                if np.random.rand() < pitch_adjustment_rate:
                    new_harmony += np.random.uniform(-1, 1, self.dim) * 0.1 * (ub - lb)

            new_harmony = self.enforce_periodicity(new_harmony)
            new_fitness = func(new_harmony)
            self.evaluations += 1

            worst_idx = np.argmax(harmony_fitness)
            if new_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness
            
            # Adaptive pitch adjustment rate
            pitch_adjustment_rate = max(0.1, pitch_adjustment_rate * 0.99)

        best_idx = np.argmin(harmony_fitness)
        return harmony_memory[best_idx], harmony_fitness[best_idx]

    def enforce_periodicity(self, harmony):
        pattern = self.detect_periodic_pattern(harmony)
        if pattern is not None:
            harmony[:] = np.tile(pattern, len(harmony) // len(pattern) + 1)[:len(harmony)]
        return harmony

    def detect_periodic_pattern(self, sequence):
        length = len(sequence)
        autocorrelation = np.correlate(sequence, sequence, mode='full')
        autocorrelation = autocorrelation[length-1:]
        peaks = np.where((autocorrelation[1:] < autocorrelation[:-1]) &
                         (autocorrelation[:-1] > np.mean(autocorrelation) * 1.05))[0]
        if peaks.size > 0:
            period = peaks[0] + 1
            return sequence[:period]
        return None

    def local_search(self, func, x0, bounds):
        std_bounds = [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]
        result = minimize(func, x0, bounds=std_bounds, method='BFGS')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.harmony_search(func, bounds)
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)
        return best_solution