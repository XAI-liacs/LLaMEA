import numpy as np
from scipy.optimize import minimize

class HSPA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        hmcr = 0.9  # Harmony memory considering rate
        par = 0.3   # Pitch adjustment rate
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize harmony memory
        harmony_memory = lb + (ub - lb) * np.random.rand(population_size, self.dim)
        fitness = np.array([func(ind) for ind in harmony_memory])
        eval_count = population_size

        while eval_count < self.budget:
            if eval_count >= self.budget:
                break

            # Generate new harmony
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < hmcr:
                    # Memory consideration
                    idx = np.random.randint(population_size)
                    new_harmony[i] = harmony_memory[idx, i]
                    # Pitch adjustment
                    if np.random.rand() < par:
                        new_harmony[i] += np.random.uniform(-1, 1) * (ub[i] - lb[i]) * 0.01
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    # Random consideration
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

            # Apply periodic adjustment
            new_harmony = self.apply_periodicity(new_harmony, lb, ub)

            # Calculate fitness
            f_new_harmony = func(new_harmony)
            eval_count += 1

            # Update harmony memory
            worst_idx = np.argmax(fitness)
            if f_new_harmony < fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = f_new_harmony
            
            # Local refinement of the best solution
            best_idx = np.argmin(fitness)
            if eval_count + self.dim <= self.budget:
                bounds = [(lb[j], ub[j]) for j in range(self.dim)]
                res = minimize(lambda x: func(np.clip(x, lb, ub)), harmony_memory[best_idx], method='L-BFGS-B', bounds=bounds)
                eval_count += res.nfev
                if res.fun < fitness[best_idx]:
                    harmony_memory[best_idx] = res.x
                    fitness[best_idx] = res.fun

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]

    def apply_periodicity(self, trial, lb, ub):
        # Encourage periodic patterns in layer thicknesses
        period = self.dim // 2
        for i in range(0, self.dim, period):
            period_mean = np.mean(trial[i:i+period])
            trial[i:i+period] = np.clip(period_mean, lb[i:i+period], ub[i:i+period])
        return trial

# Example usage:
# func = YourBlackBoxFunction()
# optimizer = HSPA(budget=1000, dim=10)
# best_solution = optimizer(func)