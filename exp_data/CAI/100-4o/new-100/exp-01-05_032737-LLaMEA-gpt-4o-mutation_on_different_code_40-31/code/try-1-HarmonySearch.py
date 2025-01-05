import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hm_size=20, hmcr=0.9, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hm_size = hm_size
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize Harmony Memory (HM) with random solutions
        HM = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.hm_size)]
        HM_fitness = [func(hm) for hm in HM]

        # Update best solution
        self.f_opt = min(HM_fitness)
        self.x_opt = HM[np.argmin(HM_fitness)]

        evals = self.hm_size  # Initial evaluations

        while evals < self.budget:
            # Dynamic adjustment of parameters
            dynamic_hmcr = self.hmcr - (self.hmcr * evals / self.budget)
            dynamic_par = self.par + (0.2 * evals / self.budget)
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < dynamic_hmcr:  # Memory consideration
                    new_harmony[i] = np.random.choice([harmony[i] for harmony in HM])
                    if np.random.rand() < dynamic_par:  # Pitch adjustment
                        new_harmony[i] += self.bw * (np.random.rand() - 0.5) * 2
                        new_harmony[i] = np.clip(new_harmony[i], func.bounds.lb[i], func.bounds.ub[i])
                else:
                    new_harmony[i] = np.random.uniform(func.bounds.lb[i], func.bounds.ub[i])  # Randomization

            # Evaluate new harmony
            new_harmony_fitness = func(new_harmony)
            evals += 1

            # Elitism: Keep the best solution intact
            if new_harmony_fitness < max(HM_fitness):
                worst_idx = np.argmax(HM_fitness)
                if new_harmony_fitness < self.f_opt:
                    HM[worst_idx] = new_harmony
                    HM_fitness[worst_idx] = new_harmony_fitness
                    self.f_opt = new_harmony_fitness
                    self.x_opt = new_harmony
                else:
                    if worst_idx != np.argmin(HM_fitness):  # Preserve the best
                        HM[worst_idx] = new_harmony
                        HM_fitness[worst_idx] = new_harmony_fitness

        return self.f_opt, self.x_opt