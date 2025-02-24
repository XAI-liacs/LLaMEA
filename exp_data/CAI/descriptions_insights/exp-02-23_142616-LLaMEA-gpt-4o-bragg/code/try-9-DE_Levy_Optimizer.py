import numpy as np

class DE_Levy_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover probability
        self.alpha = 0.5  # control parameter for Levy flight

    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return self.alpha * step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        
        evaluations = self.population_size
        best_idx = scores.argmin()
        global_best_position = population[best_idx].copy()
        global_best_score = scores[best_idx]
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
                
                # Modified mutation strategy
                F_dynamic = self.F * (1 + 0.1 * np.std(scores) / np.mean(scores))
                mutant = np.clip(a + F_dynamic * (b - c), lb, ub)

                # Dynamically adjust CR based on population diversity
                CR_dynamic = self.CR * (1 + 0.1 * np.std(scores) / np.mean(scores))
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_position = trial.copy()
                        global_best_score = trial_score
                
                # Perform Levy Flight
                if evaluations < self.budget and np.random.rand() < 0.1:
                    levy_step = self.levy_flight(self.dim)
                    levied_position = np.clip(global_best_position + levy_step, lb, ub)
                    levied_score = func(levied_position)
                    evaluations += 1
                    if levied_score < global_best_score:
                        global_best_position = levied_position.copy()
                        global_best_score = levied_score

                if evaluations >= self.budget:
                    break
        
        return global_best_position, global_best_score