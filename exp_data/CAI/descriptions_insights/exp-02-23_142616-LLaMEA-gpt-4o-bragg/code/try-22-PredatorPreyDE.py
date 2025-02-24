import numpy as np

class PredatorPreyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.f = 0.5  # mutation factor
        self.cr = 0.9  # crossover probability

    def predator_prey_dynamics(self, population, prey_best):
        predator = np.random.uniform(np.min(population, axis=0), np.max(population, axis=0), self.dim)
        for i in range(self.population_size):
            prey = population[i]
            if np.random.rand() < 0.5:
                prey = prey + self.f * (prey_best - predator)
            else:
                prey = prey - self.f * (predator - prey_best)
            population[i] = np.clip(prey, lb, ub)
        return population

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
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = x1 + self.f * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.cr
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

                if evaluations >= self.budget:
                    break

            population = self.predator_prey_dynamics(population, global_best_position)

        return global_best_position, global_best_score