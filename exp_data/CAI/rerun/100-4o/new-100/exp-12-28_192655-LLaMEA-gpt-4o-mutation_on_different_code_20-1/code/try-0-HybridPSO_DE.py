import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 40
        self.w = 0.7  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.f = 0.5  # differential weight
        self.cr = 0.9  # crossover probability

    def __call__(self, func):
        np.random.seed(42)
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(x) for x in population])
        global_best_score = np.min(personal_best_scores)
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # PSO update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Evaluate population
            scores = np.array([func(x) for x in population])
            evaluations += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i]
                    if scores[i] < global_best_score:
                        global_best_score = scores[i]
                        global_best_position = population[i]

            # DE operation
            if evaluations < self.budget:
                for i in range(self.population_size):
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), lb, ub)
                    trial = np.copy(population[i])
                    
                    for j in range(self.dim):
                        if np.random.rand() < self.cr:
                            trial[j] = mutant[j]
                            
                    trial_score = func(trial)
                    evaluations += 1
                    
                    if trial_score < scores[i]:
                        scores[i] = trial_score
                        population[i] = trial

                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial

            if evaluations >= self.budget:
                break

        self.f_opt, self.x_opt = global_best_score, global_best_position
        return self.f_opt, self.x_opt