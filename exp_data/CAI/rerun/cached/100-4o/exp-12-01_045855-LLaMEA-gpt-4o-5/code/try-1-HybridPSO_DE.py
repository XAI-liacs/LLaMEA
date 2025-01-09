import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.c1 = 1.49445  # Cognitive component
        self.c2 = 1.49445  # Social component
        self.w = 0.729  # Inertia weight
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        np.random.seed(0)
        bounds = (-5.0, 5.0)
        # Initialize particle swarm
        positions = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(bounds[1] - bounds[0]), abs(bounds[1] - bounds[0]), (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        # Main optimization loop
        eval_count = self.population_size
        while eval_count < self.budget:
            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = positions[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, positions[i])
                
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_position = trial
                        global_best_score = trial_score
                if eval_count >= self.budget:
                    break
            
            # Particle Swarm update
            if eval_count < self.budget:
                r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
                velocities = (self.w * velocities 
                              + self.c1 * r1 * (personal_best_positions - positions)
                              + self.c2 * r2 * (global_best_position - positions))
                positions = np.clip(positions + velocities, bounds[0], bounds[1])

                for i in range(self.population_size):
                    score = func(positions[i])
                    eval_count += 1
                    if score < personal_best_scores[i]:
                        personal_best_positions[i] = positions[i]
                        personal_best_scores[i] = score
                        if score < global_best_score:
                            global_best_position = positions[i]
                            global_best_score = score
                    if eval_count >= self.budget:
                        break

        return global_best_position