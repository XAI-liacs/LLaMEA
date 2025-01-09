import numpy as np

class HPEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.f = 0.8  # differential weight
        self.cr = 0.9  # crossover probability

    def __call__(self, func):
        np.random.seed(42)

        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest_positions = np.copy(positions)
        pbest_scores = np.array([func(ind) for ind in positions])
        gbest_index = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_index]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            # PSO Update
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest_positions - positions) +
                          self.c2 * r2 * (gbest_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            # Evaluate
            scores = np.array([func(ind) for ind in positions])
            eval_count += self.population_size

            # Update personal bests
            better_idxs = scores < pbest_scores
            pbest_scores[better_idxs] = scores[better_idxs]
            pbest_positions[better_idxs] = positions[better_idxs]
            
            # Update global best
            gbest_index = np.argmin(pbest_scores)
            gbest_position = pbest_positions[gbest_index]

            # DE Update
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = positions[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, positions[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < scores[i]:
                    positions[i] = trial
                    scores[i] = trial_score

            # Update global best from DE
            min_score_index = np.argmin(scores)
            min_score = scores[min_score_index]
            if min_score < pbest_scores[gbest_index]:
                gbest_position = positions[min_score_index]
                pbest_scores[gbest_index] = min_score

        return gbest_position