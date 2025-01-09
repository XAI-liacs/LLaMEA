import numpy as np

class DEPSO:
    def __init__(self, budget, dim, pop_size=50, F=0.5, CR=0.9, w=0.9, c1=2.0, c2=2.0):  # Change 1: w initial to 0.9 for better exploration
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential evolution scaling factor
        self.CR = CR  # Crossover rate
        self.w = w  # Inertia weight for PSO
        self.c1 = c1  # Cognitive coefficient for PSO
        self.c2 = c2  # Social coefficient for PSO
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Differential evolution mutation
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                mutant = pop[idxs[0]] + self.F * (pop[idxs[1]] - pop[idxs[2]])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score

            # Particle swarm optimization update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best - pop) + 
                          self.c2 * r2 * (global_best - pop))
            pop = np.clip(pop + velocities, self.lower_bound, self.upper_bound)
            
            self.w = 0.9 - (0.5 * (eval_count / self.budget))  # Change 2: Adaptive inertia weight decreasing over time

        return global_best, global_best_score