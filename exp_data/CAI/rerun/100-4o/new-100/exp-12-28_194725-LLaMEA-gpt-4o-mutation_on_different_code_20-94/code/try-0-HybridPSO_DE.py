import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))

        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(x) for x in pop])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        w, c1, c2 = 0.7, 1.5, 1.5  # PSO parameters
        F, CR = 0.8, 0.9           # DE parameters
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - pop) +
                          c2 * r2 * (global_best_position - pop))
            pop = np.clip(pop + velocities, lb, ub)

            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < personal_best_scores[i]:
                    personal_best_scores[i] = f_trial
                    personal_best_positions[i] = trial
                    if f_trial < func(global_best_position):
                        global_best_position = trial

            # Update global best
            current_global_best_index = np.argmin(personal_best_scores)
            current_global_best_score = personal_best_scores[current_global_best_index]
            if current_global_best_score < func(global_best_position):
                global_best_position = personal_best_positions[current_global_best_index]

            if eval_count >= self.budget:
                break

        self.f_opt = func(global_best_position)
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt