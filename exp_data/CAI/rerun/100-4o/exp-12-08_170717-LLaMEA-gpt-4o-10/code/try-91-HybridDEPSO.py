import numpy as np

class HybridDEPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 50
        self.cr = 0.9
        self.f = 0.8
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = pop.copy()
        personal_best_scores = np.array([func(x) for x in pop])
        
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()

        evals = self.pop_size

        while evals < self.budget:
            # Differential Evolution Step
            self.f = 0.5 + 0.3 * np.random.rand()  # Adaptive F
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.f * (b - c)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < personal_best_scores[i]:
                    personal_best_scores[i] = trial_fitness
                    personal_best_positions[i] = trial

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

            # Particle Swarm Optimization Step
            if evals >= self.budget:
                break
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            pop = np.clip(pop + velocities, lb, ub)

            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                fitness = func(pop[i])
                evals += 1

                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = pop[i]

                    if fitness < self.f_opt:
                        self.f_opt = fitness
                        self.x_opt = pop[i]
                        global_best_position = pop[i]
            
            # Local Search around the best
            if evals + self.dim <= self.budget:
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                local_search_candidate = np.clip(global_best_position + perturbation, lb, ub)
                local_search_fitness = func(local_search_candidate)
                evals += self.dim

                if local_search_fitness < self.f_opt:
                    self.f_opt = local_search_fitness
                    self.x_opt = local_search_candidate
                    global_best_position = local_search_candidate

        return self.f_opt, self.x_opt