import numpy as np

class HybridDEPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.w = 0.5  # PSO inertia weight
        self.c1 = 1.5  # PSO cognitive parameter
        self.c2 = 1.5  # PSO social parameter

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.rand(self.population_size, self.dim)
        personal_best_positions = np.copy(pop)
        personal_best_values = np.array([func(x) for x in pop])
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        
        evaluations = len(pop)

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Step
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate Trial
                f = func(trial)
                evaluations += 1
                if f < personal_best_values[i]:
                    personal_best_values[i] = f
                    personal_best_positions[i] = trial

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
            
            # Particle Swarm Optimization Step
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.c2 * r2 * (global_best_position - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)

            # Update Global Best
            global_best_index = np.argmin(personal_best_values)
            global_best_position = personal_best_positions[global_best_index]
            self.f_opt = personal_best_values[global_best_index]
            self.x_opt = global_best_position

        return self.f_opt, self.x_opt