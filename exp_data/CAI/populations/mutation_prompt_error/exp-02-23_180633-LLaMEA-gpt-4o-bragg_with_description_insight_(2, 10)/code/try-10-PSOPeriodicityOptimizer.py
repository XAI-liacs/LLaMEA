import numpy as np

class PSOPeriodicityOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_coef = 2.0
        self.social_coef = 2.0
        self.bounds = None

    def particle_swarm_optimization(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_fitness = np.array([func(ind) for ind in position])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_idx]

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                inertia = self.inertia_weight * velocity[i]
                cognitive = self.cognitive_coef * np.random.random() * (personal_best_position[i] - position[i])
                social = self.social_coef * np.random.random() * (global_best_position - position[i])
                velocity[i] = inertia + cognitive + social
                position[i] = np.clip(position[i] + velocity[i], self.bounds.lb, self.bounds.ub)

                fitness = func(position[i])
                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = position[i]
                    personal_best_fitness[i] = fitness

            global_best_idx = np.argmin(personal_best_fitness)
            global_best_position = personal_best_position[global_best_idx]

        return global_best_position

    def periodicity_enhancing_local_search(self, func, initial_guess):
        # Custom objective to enforce periodicity
        def periodic_objective(x):
            return func(x) + 0.01 * np.sum(np.abs(np.diff(x, n=2)))

        res = minimize(periodic_objective, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_global_solution = self.particle_swarm_optimization(func)
        best_solution = self.periodicity_enhancing_local_search(func, best_global_solution)
        return best_solution