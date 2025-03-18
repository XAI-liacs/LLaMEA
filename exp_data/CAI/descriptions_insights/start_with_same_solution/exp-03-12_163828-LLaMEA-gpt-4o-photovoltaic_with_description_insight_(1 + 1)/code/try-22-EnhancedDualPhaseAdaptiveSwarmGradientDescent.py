import numpy as np

class EnhancedDualPhaseAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + 2 * int(np.sqrt(dim))
        self.population_size = self.initial_population_size
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.01
        self.global_search_fraction = 0.3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        phase_change = self.budget // 2

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.4 + 0.3 * adaptive_factor
            cognitive_coeff = 2.0 if evaluations < phase_change else 1.5
            social_coeff = 1.5 if evaluations < phase_change else 1.9

            dynamic_pop_size = max(2, int(self.initial_population_size * (self.global_search_fraction + (1 - self.global_search_fraction) * evaluations / self.budget)))
            if dynamic_pop_size != self.population_size:
                self.population_size = dynamic_pop_size
                swarm = swarm[:self.population_size]
                self.velocity = self.velocity[:self.population_size]
                personal_best = personal_best[:self.population_size]
                personal_best_value = personal_best_value[:self.population_size]

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                learning_rate = 0.3 * adaptive_factor
                self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_coeff * r1 * (personal_best[i] - swarm[i]) + social_coeff * r2 * (global_best - swarm[i])
                swarm[i] += learning_rate * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                noise = np.random.normal(0, self.noise_factor * 0.5 * adaptive_factor, self.dim)
                swarm[i] += noise
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value