import numpy as np

class DualPhaseAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.01
        self.layer_increment = int(dim / 4)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        phase_change = self.budget // 3 

        while evaluations < self.budget:
            if evaluations % phase_change == 0 and evaluations // phase_change < 3:
                self.dim += self.layer_increment
                self.velocity = np.resize(self.velocity, (self.population_size, self.dim))
                swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
                personal_best = np.resize(personal_best, (self.population_size, self.dim))

            adaptive_factor = (1 - evaluations / self.budget) ** 2
            inertia_weight = 0.5 + 0.2 * adaptive_factor
            cognitive_coeff = 2.2 if evaluations < phase_change else 1.4
            social_coeff = 1.4 if evaluations < phase_change else 1.8  

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                learning_rate = 0.2 * adaptive_factor
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += learning_rate * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                noise = np.random.normal(0, self.noise_factor * (1 - adaptive_factor), self.dim)
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