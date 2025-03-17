import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - (evaluations / self.budget)**2
            inertia_weight = 0.7 + 0.3 * adaptive_factor
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.5 - 0.5 * adaptive_factor

            learning_rate = 0.5 * (1 - evaluations / self.budget)

            elite_indices = np.argsort(personal_best_value)[:max(1, self.population_size // 5)]
            elite_swarm = swarm[elite_indices]

            dynamic_mutation_rate = 0.1 * (1 - adaptive_factor)

            # Ecological Niche Partitioning
            niche_radius = np.sqrt(self.dim) * adaptive_factor
            niches = [np.mean(swarm[i:i+5], axis=0) for i in range(0, self.population_size, 5)]
            
            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                niche_influence = niches[np.random.randint(len(niches))]
                niche_distance = np.linalg.norm(swarm[i] - niche_influence)
                if niche_distance < niche_radius:
                    swarm[i] += 0.25 * (niche_influence - swarm[i])
                
                # Hybrid Crossover Strategy
                if np.random.rand() < 0.3:
                    partner = swarm[np.random.randint(self.population_size)]
                    crossover_point = np.random.randint(1, self.dim)
                    swarm[i][:crossover_point] = partner[:crossover_point]

                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += learning_rate * self.velocity[i]

                if np.random.rand() < dynamic_mutation_rate:
                    mutation_vector = np.random.normal(0, 0.05 * adaptive_factor, self.dim)
                    swarm[i] += mutation_vector

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

            elite_best_value = np.min([func(x) for x in elite_swarm])
            if elite_best_value < global_best_value:
                global_best_value = elite_best_value
                global_best = elite_swarm[np.argmin([func(x) for x in elite_swarm])]

        return global_best, global_best_value