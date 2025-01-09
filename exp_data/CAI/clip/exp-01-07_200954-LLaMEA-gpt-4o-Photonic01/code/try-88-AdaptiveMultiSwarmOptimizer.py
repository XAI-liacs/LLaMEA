import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_min = 0.2
        self.inertia_max = 0.9
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.num_swarms = 3
        self.swarms = [self._initialize_swarm() for _ in range(self.num_swarms)]
        self.iterations = budget // (self.population_size * self.num_swarms)

    def _initialize_swarm(self):
        return {
            'positions': None,
            'velocities': None,
            'fitness': None,
            'personal_best_positions': None,
            'personal_best_fitness': None
        }

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        global_best_position = None
        global_best_fitness = float('inf')

        for swarm in self.swarms:
            swarm['positions'] = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
            swarm['velocities'] = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim)) * (ub - lb)
            swarm['fitness'] = np.array([func(ind) for ind in swarm['positions']])
            evals += self.population_size

            swarm['personal_best_positions'] = np.copy(swarm['positions'])
            swarm['personal_best_fitness'] = np.copy(swarm['fitness'])

            min_idx = np.argmin(swarm['fitness'])
            if swarm['fitness'][min_idx] < global_best_fitness:
                global_best_position = swarm['positions'][min_idx]
                global_best_fitness = swarm['fitness'][min_idx]

        for iteration in range(self.iterations):
            if evals >= self.budget:
                break

            for i, swarm in enumerate(self.swarms):
                inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * ((iteration / self.iterations)**0.5)
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()

                adaptive_cognitive_weight = self.cognitive_weight * (1 - (evals / self.budget)**0.3)
                adaptive_social_weight = self.social_weight * ((evals / self.budget)**0.5)

                feedback_term = 0.15 * np.std(swarm['fitness']) * (1 - iteration / self.iterations) * (0.5 + 0.5 * np.sin(2 * np.pi * iteration / self.iterations))

                # Introduce influence from best particles of neighboring swarms
                neighbor_influence = 0.2 * (self.swarms[(i-1) % self.num_swarms]['personal_best_positions'][np.argmin(self.swarms[(i-1) % self.num_swarms]['personal_best_fitness'])] - swarm['positions'])

                swarm['velocities'] = (inertia_weight * swarm['velocities']
                                       + adaptive_cognitive_weight * r1 * (swarm['personal_best_positions'] - swarm['positions'])
                                       + adaptive_social_weight * r2 * (global_best_position - swarm['positions'])
                                       + 0.8 * r3 * (np.mean(swarm['positions'], axis=0) - swarm['positions'])
                                       + neighbor_influence
                                       + feedback_term * np.random.randn(*swarm['positions'].shape))
                swarm['positions'] = np.clip(swarm['positions'] + swarm['velocities'] + 0.02 * np.random.uniform(-1, 1, swarm['positions'].shape), lb, ub)

                swarm['fitness'] = np.array([func(ind) for ind in swarm['positions']])
                evals += self.population_size

                better_mask = swarm['fitness'] < swarm['personal_best_fitness']
                swarm['personal_best_positions'][better_mask] = swarm['positions'][better_mask]
                swarm['personal_best_fitness'][better_mask] = swarm['fitness'][better_mask]

                min_idx = np.argmin(swarm['fitness'])
                if swarm['fitness'][min_idx] < global_best_fitness:
                    global_best_position = swarm['positions'][min_idx]
                    global_best_fitness = swarm['fitness'][min_idx]

        return global_best_position, global_best_fitness