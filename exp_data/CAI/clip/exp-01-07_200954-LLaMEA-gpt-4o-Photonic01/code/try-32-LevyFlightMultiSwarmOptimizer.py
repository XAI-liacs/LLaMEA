import numpy as np

class LevyFlightMultiSwarmOptimizer:
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

    def levy_flight(self, size, beta=1.5):
        # Implementing Levy Flight
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

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

            for swarm in self.swarms:
                inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * (iteration / self.iterations)
                r1, r2, r3 = np.random.rand(3)
                
                # Adaptive cognitive and social weights
                cognitive_weight = self.cognitive_weight * (1 - iteration / self.iterations)
                social_weight = self.social_weight * (iteration / self.iterations)
                
                # Update velocities with Levy flight influence
                levy_step = self.levy_flight((self.population_size, self.dim)) * 0.01  # Levy step size
                swarm['velocities'] = (inertia_weight * swarm['velocities']
                                       + cognitive_weight * r1 * (swarm['personal_best_positions'] - swarm['positions'])
                                       + social_weight * r2 * (global_best_position - swarm['positions'])
                                       + 0.8 * r3 * (np.mean(swarm['positions'], axis=0) - swarm['positions'])
                                       + levy_step)
                swarm['positions'] = np.clip(swarm['positions'] + swarm['velocities'], lb, ub)

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