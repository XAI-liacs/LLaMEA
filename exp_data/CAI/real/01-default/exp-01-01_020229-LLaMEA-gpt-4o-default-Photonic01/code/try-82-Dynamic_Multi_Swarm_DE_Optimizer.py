import numpy as np

class Dynamic_Multi_Swarm_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_count = 3
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.adaptation_factor = 0.99
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        swarms = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        best_positions = [swarm[np.argmin([func(ind) for ind in swarm])] for swarm in swarms]
        global_best_position = min(best_positions, key=func)
        global_best_value = func(global_best_position)
        
        evaluations = self.population_size * self.swarm_count

        while evaluations < self.budget:
            for s in range(self.swarm_count):
                new_swarm = []
                for i in range(self.population_size):
                    a, b, c = swarms[s][np.random.choice(self.population_size, 3, replace=False)]
                    mutant_vector = np.clip(a + self.F * (b - c), lb, ub)
                    trial_vector = np.copy(swarms[s][i])
                    
                    for j in range(self.dim):
                        if np.random.rand() < self.CR:
                            trial_vector[j] = mutant_vector[j]
                    
                    trial_value = func(trial_vector)
                    evaluations += 1

                    if trial_value < func(swarms[s][i]):
                        new_swarm.append(trial_vector)
                    else:
                        new_swarm.append(swarms[s][i])

                    if trial_value < func(best_positions[s]):
                        best_positions[s] = trial_vector

                    if trial_value < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_value

                    if evaluations >= self.budget:
                        break

                swarms[s] = np.array(new_swarm)
                self.F *= self.adaptation_factor
                self.CR *= self.adaptation_factor

                # Adaptive Multi-Swarm Interaction
                if s < self.swarm_count - 1 and evaluations % 5 == 0:
                    exchange_ratio = 0.1
                    exchange_count = int(exchange_ratio * self.population_size)
                    for _ in range(exchange_count):
                        idx = np.random.randint(self.population_size)
                        swarms[s][idx] = swarms[s+1][np.random.randint(self.population_size)]

        return global_best_position, global_best_value