import numpy as np
from sklearn.cluster import KMeans

class QuantumEnhancedAdaptivePSO_DynamicClustering:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1
        self.num_clusters = max(2, dim // 10)

    def quantum_update(self, position, personal_best, cluster_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = (position + personal_best) / 2 + quantum_factor * (cluster_best - position) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(pop)
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = personal_best_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            inertia_weight = (self.initial_inertia_weight - self.final_inertia_weight) * \
                             (1 - eval_count / self.budget) + self.final_inertia_weight

            kmeans = KMeans(n_clusters=self.num_clusters)
            cluster_labels = kmeans.fit_predict(pop)
            
            cluster_bests = []
            for cluster in range(self.num_clusters):
                cluster_indices = np.where(cluster_labels == cluster)[0]
                cluster_best_idx = cluster_indices[np.argmin(personal_best_values[cluster_indices])]
                cluster_bests.append(personal_best[cluster_best_idx])

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cluster_best = cluster_bests[cluster_labels[i]]
                velocities[i] = (inertia_weight * velocities[i]
                                 + self.c1 * r1 * (personal_best[i] - pop[i])
                                 + self.c2 * r2 * (cluster_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], bounds[:, 0], bounds[:, 1])

                trial = self.quantum_update(pop[i], personal_best[i], cluster_best, eval_count)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                
                trial_value = func(trial)
                eval_count += 1
                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if eval_count >= self.budget:
                    break

        return global_best