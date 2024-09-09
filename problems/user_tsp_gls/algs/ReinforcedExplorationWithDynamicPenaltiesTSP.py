import numpy as np

class ReinforcedExplorationWithDynamicPenaltiesTSP:
    def __init__(self, alpha, beta, gamma, delta, epsilon, zeta, eta):
        """
        :param alpha: A parameter controlling the penalty strength based on the usage frequency of edges.
        :param beta: A decay parameter that reduces the penalty strength over iterations.
        :param gamma: A parameter for increasing exploration by adding penalties to frequently used edges in the local optimal tour.
        :param delta: A parameter for dynamically adjusting penalties based on the variance of edge usage.
        :param epsilon: A parameter for boosting the penalty for the most frequently used edges.
        :param zeta: A parameter for introducing randomness to escape local optima.
        :param eta: A parameter for reinforcing penalties to edges already in the local optimal tour.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        """
        Updates the edge distances to avoid being trapped in the local optimum.
        
        :param edge_distance: 2D numpy array representing the edge distances.
        :param local_opt_tour: 1D numpy array representing the current local optimal tour.
        :param edge_n_used: 2D numpy array representing the number of times each edge has been used.
        :return: 2D numpy array representing the updated edge distances.
        """
        n = edge_distance.shape[0]
        updated_edge_distance = np.copy(edge_distance)

        # Compute the variance of edge usage for dynamic adjustment
        edge_usage_variance = np.var(edge_n_used)

        # Determine the most frequently used edges
        max_usage = np.max(edge_n_used)

        # Apply penalties to edges based on their usage frequency and the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u, v = local_opt_tour[i], local_opt_tour[i + 1]
            # Basic usage penalty
            usage_penalty = self.alpha * (edge_n_used[u, v] + self.gamma) + self.delta * edge_usage_variance

            # Boost penalty for the most frequently used edges
            if edge_n_used[u, v] == max_usage:
                usage_penalty *= self.epsilon

            # Extra penalty for edges in the local optimal tour
            tour_penalty = self.eta * edge_n_used[u, v]
            
            updated_edge_distance[u, v] += usage_penalty + tour_penalty
            updated_edge_distance[v, u] += usage_penalty + tour_penalty  # Since the matrix is symmetric

        # Apply a decay to the previous penalties to allow exploration of other paths
        decayed_penalty = np.exp(-self.beta * edge_n_used)
        updated_edge_distance = updated_edge_distance * decayed_penalty

        # Introduce randomness to help escape local optima
        random_penalty = np.random.rand(n, n) * self.zeta
        updated_edge_distance += random_penalty

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # -3.700743415417189e-15
    config = {'alpha': 0.2604586150497, 'beta': 0.439248911757, 'delta': 1.4343399837613, 'epsilon': 3.9377145059407, 'eta': 1.6553582492284, 'gamma': 0.3244421521202, 'zeta': 0.0464210699685}
    scoringalg = ReinforcedExplorationWithDynamicPenaltiesTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)