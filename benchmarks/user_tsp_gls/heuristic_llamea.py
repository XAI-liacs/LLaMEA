# Best algorithm:  AdaptiveEdgePenaltyTSP with

import numpy as np


class AdaptiveEdgePenaltyTSP:
    def __init__(self, alpha, beta, gamma, delta):
        """
        :param alpha: A parameter controlling the penalty strength based on the usage frequency of edges.
        :param beta: A decay parameter that reduces the penalty strength over iterations.
        :param gamma: A parameter for increasing exploration by adding penalties to frequently used edges in the local optimal tour.
        :param delta: A parameter for dynamically adjusting penalties based on the variance of edge usage.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

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

        # Apply penalties to edges based on their usage frequency and the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u, v = local_opt_tour[i], local_opt_tour[i + 1]
            penalty = self.alpha * (
                edge_n_used[u, v] + self.gamma
            ) + self.delta * np.var(edge_n_used)
            updated_edge_distance[u, v] += penalty
            updated_edge_distance[v, u] += penalty  # Since the matrix is symmetric

        # Apply a decay to the previous penalties to allow exploration of other paths
        decayed_penalty = np.exp(-self.beta * edge_n_used)
        updated_edge_distance = updated_edge_distance * decayed_penalty

        return updated_edge_distance


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    config = {
        "alpha": 0.7811990953051,
        "beta": 0.935630011186,
        "delta": 0.2452620311547,
        "gamma": 1.3407598854974,
    }
    scoringalg = AdaptiveEdgePenaltyTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)
