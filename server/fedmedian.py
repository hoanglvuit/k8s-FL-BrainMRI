import numpy as np
from typing import List, Tuple

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedMedian(FedAvg):
    """Federated Median Strategy."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Parameters | None, dict[str, Scalar]]:

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # ---- Convert client parameters to ndarray list ----
        weights = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # weights: List[client][layer][ndarray]

        num_layers = len(weights[0])
        median_weights = []

        for layer_idx in range(num_layers):
            # Stack all clients' weights for this layer
            layer_stack = np.stack(
                [client[layer_idx] for client in weights],
                axis=0
            )
            # Take median along client axis
            layer_median = np.median(layer_stack, axis=0)
            median_weights.append(layer_median)

        parameters_aggregated = ndarrays_to_parameters(median_weights)

        # ---- Metrics aggregation (reuse FedAvg logic) ----
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated
