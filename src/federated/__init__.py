"""Federated learning module for MedFedAgent."""

from .client import (
    MedFedClient,
    create_flower_client,
    get_parameters,
    set_parameters
)

from .server import (
    MedFedStrategy,
    RoundMetrics,
    create_server_config,
    create_medfed_strategy
)

from .robust_aggregation import (
    AggregationMethod,
    AggregationConfig,
    AggregationResult,
    ByzantineRobustAggregator,
    create_robust_aggregator,
    simulate_byzantine_attack,
    demonstrate_robust_aggregation
)

__all__ = [
    # Client
    "MedFedClient",
    "create_flower_client",
    "get_parameters",
    "set_parameters",
    # Server
    "MedFedStrategy",
    "RoundMetrics",
    "create_server_config",
    "create_medfed_strategy",
    # Robust Aggregation
    "AggregationMethod",
    "AggregationConfig",
    "AggregationResult",
    "ByzantineRobustAggregator",
    "create_robust_aggregator",
    "simulate_byzantine_attack",
    "demonstrate_robust_aggregation"
]
