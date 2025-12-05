"""
MedFedAgent Byzantine-Robust Aggregation Module

Implements robust aggregation methods to handle malicious or faulty client updates
in federated learning. These methods protect against Byzantine attacks where some
clients may send corrupted gradients.

Implemented Methods:
1. Coordinate-wise Median: Takes median of each parameter coordinate
2. Coordinate-wise Trimmed Mean: Trims extreme values before averaging
3. Krum: Selects the update closest to its neighbors
4. Multi-Krum: Selects multiple closest updates and averages
5. Bulyan: Combines Krum selection with trimmed mean
6. FoolsGold: Detects sybil attacks via gradient similarity

References:
- Blanchard et al., "Machine Learning with Adversaries" (NeurIPS 2017) - Krum
- Yin et al., "Byzantine-Robust Distributed Learning" (ICML 2018) - Median, Trimmed Mean
- El Mhamdi et al., "The Hidden Vulnerability of Distributed Learning" (ICML 2018) - Bulyan
- Fung et al., "FoolsGold: Mitigating Sybils in Federated Learning" (2020)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import warnings


class AggregationMethod(Enum):
    """Available robust aggregation methods."""
    FEDAVG = "fedavg"                    # Standard weighted average
    MEDIAN = "median"                     # Coordinate-wise median
    TRIMMED_MEAN = "trimmed_mean"        # Trimmed mean (remove extremes)
    KRUM = "krum"                         # Single-Krum
    MULTI_KRUM = "multi_krum"            # Multi-Krum
    BULYAN = "bulyan"                     # Bulyan (Krum + trimmed mean)
    FOOLSGOLD = "foolsgold"              # Sybil-resistant aggregation


@dataclass
class AggregationConfig:
    """Configuration for robust aggregation."""
    method: AggregationMethod = AggregationMethod.FEDAVG
    num_byzantine: int = 0                # Expected number of Byzantine clients
    trim_ratio: float = 0.1               # Ratio to trim for trimmed mean (0-0.5)
    multi_krum_k: int = 1                 # Number of clients to select in multi-Krum
    enable_detection: bool = True         # Enable Byzantine client detection
    detection_threshold: float = 3.0      # Z-score threshold for detection
    
    def __post_init__(self):
        if not 0 <= self.trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        if self.num_byzantine < 0:
            raise ValueError("num_byzantine must be non-negative")


@dataclass
class AggregationResult:
    """Result of aggregation operation."""
    aggregated_params: List[np.ndarray]
    method_used: str
    num_clients: int
    num_excluded: int
    excluded_clients: List[int]
    client_scores: Dict[int, float]       # Scores/distances per client
    is_byzantine_detected: bool
    detection_details: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> str:
        return (
            f"Aggregation: {self.method_used}, "
            f"clients={self.num_clients}, "
            f"excluded={self.num_excluded}, "
            f"byzantine_detected={self.is_byzantine_detected}"
        )


class ByzantineRobustAggregator:
    """
    Byzantine-robust aggregator for federated learning.
    
    Provides multiple aggregation strategies that are resilient to
    malicious or faulty client updates.
    """
    
    def __init__(self, config: AggregationConfig):
        """
        Initialize the aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config
        self.history: List[AggregationResult] = []
        self.client_trust_scores: Dict[int, float] = {}  # For FoolsGold
        self.gradient_history: Dict[int, List[np.ndarray]] = {}
        
        logger.info(f"ByzantineRobustAggregator initialized: method={config.method.value}")
    
    def aggregate(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]],
        global_params: Optional[List[np.ndarray]] = None
    ) -> AggregationResult:
        """
        Aggregate client updates using the configured method.
        
        Args:
            client_updates: List of (client_id, parameters, num_samples) tuples
            global_params: Current global model parameters (for computing deltas)
            
        Returns:
            AggregationResult with aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        method = self.config.method
        
        if method == AggregationMethod.FEDAVG:
            result = self._fedavg(client_updates)
        elif method == AggregationMethod.MEDIAN:
            result = self._coordinate_median(client_updates)
        elif method == AggregationMethod.TRIMMED_MEAN:
            result = self._trimmed_mean(client_updates)
        elif method == AggregationMethod.KRUM:
            result = self._krum(client_updates, multi=False)
        elif method == AggregationMethod.MULTI_KRUM:
            result = self._krum(client_updates, multi=True)
        elif method == AggregationMethod.BULYAN:
            result = self._bulyan(client_updates)
        elif method == AggregationMethod.FOOLSGOLD:
            result = self._foolsgold(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Store in history
        self.history.append(result)
        
        logger.debug(result.get_summary())
        
        return result
    
    def _fedavg(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]]
    ) -> AggregationResult:
        """Standard Federated Averaging."""
        total_samples = sum(ns for _, _, ns in client_updates)
        
        # Weighted average
        aggregated = [np.zeros_like(p) for p in client_updates[0][1]]
        
        for client_id, params, num_samples in client_updates:
            weight = num_samples / total_samples
            for i, p in enumerate(params):
                aggregated[i] += p * weight
        
        return AggregationResult(
            aggregated_params=aggregated,
            method_used="fedavg",
            num_clients=len(client_updates),
            num_excluded=0,
            excluded_clients=[],
            client_scores={cid: 1.0 for cid, _, _ in client_updates},
            is_byzantine_detected=False
        )
    
    def _coordinate_median(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]]
    ) -> AggregationResult:
        """
        Coordinate-wise median aggregation.
        
        For each parameter coordinate, takes the median value across clients.
        Robust to up to 50% Byzantine clients.
        """
        num_params = len(client_updates[0][1])
        aggregated = []
        
        for param_idx in range(num_params):
            # Stack all client values for this parameter
            stacked = np.stack([params[param_idx] for _, params, _ in client_updates])
            # Take coordinate-wise median
            median_param = np.median(stacked, axis=0)
            aggregated.append(median_param)
        
        # Compute distance from median for each client (for detection)
        client_scores = {}
        for client_id, params, _ in client_updates:
            distance = sum(
                np.linalg.norm(params[i] - aggregated[i])
                for i in range(num_params)
            )
            client_scores[client_id] = distance
        
        # Detect outliers
        excluded, is_byzantine = self._detect_outliers(client_scores)
        
        return AggregationResult(
            aggregated_params=aggregated,
            method_used="coordinate_median",
            num_clients=len(client_updates),
            num_excluded=len(excluded),
            excluded_clients=excluded,
            client_scores=client_scores,
            is_byzantine_detected=is_byzantine
        )
    
    def _trimmed_mean(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]]
    ) -> AggregationResult:
        """
        Coordinate-wise trimmed mean aggregation.
        
        For each parameter coordinate, removes the top and bottom trim_ratio
        values before computing the mean.
        """
        n_clients = len(client_updates)
        trim_count = int(n_clients * self.config.trim_ratio)
        
        if trim_count * 2 >= n_clients:
            logger.warning("Trim ratio too high, falling back to median")
            return self._coordinate_median(client_updates)
        
        num_params = len(client_updates[0][1])
        aggregated = []
        
        for param_idx in range(num_params):
            # Stack all client values
            stacked = np.stack([params[param_idx] for _, params, _ in client_updates])
            
            # Sort along client axis for each coordinate
            sorted_stacked = np.sort(stacked, axis=0)
            
            # Trim extremes
            if trim_count > 0:
                trimmed = sorted_stacked[trim_count:-trim_count]
            else:
                trimmed = sorted_stacked
            
            # Mean of remaining values
            trimmed_mean = np.mean(trimmed, axis=0)
            aggregated.append(trimmed_mean)
        
        # Compute scores
        client_scores = {}
        for client_id, params, _ in client_updates:
            distance = sum(
                np.linalg.norm(params[i] - aggregated[i])
                for i in range(num_params)
            )
            client_scores[client_id] = distance
        
        excluded, is_byzantine = self._detect_outliers(client_scores)
        
        return AggregationResult(
            aggregated_params=aggregated,
            method_used="trimmed_mean",
            num_clients=n_clients,
            num_excluded=len(excluded),
            excluded_clients=excluded,
            client_scores=client_scores,
            is_byzantine_detected=is_byzantine,
            detection_details={"trim_count": trim_count}
        )
    
    def _krum(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]],
        multi: bool = False
    ) -> AggregationResult:
        """
        Krum / Multi-Krum aggregation.
        
        Selects the update that is closest to its (n - f - 2) nearest neighbors,
        where f is the expected number of Byzantine clients.
        
        Multi-Krum averages the top-k selected updates.
        """
        n = len(client_updates)
        f = self.config.num_byzantine
        k = self.config.multi_krum_k if multi else 1
        
        # Need at least n > 2f + 2 for Krum
        if n <= 2 * f + 2:
            logger.warning(f"Not enough clients for Krum (n={n}, f={f}), falling back to median")
            return self._coordinate_median(client_updates)
        
        # Flatten parameters for distance computation
        flattened = []
        for client_id, params, _ in client_updates:
            flat = np.concatenate([p.flatten() for p in params])
            flattened.append((client_id, flat))
        
        # Compute pairwise distances
        n_clients = len(flattened)
        distances = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(flattened[i][1] - flattened[j][1])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores (sum of distances to n-f-2 nearest neighbors)
        num_neighbors = n - f - 2
        scores = []
        
        for i in range(n_clients):
            # Get sorted distances to other clients
            dists_to_others = np.sort(distances[i])
            # Sum of distances to nearest neighbors (excluding self which is 0)
            score = np.sum(dists_to_others[1:num_neighbors + 1])
            scores.append((i, score))
        
        # Sort by score (lower is better)
        scores.sort(key=lambda x: x[1])
        
        # Select top-k clients
        selected_indices = [scores[i][0] for i in range(min(k, n_clients))]
        
        # Map back to client IDs
        client_ids = [client_updates[i][0] for i in selected_indices]
        client_scores = {
            client_updates[i][0]: scores[i][1]
            for i in range(n_clients)
            for scores_idx, (idx, score) in enumerate(scores)
            if idx == i
        }
        client_scores = {client_updates[i][0]: s for i, s in scores}
        
        # Aggregate selected updates
        if k == 1:
            # Single Krum: use the best update directly
            aggregated = client_updates[selected_indices[0]][1]
        else:
            # Multi-Krum: average selected updates
            total_samples = sum(client_updates[i][2] for i in selected_indices)
            aggregated = [np.zeros_like(p) for p in client_updates[0][1]]
            
            for idx in selected_indices:
                _, params, num_samples = client_updates[idx]
                weight = num_samples / total_samples
                for i, p in enumerate(params):
                    aggregated[i] += p * weight
        
        # Excluded clients
        excluded = [
            client_updates[i][0]
            for i in range(n_clients)
            if i not in selected_indices
        ]
        
        return AggregationResult(
            aggregated_params=aggregated,
            method_used="multi_krum" if multi else "krum",
            num_clients=n_clients,
            num_excluded=len(excluded),
            excluded_clients=excluded,
            client_scores=client_scores,
            is_byzantine_detected=len(excluded) > 0,
            detection_details={"selected_clients": client_ids, "k": k}
        )
    
    def _bulyan(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]]
    ) -> AggregationResult:
        """
        Bulyan aggregation.
        
        Two-step process:
        1. Use Multi-Krum to select θ = n - 2f clients
        2. Apply coordinate-wise trimmed mean on selected clients
        """
        n = len(client_updates)
        f = self.config.num_byzantine
        
        # Need n >= 4f + 3 for Bulyan
        if n < 4 * f + 3:
            logger.warning(f"Not enough clients for Bulyan (n={n}, f={f}), falling back to Multi-Krum")
            return self._krum(client_updates, multi=True)
        
        theta = n - 2 * f
        
        # Step 1: Multi-Krum selection
        self.config.multi_krum_k = theta
        krum_result = self._krum(client_updates, multi=True)
        
        # Get selected client updates
        selected_indices = [
            i for i, (cid, _, _) in enumerate(client_updates)
            if cid not in krum_result.excluded_clients
        ]
        selected_updates = [client_updates[i] for i in selected_indices]
        
        # Step 2: Trimmed mean on selected clients
        if len(selected_updates) < 3:
            # Not enough for trimmed mean, just average
            return self._fedavg(selected_updates)
        
        # Apply trimmed mean with f/theta trim ratio
        old_trim = self.config.trim_ratio
        self.config.trim_ratio = min(0.4, f / theta) if theta > 0 else 0.1
        
        trimmed_result = self._trimmed_mean(selected_updates)
        
        # Restore config
        self.config.trim_ratio = old_trim
        
        return AggregationResult(
            aggregated_params=trimmed_result.aggregated_params,
            method_used="bulyan",
            num_clients=n,
            num_excluded=n - len(selected_updates),
            excluded_clients=krum_result.excluded_clients,
            client_scores=krum_result.client_scores,
            is_byzantine_detected=krum_result.is_byzantine_detected,
            detection_details={
                "krum_selected": theta,
                "final_aggregated": len(selected_updates)
            }
        )
    
    def _foolsgold(
        self,
        client_updates: List[Tuple[int, List[np.ndarray], int]]
    ) -> AggregationResult:
        """
        FoolsGold aggregation for Sybil attack resistance.
        
        Detects clients with similar gradient updates (potential Sybils)
        and reduces their influence on the aggregation.
        """
        n_clients = len(client_updates)
        
        # Flatten updates for similarity computation
        flattened = {}
        for client_id, params, _ in client_updates:
            flat = np.concatenate([p.flatten() for p in params])
            flattened[client_id] = flat
            
            # Update gradient history
            if client_id not in self.gradient_history:
                self.gradient_history[client_id] = []
            self.gradient_history[client_id].append(flat)
        
        # Compute pairwise cosine similarities
        client_ids = list(flattened.keys())
        n = len(client_ids)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                vec_i = flattened[client_ids[i]]
                vec_j = flattened[client_ids[j]]
                
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    cos_sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                else:
                    cos_sim = 0
                
                similarities[i, j] = max(0, cos_sim)  # Only positive similarities
                similarities[j, i] = similarities[i, j]
        
        # Compute contribution scores (inversely proportional to max similarity)
        scores = np.zeros(n)
        for i in range(n):
            max_sim = np.max(similarities[i, :])
            # Higher similarity = lower score = less contribution
            scores[i] = 1.0 - max_sim
        
        # Normalize scores
        if np.sum(scores) > 0:
            scores = scores / np.sum(scores)
        else:
            scores = np.ones(n) / n
        
        # Update trust scores
        for i, cid in enumerate(client_ids):
            self.client_trust_scores[cid] = float(scores[i])
        
        # Aggregate with FoolsGold weights
        aggregated = [np.zeros_like(p) for p in client_updates[0][1]]
        
        for i, (client_id, params, num_samples) in enumerate(client_updates):
            weight = scores[client_ids.index(client_id)]
            for j, p in enumerate(params):
                aggregated[j] += p * weight
        
        # Identify potential Sybils (high similarity pairs)
        sybil_threshold = 0.9
        potential_sybils = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > sybil_threshold:
                    potential_sybils.extend([client_ids[i], client_ids[j]])
        potential_sybils = list(set(potential_sybils))
        
        client_scores = {cid: float(scores[i]) for i, cid in enumerate(client_ids)}
        
        return AggregationResult(
            aggregated_params=aggregated,
            method_used="foolsgold",
            num_clients=n_clients,
            num_excluded=0,  # FoolsGold doesn't exclude, just reweights
            excluded_clients=[],
            client_scores=client_scores,
            is_byzantine_detected=len(potential_sybils) > 0,
            detection_details={
                "potential_sybils": potential_sybils,
                "trust_scores": dict(client_scores)
            }
        )
    
    def _detect_outliers(
        self,
        client_scores: Dict[int, float]
    ) -> Tuple[List[int], bool]:
        """
        Detect outlier clients based on their aggregation scores.
        
        Args:
            client_scores: Distance/score per client
            
        Returns:
            Tuple of (excluded client IDs, whether Byzantine was detected)
        """
        if not self.config.enable_detection or len(client_scores) < 3:
            return [], False
        
        scores = np.array(list(client_scores.values()))
        client_ids = list(client_scores.keys())
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return [], False
        
        # Z-score based detection
        z_scores = (scores - mean_score) / std_score
        
        excluded = []
        for i, z in enumerate(z_scores):
            if abs(z) > self.config.detection_threshold:
                excluded.append(client_ids[i])
        
        return excluded, len(excluded) > 0
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about aggregation history."""
        if not self.history:
            return {}
        
        return {
            "total_rounds": len(self.history),
            "method": self.config.method.value,
            "avg_excluded": np.mean([r.num_excluded for r in self.history]),
            "total_byzantine_detections": sum(
                1 for r in self.history if r.is_byzantine_detected
            ),
            "client_trust_scores": self.client_trust_scores
        }


def create_robust_aggregator(
    method: str = "fedavg",
    num_byzantine: int = 0,
    trim_ratio: float = 0.1,
    **kwargs
) -> ByzantineRobustAggregator:
    """
    Factory function to create a robust aggregator.
    
    Args:
        method: Aggregation method name
        num_byzantine: Expected number of Byzantine clients
        trim_ratio: Trim ratio for trimmed mean
        **kwargs: Additional configuration options
        
    Returns:
        Configured ByzantineRobustAggregator
    """
    method_enum = AggregationMethod(method.lower())
    
    config = AggregationConfig(
        method=method_enum,
        num_byzantine=num_byzantine,
        trim_ratio=trim_ratio,
        **kwargs
    )
    
    return ByzantineRobustAggregator(config)


def simulate_byzantine_attack(
    num_honest: int = 5,
    num_byzantine: int = 2,
    param_shape: Tuple[int, ...] = (100,),
    attack_type: str = "random"
) -> List[Tuple[int, List[np.ndarray], int]]:
    """
    Simulate client updates with Byzantine attackers.
    
    Args:
        num_honest: Number of honest clients
        num_byzantine: Number of Byzantine clients
        param_shape: Shape of parameter arrays
        attack_type: Type of attack ("random", "sign_flip", "scale")
        
    Returns:
        List of (client_id, parameters, num_samples) tuples
    """
    updates = []
    
    # Honest clients: similar gradients with small noise
    base_gradient = np.random.randn(*param_shape).astype(np.float32)
    
    for i in range(num_honest):
        noise = np.random.randn(*param_shape).astype(np.float32) * 0.1
        params = [base_gradient + noise]
        updates.append((i, params, 100))
    
    # Byzantine clients
    for i in range(num_byzantine):
        client_id = num_honest + i
        
        if attack_type == "random":
            # Random garbage
            params = [np.random.randn(*param_shape).astype(np.float32) * 10]
        elif attack_type == "sign_flip":
            # Flip signs to push model in wrong direction
            params = [-base_gradient * 5]
        elif attack_type == "scale":
            # Scaled attack
            params = [base_gradient * 100]
        else:
            params = [np.zeros(param_shape, dtype=np.float32)]
        
        updates.append((client_id, params, 100))
    
    return updates


def demonstrate_robust_aggregation():
    """Demonstrate Byzantine-robust aggregation methods."""
    print("\n" + "=" * 70)
    print("BYZANTINE-ROBUST AGGREGATION DEMONSTRATION")
    print("=" * 70)
    
    # Simulate scenario with 2 Byzantine clients out of 7
    num_honest = 5
    num_byzantine = 2
    
    print(f"\nScenario: {num_honest} honest clients, {num_byzantine} Byzantine attackers")
    print(f"Attack type: sign_flip (adversarial gradient direction)")
    
    updates = simulate_byzantine_attack(
        num_honest=num_honest,
        num_byzantine=num_byzantine,
        param_shape=(50,),
        attack_type="sign_flip"
    )
    
    # True mean of honest clients
    honest_mean = np.mean(
        [updates[i][1][0] for i in range(num_honest)],
        axis=0
    )
    
    methods = [
        ("fedavg", {}),
        ("median", {}),
        ("trimmed_mean", {"trim_ratio": 0.2}),
        ("krum", {"num_byzantine": 2}),
        ("multi_krum", {"num_byzantine": 2, "multi_krum_k": 3}),
        ("foolsgold", {})
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Method':<20} {'Error vs Honest':<20} {'Byzantine Detected':<20}")
    print("-" * 70)
    
    for method_name, kwargs in methods:
        aggregator = create_robust_aggregator(method=method_name, **kwargs)
        result = aggregator.aggregate(updates)
        
        # Error compared to honest mean
        error = np.linalg.norm(result.aggregated_params[0] - honest_mean)
        detected = "Yes" if result.is_byzantine_detected else "No"
        
        print(f"{method_name:<20} {error:<20.4f} {detected:<20}")
    
    print("-" * 70)
    print("\nNote: Lower error = better protection against Byzantine attack")
    print("FedAvg has high error because it includes Byzantine updates")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_robust_aggregation()
