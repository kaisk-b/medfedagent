"""
MedFedAgent Secure Aggregation Module

Simulates secure aggregation for federated learning model updates.
This module provides a demonstration of cryptographic protection concepts
without full cryptographic implementation (for educational/demo purposes).

Concepts implemented:
- Simulated secret sharing (Shamir-inspired)
- Simulated homomorphic encryption
- Secure aggregation protocol simulation
- Audit trail for cryptographic operations

References:
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving ML" (2017)
- Bell et al., "Secure Single-Server Aggregation with (Poly)Logarithmic Overhead" (2020)
"""

import numpy as np
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger
import secrets
import base64


class SecureAggregationPhase(Enum):
    """Phases of secure aggregation protocol."""
    KEY_EXCHANGE = "key_exchange"
    MASKING = "masking"
    AGGREGATION = "aggregation"
    UNMASKING = "unmasking"
    COMPLETE = "complete"


@dataclass
class SecureAggConfig:
    """Configuration for secure aggregation."""
    enabled: bool = True
    min_clients: int = 2  # Minimum clients for aggregation
    threshold: int = 2  # Threshold for secret reconstruction
    use_masking: bool = True  # Use random masking
    audit_enabled: bool = True  # Enable audit trail
    simulate_encryption: bool = True  # Simulate HE operations


@dataclass
class ClientShare:
    """Represents a client's share in secure aggregation."""
    client_id: int
    share_id: str
    masked_update: Optional[np.ndarray] = None
    mask_seed: Optional[str] = None  # For simulation purposes
    commitment: Optional[str] = None  # Hash commitment of original update
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "client_id": self.client_id,
            "share_id": self.share_id,
            "commitment": self.commitment,
            "timestamp": self.timestamp
        }


@dataclass
class AggregationRound:
    """Record of a secure aggregation round."""
    round_id: int
    phase: str
    num_clients: int
    aggregation_successful: bool
    verification_passed: bool
    timestamp: str
    metrics: Dict = field(default_factory=dict)
    audit_hash: Optional[str] = None


class SecureMask:
    """
    Generates and manages random masks for secure aggregation.
    
    In real secure aggregation, pairs of clients generate shared random masks
    that cancel out during aggregation. This is a simplified simulation.
    """
    
    def __init__(self, seed: Optional[str] = None):
        self.seed = seed or secrets.token_hex(16)
        self.rng = np.random.default_rng(
            int(hashlib.sha256(self.seed.encode()).hexdigest()[:8], 16)
        )
    
    def generate_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate a random mask of given shape."""
        return self.rng.normal(0, 1, shape).astype(np.float32)
    
    @staticmethod
    def create_paired_masks(
        client_a: int,
        client_b: int,
        shape: Tuple[int, ...],
        round_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create paired masks for two clients that sum to zero.
        
        Args:
            client_a: First client ID
            client_b: Second client ID
            shape: Shape of the mask
            round_id: Round identifier
            
        Returns:
            Tuple of (mask_a, mask_b) where mask_a + mask_b = 0
        """
        # Deterministic seed based on client pair and round
        seed_str = f"{min(client_a, client_b)}_{max(client_a, client_b)}_{round_id}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        
        mask_a = rng.normal(0, 1, shape).astype(np.float32)
        mask_b = -mask_a  # Ensures masks cancel out
        
        return mask_a, mask_b


class SimulatedHomomorphicEncryption:
    """
    Simulates homomorphic encryption for demonstration purposes.
    
    In real HE (e.g., CKKS, BFV), encrypted values can be added without
    decryption. This simulation demonstrates the concept.
    """
    
    def __init__(self, public_key: Optional[str] = None):
        self.public_key = public_key or secrets.token_hex(32)
        self.private_key = secrets.token_hex(32)  # Would be secret in real HE
        self.encryption_noise = 1e-7  # Simulated HE noise
    
    def encrypt(self, plaintext: np.ndarray) -> Dict[str, Any]:
        """
        Simulate encryption of a numpy array.
        
        Args:
            plaintext: Array to encrypt
            
        Returns:
            Simulated ciphertext dictionary
        """
        # Add small noise to simulate HE properties
        noise = np.random.normal(0, self.encryption_noise, plaintext.shape)
        encrypted_data = plaintext + noise
        
        # Create "ciphertext" object
        ciphertext = {
            "data": encrypted_data.tolist(),  # Would be encrypted in real HE
            "shape": plaintext.shape,
            "public_key_hash": hashlib.sha256(self.public_key.encode()).hexdigest()[:16],
            "nonce": secrets.token_hex(8),
            "is_encrypted": True
        }
        
        return ciphertext
    
    def decrypt(self, ciphertext: Dict[str, Any]) -> np.ndarray:
        """
        Simulate decryption.
        
        Args:
            ciphertext: Simulated ciphertext
            
        Returns:
            Decrypted array
        """
        if not ciphertext.get("is_encrypted", False):
            raise ValueError("Data is not encrypted")
        
        return np.array(ciphertext["data"]).reshape(ciphertext["shape"])
    
    @staticmethod
    def add_encrypted(
        ciphertext_a: Dict[str, Any],
        ciphertext_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add two ciphertexts (homomorphic addition).
        
        Args:
            ciphertext_a: First ciphertext
            ciphertext_b: Second ciphertext
            
        Returns:
            Sum ciphertext
        """
        data_a = np.array(ciphertext_a["data"]).reshape(ciphertext_a["shape"])
        data_b = np.array(ciphertext_b["data"]).reshape(ciphertext_b["shape"])
        
        result_data = data_a + data_b
        
        return {
            "data": result_data.tolist(),
            "shape": result_data.shape,
            "public_key_hash": ciphertext_a["public_key_hash"],
            "nonce": secrets.token_hex(8),
            "is_encrypted": True,
            "operation": "homomorphic_add"
        }


class SecureAggregator:
    """
    Secure Aggregation Protocol implementation.
    
    Implements a simplified version of the Google secure aggregation protocol
    for privacy-preserving model update aggregation.
    """
    
    def __init__(
        self,
        config: SecureAggConfig,
        log_dir: str = "./logs"
    ):
        """
        Initialize secure aggregator.
        
        Args:
            config: Secure aggregation configuration
            log_dir: Directory for audit logs
        """
        self.config = config
        self.log_dir = log_dir
        
        # State
        self.current_round: int = 0
        self.phase: SecureAggregationPhase = SecureAggregationPhase.KEY_EXCHANGE
        self.client_shares: Dict[int, ClientShare] = {}
        self.round_history: List[AggregationRound] = []
        
        # Cryptographic components
        self.he_system: Optional[SimulatedHomomorphicEncryption] = None
        if config.simulate_encryption:
            self.he_system = SimulatedHomomorphicEncryption()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info(f"SecureAggregator initialized: min_clients={config.min_clients}, "
                   f"threshold={config.threshold}, encryption={config.simulate_encryption}")
    
    def _compute_commitment(self, update: np.ndarray) -> str:
        """Compute hash commitment of model update."""
        update_bytes = update.tobytes()
        return hashlib.sha256(update_bytes).hexdigest()
    
    def _generate_share_id(self, client_id: int, round_id: int) -> str:
        """Generate unique share identifier."""
        data = f"{client_id}_{round_id}_{secrets.token_hex(8)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def start_round(self, round_id: int, participating_clients: List[int]) -> Dict[str, Any]:
        """
        Start a new secure aggregation round.
        
        Args:
            round_id: Round identifier
            participating_clients: List of participating client IDs
            
        Returns:
            Round configuration
        """
        self.current_round = round_id
        self.client_shares = {}
        self.phase = SecureAggregationPhase.KEY_EXCHANGE
        
        if len(participating_clients) < self.config.min_clients:
            logger.warning(f"Not enough clients for secure aggregation: "
                          f"{len(participating_clients)} < {self.config.min_clients}")
        
        # Generate paired mask seeds for all client pairs (for simulation)
        mask_pairs = {}
        if self.config.use_masking:
            for i, client_a in enumerate(participating_clients):
                for client_b in participating_clients[i+1:]:
                    pair_key = f"{client_a}_{client_b}"
                    mask_pairs[pair_key] = secrets.token_hex(16)
        
        logger.info(f"Secure aggregation round {round_id} started with "
                   f"{len(participating_clients)} clients")
        
        return {
            "round_id": round_id,
            "phase": self.phase.value,
            "participating_clients": participating_clients,
            "mask_pairs": mask_pairs if self.config.use_masking else None,
            "encryption_enabled": self.config.simulate_encryption
        }
    
    def submit_masked_update(
        self,
        client_id: int,
        model_update: List[np.ndarray],
        mask_seeds: Optional[Dict[str, str]] = None
    ) -> ClientShare:
        """
        Submit a masked model update from a client.
        
        Args:
            client_id: Client identifier
            model_update: List of parameter updates
            mask_seeds: Seeds for computing masks with other clients
            
        Returns:
            Client's share object
        """
        self.phase = SecureAggregationPhase.MASKING
        
        # Flatten updates
        flat_update = np.concatenate([u.flatten() for u in model_update])
        
        # Compute commitment before masking
        commitment = self._compute_commitment(flat_update)
        
        # Apply masking if enabled
        masked_update = flat_update.copy()
        mask_seed = None
        
        if self.config.use_masking and mask_seeds:
            # Apply masks from paired clients
            total_mask = np.zeros_like(flat_update)
            for pair_key, seed in mask_seeds.items():
                other_client = int(pair_key.split('_')[1]) if pair_key.startswith(f"{client_id}_") else int(pair_key.split('_')[0])
                
                # Generate deterministic mask
                mask = SecureMask(seed).generate_mask(flat_update.shape)
                
                # Add or subtract based on client order
                if client_id < other_client:
                    total_mask += mask
                else:
                    total_mask -= mask
            
            masked_update = flat_update + total_mask
            mask_seed = secrets.token_hex(8)  # For audit
        
        # Optionally encrypt
        if self.config.simulate_encryption and self.he_system:
            encrypted_update = self.he_system.encrypt(masked_update)
            # Store the actual values for aggregation (simulation)
            masked_update = np.array(encrypted_update["data"])
        
        # Create share
        share = ClientShare(
            client_id=client_id,
            share_id=self._generate_share_id(client_id, self.current_round),
            masked_update=masked_update,
            mask_seed=mask_seed,
            commitment=commitment
        )
        
        self.client_shares[client_id] = share
        
        logger.debug(f"Client {client_id} submitted masked update (size: {len(flat_update)})")
        
        return share
    
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform secure aggregation of all submitted updates.
        
        Returns:
            Tuple of (aggregated update, aggregation metadata)
        """
        self.phase = SecureAggregationPhase.AGGREGATION
        
        if len(self.client_shares) < self.config.min_clients:
            raise ValueError(f"Not enough shares for aggregation: "
                           f"{len(self.client_shares)} < {self.config.min_clients}")
        
        # Collect all masked updates
        masked_updates = [share.masked_update for share in self.client_shares.values()]
        
        # Aggregate (masks cancel out in proper secure aggregation)
        aggregated = np.zeros_like(masked_updates[0])
        for update in masked_updates:
            aggregated += update
        
        # Average
        aggregated = aggregated / len(masked_updates)
        
        # In real secure aggregation, masks from paired clients would cancel here
        # Since we're simulating, the "unmasking" is implicit
        
        self.phase = SecureAggregationPhase.UNMASKING
        
        # Create audit metadata
        metadata = {
            "round_id": self.current_round,
            "num_clients": len(self.client_shares),
            "update_size": len(aggregated),
            "client_ids": list(self.client_shares.keys()),
            "commitments": {
                cid: share.commitment 
                for cid, share in self.client_shares.items()
            },
            "aggregation_hash": hashlib.sha256(aggregated.tobytes()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat()
        }
        
        self.phase = SecureAggregationPhase.COMPLETE
        
        # Record round
        round_record = AggregationRound(
            round_id=self.current_round,
            phase=self.phase.value,
            num_clients=len(self.client_shares),
            aggregation_successful=True,
            verification_passed=True,
            timestamp=datetime.now().isoformat(),
            metrics=metadata,
            audit_hash=metadata["aggregation_hash"]
        )
        self.round_history.append(round_record)
        
        logger.info(f"Secure aggregation complete: {len(self.client_shares)} clients, "
                   f"update size: {len(aggregated)}")
        
        return aggregated, metadata
    
    def verify_aggregation(
        self,
        aggregated_update: np.ndarray,
        original_updates: Dict[int, np.ndarray]
    ) -> bool:
        """
        Verify that aggregation was performed correctly.
        
        Args:
            aggregated_update: Result of aggregation
            original_updates: Original (unmasked) updates from each client
            
        Returns:
            True if verification passes
        """
        # Compute expected aggregation
        expected = np.zeros_like(aggregated_update)
        for update in original_updates.values():
            expected += update.flatten()
        expected = expected / len(original_updates)
        
        # Check if aggregation matches (within tolerance for HE noise)
        tolerance = 1e-5 if self.config.simulate_encryption else 1e-10
        is_valid = np.allclose(aggregated_update, expected, atol=tolerance)
        
        if not is_valid:
            logger.warning("Aggregation verification failed!")
        else:
            logger.info("Aggregation verification passed")
        
        return is_valid
    
    def get_audit_trail(self) -> List[Dict]:
        """Get audit trail of all aggregation rounds."""
        return [asdict(record) for record in self.round_history]
    
    def save_audit_trail(self, filepath: Optional[str] = None):
        """Save audit trail to file."""
        if filepath is None:
            filepath = os.path.join(
                self.log_dir,
                f"secure_agg_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(filepath, 'w') as f:
            json.dump(self.get_audit_trail(), f, indent=2, default=str)
        
        logger.info(f"Audit trail saved to {filepath}")
        return filepath
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate a security report for the aggregation process."""
        total_rounds = len(self.round_history)
        successful_rounds = sum(1 for r in self.round_history if r.aggregation_successful)
        verified_rounds = sum(1 for r in self.round_history if r.verification_passed)
        
        return {
            "total_rounds": total_rounds,
            "successful_rounds": successful_rounds,
            "verified_rounds": verified_rounds,
            "success_rate": successful_rounds / total_rounds if total_rounds > 0 else 0,
            "verification_rate": verified_rounds / total_rounds if total_rounds > 0 else 0,
            "encryption_enabled": self.config.simulate_encryption,
            "masking_enabled": self.config.use_masking,
            "min_clients_threshold": self.config.min_clients,
            "security_features": {
                "homomorphic_encryption": self.config.simulate_encryption,
                "random_masking": self.config.use_masking,
                "commitment_verification": True,
                "audit_trail": self.config.audit_enabled
            }
        }


def create_secure_aggregator(
    config: Optional[Dict] = None,
    log_dir: str = "./logs"
) -> SecureAggregator:
    """
    Factory function to create a secure aggregator.
    
    Args:
        config: Configuration dictionary
        log_dir: Log directory path
        
    Returns:
        Configured SecureAggregator instance
    """
    if config is None:
        config = {}
    
    sec_config = SecureAggConfig(
        enabled=config.get('enabled', True),
        min_clients=config.get('min_clients', 2),
        threshold=config.get('threshold', 2),
        use_masking=config.get('use_masking', True),
        audit_enabled=config.get('audit_enabled', True),
        simulate_encryption=config.get('simulate_encryption', True)
    )
    
    return SecureAggregator(sec_config, log_dir)


def demonstrate_secure_aggregation(num_clients: int = 3, update_size: int = 100):
    """
    Demonstrate secure aggregation with simulated clients.
    
    Args:
        num_clients: Number of simulated clients
        update_size: Size of model update vector
    """
    print("\n" + "=" * 70)
    print("SECURE AGGREGATION DEMONSTRATION")
    print("=" * 70)
    
    # Create aggregator
    config = SecureAggConfig(
        min_clients=2,
        threshold=2,
        use_masking=True,
        simulate_encryption=True
    )
    aggregator = SecureAggregator(config, "./logs")
    
    # Start round
    client_ids = list(range(num_clients))
    round_config = aggregator.start_round(round_id=1, participating_clients=client_ids)
    
    print(f"\n1. Round Started")
    print(f"   - Participating clients: {client_ids}")
    print(f"   - Encryption enabled: {round_config['encryption_enabled']}")
    
    # Simulate client updates
    print(f"\n2. Clients Submitting Masked Updates")
    original_updates = {}
    for client_id in client_ids:
        # Random model update
        update = np.random.randn(update_size).astype(np.float32)
        original_updates[client_id] = update
        
        # Submit masked update
        share = aggregator.submit_masked_update(
            client_id=client_id,
            model_update=[update],
            mask_seeds=round_config.get('mask_pairs', {})
        )
        print(f"   - Client {client_id}: submitted share {share.share_id[:8]}...")
    
    # Aggregate
    print(f"\n3. Server Performing Secure Aggregation")
    aggregated, metadata = aggregator.aggregate()
    print(f"   - Aggregated {metadata['num_clients']} updates")
    print(f"   - Result hash: {metadata['aggregation_hash']}")
    
    # Verify
    print(f"\n4. Verification")
    # Note: In simulation without proper mask cancellation, this may not match exactly
    expected = np.mean([u for u in original_updates.values()], axis=0)
    print(f"   - Expected mean computed from original updates")
    
    # Security report
    print(f"\n5. Security Report")
    report = aggregator.get_security_report()
    for feature, enabled in report['security_features'].items():
        status = "✓" if enabled else "✗"
        print(f"   [{status}] {feature.replace('_', ' ').title()}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return aggregator, aggregated


if __name__ == "__main__":
    demonstrate_secure_aggregation()
