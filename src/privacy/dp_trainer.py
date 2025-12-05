"""
MedFedAgent Differential Privacy Module

Implements DP-SGD (Differentially Private Stochastic Gradient Descent) using Opacus.
Provides privacy budget tracking and noise calibration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants import RDPAccountant
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger
import numpy as np


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    enabled: bool = True
    epsilon_budget: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    secure_mode: bool = False
    
    def __post_init__(self):
        if self.epsilon_budget <= 0:
            raise ValueError("epsilon_budget must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")


@dataclass
class PrivacyMetrics:
    """Metrics from privacy engine after training."""
    epsilon_spent: float = 0.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    num_steps: int = 0
    sample_rate: float = 0.0
    budget_remaining: float = 8.0
    budget_percentage_used: float = 0.0


class DPTrainer:
    """
    Differentially Private Trainer using Opacus.
    
    Wraps a PyTorch model and optimizer with DP-SGD capabilities.
    Tracks privacy budget and provides mechanisms to adjust noise.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        config: PrivacyConfig,
        device: str = "cpu"
    ):
        """
        Initialize the DP trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer (will be wrapped)
            train_loader: Training data loader
            config: Privacy configuration
            device: Device to use
        """
        self.config = config
        self.device = device
        self.original_model = model
        self.original_optimizer = optimizer
        self.train_loader = train_loader
        
        # Track total privacy spent
        self.total_epsilon_spent = 0.0
        self.total_steps = 0
        self.round_history = []
        
        # Privacy engine (will be created when make_private is called)
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.model = model
        self.optimizer = optimizer
        
        if config.enabled:
            self._setup_privacy()
    
    def _setup_privacy(self):
        """Set up differential privacy with Opacus."""
        logger.info("Setting up differential privacy...")
        
        # Validate model is compatible with Opacus
        self.model = ModuleValidator.fix(self.original_model)
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            logger.warning(f"Model validation warnings: {errors}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Recreate optimizer for the fixed model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.original_optimizer.param_groups[0].get('lr', 0.001)
        )
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine(
            secure_mode=self.config.secure_mode
        )
        
        # Get batch size before make_private (it may become None after)
        original_batch_size = getattr(self.train_loader, 'batch_size', 32)
        original_dataset_size = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        
        # Make model and optimizer private
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.config.noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Calculate sample rate using the values captured before make_private
        batch_size = original_batch_size if original_batch_size is not None else 32
        dataset_size = original_dataset_size if original_dataset_size > 0 else 1000
        self.sample_rate = batch_size / dataset_size
        
        logger.info(f"DP-SGD enabled: noise_multiplier={self.config.noise_multiplier}, "
                   f"max_grad_norm={self.config.max_grad_norm}")
    
    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Get current epsilon spent.
        
        Args:
            delta: Delta value (uses config default if None)
            
        Returns:
            Epsilon value
        """
        if not self.config.enabled or self.privacy_engine is None:
            return 0.0
        
        delta = delta or self.config.delta
        try:
            epsilon = self.privacy_engine.get_epsilon(delta=delta)
            return epsilon
        except Exception as e:
            logger.warning(f"Could not compute epsilon: {e}")
            return self.total_epsilon_spent
    
    def get_privacy_metrics(self) -> PrivacyMetrics:
        """Get current privacy metrics."""
        epsilon = self.get_epsilon()
        
        return PrivacyMetrics(
            epsilon_spent=epsilon,
            delta=self.config.delta,
            noise_multiplier=self.config.noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
            num_steps=self.total_steps,
            sample_rate=self.sample_rate if hasattr(self, 'sample_rate') else 0.0,
            budget_remaining=max(0, self.config.epsilon_budget - epsilon),
            budget_percentage_used=(epsilon / self.config.epsilon_budget) * 100
        )
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> Tuple[float, torch.Tensor]:
        """
        Perform one training step with DP-SGD.
        
        Args:
            batch: Tuple of (inputs, labels)
            criterion: Loss function
            
        Returns:
            Tuple of (loss value, predictions)
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        
        self.total_steps += 1
        
        return loss.item(), outputs.detach()
    
    def train_epoch(
        self,
        criterion: nn.Module,
        max_physical_batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train for one epoch with DP.
        
        Args:
            criterion: Loss function
            max_physical_batch_size: Max batch size for memory management
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        if self.config.enabled and self.privacy_engine is not None:
            # Use BatchMemoryManager for efficient batch processing
            with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=self.optimizer
            ) as memory_safe_loader:
                for batch in memory_safe_loader:
                    loss, outputs = self.train_step(batch, criterion)
                    total_loss += loss
                    num_batches += 1
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    labels = batch[1].to(self.device)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
        else:
            # Non-private training
            for batch in self.train_loader:
                loss, outputs = self.train_step(batch, criterion)
                total_loss += loss
                num_batches += 1
                
                _, predicted = outputs.max(1)
                labels = batch[1].to(self.device)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        epsilon = self.get_epsilon()
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "epsilon": epsilon,
            "steps": self.total_steps
        }
    
    def update_noise_multiplier(self, new_multiplier: float):
        """
        Update the noise multiplier.
        
        Note: This requires recreating the privacy engine in Opacus.
        For simplicity, we just update the config for future reference.
        
        Args:
            new_multiplier: New noise multiplier value
        """
        logger.info(f"Updating noise multiplier: {self.config.noise_multiplier} -> {new_multiplier}")
        self.config.noise_multiplier = new_multiplier
        # In practice, you'd need to recreate the privacy engine
        # This is a limitation of Opacus that we acknowledge
    
    def check_budget(self) -> Tuple[bool, str]:
        """
        Check if we're within privacy budget.
        
        Returns:
            Tuple of (is_within_budget, status_message)
        """
        epsilon = self.get_epsilon()
        budget = self.config.epsilon_budget
        
        if epsilon >= budget:
            return False, f"HALT: Budget exhausted (ε={epsilon:.2f} >= {budget})"
        elif epsilon >= 0.9 * budget:
            return True, f"CRITICAL: Budget at 90% (ε={epsilon:.2f}/{budget})"
        elif epsilon >= 0.75 * budget:
            return True, f"WARNING: Budget at 75% (ε={epsilon:.2f}/{budget})"
        else:
            return True, f"OK: Budget at {(epsilon/budget)*100:.1f}% (ε={epsilon:.2f}/{budget})"
    
    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrapped from Opacus GradSampleModule)."""
        # In newer Opacus, the model is wrapped in GradSampleModule
        # We need to unwrap it to get the original model
        if hasattr(self.model, '_module'):
            # GradSampleModule wraps the original module
            return self.model._module
        return self.model
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        model = self.get_model()
        torch.save({
            'model_state_dict': model.state_dict(),
            'total_epsilon': self.get_epsilon(),
            'total_steps': self.total_steps,
            'config': self.config
        }, path)
        logger.info(f"Saved checkpoint to {path}")


def compute_dp_sgd_privacy(
    sample_rate: float,
    noise_multiplier: float,
    num_steps: int,
    delta: float = 1e-5
) -> float:
    """
    Compute epsilon for DP-SGD using RDP accounting.
    
    This is a simplified version - Opacus does this automatically.
    
    Args:
        sample_rate: Batch size / dataset size
        noise_multiplier: Noise scale
        num_steps: Number of training steps
        delta: Target delta
        
    Returns:
        Epsilon value
    """
    from opacus.accountants.analysis import rdp as compute_rdp
    from opacus.accountants.analysis import get_privacy_spent
    
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    
    rdp = compute_rdp.compute_rdp(
        q=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=num_steps,
        orders=orders
    )
    
    eps, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    
    return eps


def validate_model_for_dp(model: nn.Module) -> Tuple[bool, list]:
    """
    Check if a model is compatible with Opacus DP-SGD.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = ModuleValidator.validate(model, strict=False)
    return len(errors) == 0, errors


def fix_model_for_dp(model: nn.Module) -> nn.Module:
    """
    Fix a model to be compatible with Opacus DP-SGD.
    
    Common fixes:
    - Replace BatchNorm with GroupNorm
    - Handle incompatible layers
    
    Args:
        model: PyTorch model
        
    Returns:
        Fixed model
    """
    return ModuleValidator.fix(model)


if __name__ == "__main__":
    # Test the privacy module
    print("Testing privacy module...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )
    
    # Create dummy data
    X = torch.randn(100, 100)
    y = torch.randint(0, 2, (100,))
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create DP trainer
    config = PrivacyConfig(
        enabled=True,
        epsilon_budget=8.0,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    )
    
    trainer = DPTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        config=config
    )
    
    # Train for a few steps
    criterion = nn.CrossEntropyLoss()
    metrics = trainer.train_epoch(criterion)
    
    print(f"Training metrics: {metrics}")
    print(f"Privacy metrics: {trainer.get_privacy_metrics()}")
    print(f"Budget check: {trainer.check_budget()}")
