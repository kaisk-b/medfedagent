"""
MedFedAgent Federated Learning Client

Flower-based federated learning client with differential privacy support.
Each client represents a hospital node in the federated learning system.
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitRes,
    EvaluateRes,
    GetParametersRes,
    Status,
    Code
)
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score, accuracy_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import create_model, get_model_size_mb
from privacy.dp_trainer import DPTrainer, PrivacyConfig


def get_parameters(model: nn.Module) -> NDArrays:
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays):
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class MedFedClient(fl.client.NumPyClient):
    """
    Federated Learning client for medical imaging.
    
    Implements:
    - Local training with DP-SGD
    - Model evaluation
    - Privacy budget tracking
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        privacy_config: Optional[PrivacyConfig] = None,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        grad_clip_norm: float = 0.0
    ):
        """
        Initialize the federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            privacy_config: Privacy configuration (None = no DP)
            local_epochs: Number of local epochs per round
            learning_rate: Learning rate
            grad_clip_norm: Max grad norm when DP is disabled (0 to disable)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.privacy_config = privacy_config
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        
        # Initialize DP trainer if privacy enabled
        self.dp_trainer: Optional[DPTrainer] = None
        self.optimizer = None
        
        # Calculate class weights from training data to handle imbalance
        # Default weights assuming ~65% class 0, ~35% class 1
        class_weights = torch.tensor([0.54, 1.0], dtype=torch.float32).to(device)
        try:
            labels = []
            for _, label in train_loader:
                labels.extend(label.numpy().tolist() if hasattr(label, 'numpy') else [label.item()])
            if labels:
                label_counts = {}
                for l in labels:
                    label_counts[l] = label_counts.get(l, 0) + 1
                total = len(labels)
                # Inverse frequency weighting
                if len(label_counts) >= 2:
                    weights = [total / (2.0 * label_counts.get(i, 1)) for i in range(2)]
                    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        except Exception:
            pass  # Use default weights
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics tracking
        self.round_metrics: List[Dict] = []
        self.gradient_norms: List[float] = []
        
        logger.info(f"Client {client_id} initialized with {len(train_loader.dataset)} "
                   f"training samples, DP={'enabled' if privacy_config and privacy_config.enabled else 'disabled'}")
    
    def _setup_training(self):
        """Set up optimizer and DP trainer."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        if self.privacy_config and self.privacy_config.enabled:
            self.dp_trainer = DPTrainer(
                model=self.model,
                optimizer=self.optimizer,
                train_loader=self.train_loader,
                config=self.privacy_config,
                device=self.device
            )
            self.model = self.dp_trainer.model
            self.optimizer = self.dp_trainer.optimizer
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters."""
        if self.dp_trainer is not None:
            model = self.dp_trainer.get_model()
        else:
            model = self.model
        return get_parameters(model)
    
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from server."""
        if self.dp_trainer is not None:
            model = self.dp_trainer.get_model()
            set_parameters(model, parameters)
        else:
            set_parameters(self.model, parameters)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Configuration from server
            
        Returns:
            Tuple of (updated parameters, num samples, metrics)
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Set up training if not done
        if self.optimizer is None:
            self._setup_training()
        
        # Get training config
        local_epochs = config.get("local_epochs", self.local_epochs)
        
        # Train locally
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(int(local_epochs)):
            if self.dp_trainer is not None:
                metrics = self.dp_trainer.train_epoch(
                    self.criterion,
                    max_physical_batch_size=64  # Increased for better CPU efficiency
                )
                total_loss += metrics["loss"]
                num_batches += 1
            else:
                # Non-private training
                epoch_loss = 0.0
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()

                    if self.grad_clip_norm and self.grad_clip_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                    # Track gradient norm
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    self.gradient_norms.append(total_norm)
                    
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss / len(self.train_loader)
                num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)
        
        # Get privacy metrics
        epsilon = 0.0
        if self.dp_trainer is not None:
            epsilon = self.dp_trainer.get_epsilon()
        
        # Store round metrics
        metrics = {
            "client_id": float(self.client_id),
            "train_loss": avg_loss,
            "epsilon": epsilon,
            "num_samples": len(self.train_loader.dataset)
        }
        self.round_metrics.append(metrics)
        
        logger.info(f"Client {self.client_id}: loss={avg_loss:.4f}, ε={epsilon:.4f}")
        
        return self.get_parameters({}), len(self.train_loader.dataset), metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration from server
            
        Returns:
            Tuple of (loss, num samples, metrics)
        """
        self.set_parameters(parameters)
        
        if self.dp_trainer is not None:
            model = self.dp_trainer.get_model()
        else:
            model = self.model
        
        model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * len(target)
                
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
                all_labels.extend(target.cpu().numpy())
        
        num_samples = len(self.val_loader.dataset)
        avg_loss = total_loss / num_samples
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            # AUC undefined if only one class present
            auc = 0.5
        
        metrics = {
            "client_id": float(self.client_id),
            "accuracy": accuracy,
            "auc": auc,
            "val_loss": avg_loss
        }
        
        logger.info(f"Client {self.client_id} eval: loss={avg_loss:.4f}, "
                   f"acc={accuracy:.4f}, AUC={auc:.4f}")
        
        return avg_loss, num_samples, metrics
    
    def set_learning_rate(self, new_lr: float):
        """Set optimizer learning rate (updates current optimizer)."""
        self.learning_rate = new_lr

        if self.optimizer is None:
            return

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def scale_learning_rate(self, factor: float, min_lr: float = 1e-4) -> float:
        """Scale learning rate by factor while respecting a minimum."""
        target_lr = max(self.learning_rate * factor, min_lr)
        self.set_learning_rate(target_lr)
        logger.debug(
            f"Client {self.client_id} learning rate adjusted to {target_lr:.6f} (factor={factor})"
        )
        return target_lr

    def get_gradient_stats(self) -> Dict[str, float]:
        """Get statistics about gradient norms."""
        if not self.gradient_norms:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        
        norms = np.array(self.gradient_norms)
        return {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "max": float(np.max(norms)),
            "min": float(np.min(norms))
        }


def create_flower_client(
    client_id: int,
    model_name: str,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    enable_dp: bool = True,
    epsilon_budget: float = 8.0,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    local_epochs: int = 1,
    learning_rate: float = 0.001
) -> MedFedClient:
    """
    Factory function to create a Flower client.
    
    Args:
        client_id: Client identifier
        model_name: Model architecture name
        num_classes: Number of output classes
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        enable_dp: Whether to enable differential privacy
        epsilon_budget: Total privacy budget
        noise_multiplier: DP noise scale
        max_grad_norm: Gradient clipping norm
        local_epochs: Local epochs per round
        learning_rate: Learning rate
        
    Returns:
        Configured MedFedClient
    """
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.3
    )
    
    # Create privacy config
    privacy_config = None
    if enable_dp:
        privacy_config = PrivacyConfig(
            enabled=True,
            epsilon_budget=epsilon_budget,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )
    
    return MedFedClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        privacy_config=privacy_config,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )


# Client function for Flower simulation
def client_fn(cid: str) -> fl.client.Client:
    """
    Client function for Flower simulation.
    
    This function is called by Flower to create client instances.
    In practice, you'd pass the actual data loaders.
    """
    # This is a placeholder - actual implementation will use loaded data
    raise NotImplementedError("Use create_flower_client directly or set up simulation properly")


if __name__ == "__main__":
    # Test the client
    print("Testing FL client...")
    
    # Create dummy data
    X_train = torch.randn(100, 3, 224, 224)
    y_train = torch.randint(0, 2, (100,))
    X_val = torch.randn(20, 3, 224, 224)
    y_val = torch.randint(0, 2, (20,))
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create client with simple model for testing
    from models.model import SimpleCNN
    model = SimpleCNN(num_classes=2)
    
    client = MedFedClient(
        client_id=0,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        privacy_config=None,  # No DP for quick test
        local_epochs=1
    )
    
    # Test get parameters
    params = client.get_parameters({})
    print(f"Number of parameter arrays: {len(params)}")
    
    # Test fit
    updated_params, num_samples, metrics = client.fit(params, {"local_epochs": 1})
    print(f"Fit metrics: {metrics}")
    
    # Test evaluate  
    loss, num_samples, eval_metrics = client.evaluate(updated_params, {})
    print(f"Eval metrics: {eval_metrics}")
