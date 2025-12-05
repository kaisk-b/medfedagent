"""
MedFedAgent Membership Inference Attack Module

Implements membership inference attacks to validate differential privacy protection.
This module tests whether an attacker can determine if a specific sample was used
in training the model.

References:
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017)
- Yeom et al., "Privacy Risk in Machine Learning" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from loguru import logger
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import copy


@dataclass
class MIAConfig:
    """Configuration for membership inference attack."""
    attack_type: str = "threshold"  # "threshold", "shadow", "loss_based"
    num_shadow_models: int = 5
    shadow_train_ratio: float = 0.5
    attack_epochs: int = 10
    attack_learning_rate: float = 0.001
    num_classes: int = 2
    threshold_percentile: float = 50.0  # For threshold attack


@dataclass
class MIAResult:
    """Results from membership inference attack."""
    attack_type: str
    accuracy: float
    precision: float
    recall: float
    auc: float
    advantage: float  # Attacker's advantage over random guessing
    vulnerability_score: float  # 0-1 score (0 = well protected, 1 = vulnerable)
    num_members: int
    num_non_members: int
    threshold: Optional[float] = None
    per_class_results: Dict[int, Dict] = field(default_factory=dict)
    
    def is_vulnerable(self, threshold: float = 0.6) -> bool:
        """Check if model is vulnerable to MIA."""
        return self.auc > threshold
    
    def get_privacy_grade(self) -> str:
        """Get a privacy grade based on MIA results."""
        if self.auc <= 0.55:
            return "A (Excellent)"
        elif self.auc <= 0.60:
            return "B (Good)"
        elif self.auc <= 0.70:
            return "C (Fair)"
        elif self.auc <= 0.80:
            return "D (Poor)"
        else:
            return "F (Vulnerable)"


class AttackModel(nn.Module):
    """Neural network attack model for shadow model-based MIA."""
    
    def __init__(self, num_classes: int, hidden_size: int = 64):
        super().__init__()
        # Input: prediction probabilities + loss + max prob + entropy
        input_size = num_classes + 3
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 2)  # Binary: member or not
        )
    
    def forward(self, x):
        return self.network(x)


class MembershipInferenceAttack:
    """
    Membership Inference Attack implementation.
    
    Supports three attack strategies:
    1. Threshold-based: Uses loss threshold to classify members
    2. Shadow model-based: Trains attack model on shadow model outputs
    3. Loss-based: More sophisticated loss-based attack
    """
    
    def __init__(
        self,
        config: MIAConfig,
        target_model: nn.Module,
        device: str = "cpu"
    ):
        """
        Initialize the MIA attack.
        
        Args:
            config: Attack configuration
            target_model: The model to attack
            device: Device for computation
        """
        self.config = config
        self.target_model = target_model
        self.device = device
        
        # Attack components
        self.attack_model: Optional[AttackModel] = None
        self.shadow_models: List[nn.Module] = []
        self.threshold: Optional[float] = None
        
        logger.info(f"Initialized MIA attack: type={config.attack_type}")
    
    def _compute_loss_per_sample(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-sample loss for all samples in the data loader.
        
        Returns:
            Tuple of (losses, predictions, labels)
        """
        model.eval()
        losses = []
        all_preds = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                
                # Per-sample loss
                sample_losses = criterion(outputs, labels)
                losses.extend(sample_losses.cpu().numpy())
                
                # Predictions
                probs = F.softmax(outputs, dim=1)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(losses), np.array(all_preds), np.array(all_labels)
    
    def _compute_attack_features(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute features for the attack model.
        
        Features include:
        - Prediction probabilities
        - Loss value
        - Maximum probability
        - Entropy of predictions
        
        Returns:
            Tuple of (features, labels)
        """
        model.eval()
        features = []
        labels = []
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                
                # Compute probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                # Compute loss
                loss = criterion(outputs, target).cpu().numpy()
                
                # Compute max probability
                max_prob = np.max(probs, axis=1)
                
                # Compute entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                
                # Combine features
                for i in range(len(data)):
                    feature_vec = np.concatenate([
                        probs[i],
                        [loss[i], max_prob[i], entropy[i]]
                    ])
                    features.append(feature_vec)
                
                labels.extend(target.cpu().numpy())
        
        return np.array(features), np.array(labels)
    
    def threshold_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader
    ) -> MIAResult:
        """
        Perform threshold-based membership inference attack.
        
        Uses the intuition that training samples have lower loss.
        
        Args:
            member_loader: DataLoader with training samples (members)
            non_member_loader: DataLoader with non-training samples
            
        Returns:
            Attack results
        """
        logger.info("Running threshold-based MIA...")
        
        # Compute losses for members and non-members
        member_losses, _, member_labels = self._compute_loss_per_sample(
            self.target_model, member_loader
        )
        non_member_losses, _, non_member_labels = self._compute_loss_per_sample(
            self.target_model, non_member_loader
        )
        
        # Find optimal threshold
        all_losses = np.concatenate([member_losses, non_member_losses])
        true_membership = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(non_member_losses))
        ])
        
        # Try percentile-based threshold
        self.threshold = np.percentile(all_losses, self.config.threshold_percentile)
        
        # Predict: if loss < threshold, predict member (1)
        predictions = (all_losses < self.threshold).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(true_membership, predictions)
        
        # AUC (use negative loss since lower loss = more likely member)
        try:
            auc = roc_auc_score(true_membership, -all_losses)
        except ValueError:
            auc = 0.5
        
        # Precision and recall
        true_positives = np.sum((predictions == 1) & (true_membership == 1))
        false_positives = np.sum((predictions == 1) & (true_membership == 0))
        false_negatives = np.sum((predictions == 0) & (true_membership == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        # Advantage over random guessing (accuracy - 0.5) * 2
        advantage = (accuracy - 0.5) * 2
        
        # Vulnerability score (how much worse than perfect privacy)
        vulnerability_score = max(0, auc - 0.5) * 2
        
        result = MIAResult(
            attack_type="threshold",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc=auc,
            advantage=advantage,
            vulnerability_score=vulnerability_score,
            num_members=len(member_losses),
            num_non_members=len(non_member_losses),
            threshold=self.threshold
        )
        
        logger.info(f"Threshold attack results: accuracy={accuracy:.4f}, AUC={auc:.4f}")
        
        return result
    
    def loss_based_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader
    ) -> MIAResult:
        """
        Perform loss-based membership inference attack.
        
        More sophisticated than threshold attack - uses per-class thresholds.
        
        Args:
            member_loader: DataLoader with training samples
            non_member_loader: DataLoader with non-training samples
            
        Returns:
            Attack results
        """
        logger.info("Running loss-based MIA...")
        
        # Compute losses and labels
        member_losses, _, member_labels = self._compute_loss_per_sample(
            self.target_model, member_loader
        )
        non_member_losses, _, non_member_labels = self._compute_loss_per_sample(
            self.target_model, non_member_loader
        )
        
        # Per-class thresholds
        num_classes = self.config.num_classes
        class_thresholds = {}
        
        for c in range(num_classes):
            member_mask = member_labels == c
            non_member_mask = non_member_labels == c
            
            if np.sum(member_mask) > 0 and np.sum(non_member_mask) > 0:
                # Use mean of member losses as threshold
                class_thresholds[c] = np.mean(member_losses[member_mask])
            else:
                class_thresholds[c] = np.mean(member_losses)
        
        # Make predictions
        all_losses = np.concatenate([member_losses, non_member_losses])
        all_labels = np.concatenate([member_labels, non_member_labels])
        true_membership = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(non_member_losses))
        ])
        
        predictions = np.zeros(len(all_losses))
        for i, (loss, label) in enumerate(zip(all_losses, all_labels)):
            if int(label) in class_thresholds:
                predictions[i] = 1 if loss < class_thresholds[int(label)] else 0
            else:
                predictions[i] = 0
        
        # Compute metrics
        accuracy = accuracy_score(true_membership, predictions)
        
        try:
            auc = roc_auc_score(true_membership, -all_losses)
        except ValueError:
            auc = 0.5
        
        true_positives = np.sum((predictions == 1) & (true_membership == 1))
        false_positives = np.sum((predictions == 1) & (true_membership == 0))
        false_negatives = np.sum((predictions == 0) & (true_membership == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        advantage = (accuracy - 0.5) * 2
        vulnerability_score = max(0, auc - 0.5) * 2
        
        # Per-class results
        per_class_results = {}
        for c in range(num_classes):
            class_mask = all_labels == c
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(
                    true_membership[class_mask], 
                    predictions[class_mask]
                )
                per_class_results[c] = {
                    "accuracy": class_acc,
                    "threshold": class_thresholds.get(c, 0)
                }
        
        result = MIAResult(
            attack_type="loss_based",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc=auc,
            advantage=advantage,
            vulnerability_score=vulnerability_score,
            num_members=len(member_losses),
            num_non_members=len(non_member_losses),
            per_class_results=per_class_results
        )
        
        logger.info(f"Loss-based attack results: accuracy={accuracy:.4f}, AUC={auc:.4f}")
        
        return result
    
    def shadow_model_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        model_class: type,
        model_kwargs: Dict[str, Any]
    ) -> MIAResult:
        """
        Perform shadow model-based membership inference attack.
        
        Trains shadow models to mimic target model behavior, then trains
        an attack model to distinguish members from non-members.
        
        Args:
            member_loader: DataLoader with training samples
            non_member_loader: DataLoader with non-training samples
            model_class: Class to instantiate shadow models
            model_kwargs: Arguments for model class
            
        Returns:
            Attack results
        """
        logger.info(f"Running shadow model MIA with {self.config.num_shadow_models} shadow models...")
        
        # Combine data for shadow model training
        all_data = []
        all_labels = []
        
        for data, labels in member_loader:
            all_data.append(data)
            all_labels.append(labels)
        for data, labels in non_member_loader:
            all_data.append(data)
            all_labels.append(labels)
        
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Create attack training data
        attack_features = []
        attack_labels = []  # 1 = member, 0 = non-member
        
        dataset_size = len(all_data)
        train_size = int(dataset_size * self.config.shadow_train_ratio)
        
        for shadow_idx in range(self.config.num_shadow_models):
            logger.info(f"Training shadow model {shadow_idx + 1}/{self.config.num_shadow_models}")
            
            # Random split for this shadow model
            indices = np.random.permutation(dataset_size)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            # Create shadow model
            shadow_model = model_class(**model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Train shadow model
            train_dataset = TensorDataset(
                all_data[train_indices],
                all_labels[train_indices]
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            shadow_model.train()
            for epoch in range(self.config.attack_epochs):
                for data, labels in train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = shadow_model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Collect attack features from shadow model
            shadow_model.eval()
            with torch.no_grad():
                # Training samples (members)
                for idx in train_indices:
                    data = all_data[idx:idx+1].to(self.device)
                    label = all_labels[idx:idx+1].to(self.device)
                    
                    output = shadow_model(data)
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    loss = F.cross_entropy(output, label).item()
                    max_prob = np.max(probs)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    
                    feature_vec = np.concatenate([probs, [loss, max_prob, entropy]])
                    attack_features.append(feature_vec)
                    attack_labels.append(1)  # Member
                
                # Non-training samples (non-members)
                for idx in test_indices:
                    data = all_data[idx:idx+1].to(self.device)
                    label = all_labels[idx:idx+1].to(self.device)
                    
                    output = shadow_model(data)
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    loss = F.cross_entropy(output, label).item()
                    max_prob = np.max(probs)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    
                    feature_vec = np.concatenate([probs, [loss, max_prob, entropy]])
                    attack_features.append(feature_vec)
                    attack_labels.append(0)  # Non-member
        
        # Train attack model
        logger.info("Training attack model...")
        attack_features = np.array(attack_features)
        attack_labels = np.array(attack_labels)
        
        attack_dataset = TensorDataset(
            torch.FloatTensor(attack_features),
            torch.LongTensor(attack_labels)
        )
        attack_loader = DataLoader(attack_dataset, batch_size=32, shuffle=True)
        
        self.attack_model = AttackModel(
            num_classes=self.config.num_classes
        ).to(self.device)
        
        attack_optimizer = torch.optim.Adam(
            self.attack_model.parameters(),
            lr=self.config.attack_learning_rate
        )
        attack_criterion = nn.CrossEntropyLoss()
        
        self.attack_model.train()
        for epoch in range(self.config.attack_epochs):
            for features, labels in attack_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                attack_optimizer.zero_grad()
                outputs = self.attack_model(features)
                loss = attack_criterion(outputs, labels)
                loss.backward()
                attack_optimizer.step()
        
        # Evaluate on target model
        logger.info("Evaluating attack on target model...")
        
        target_member_features, _ = self._compute_attack_features(
            self.target_model, member_loader
        )
        target_non_member_features, _ = self._compute_attack_features(
            self.target_model, non_member_loader
        )
        
        all_features = np.concatenate([target_member_features, target_non_member_features])
        true_membership = np.concatenate([
            np.ones(len(target_member_features)),
            np.zeros(len(target_non_member_features))
        ])
        
        # Attack model predictions
        self.attack_model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(all_features).to(self.device)
            attack_outputs = self.attack_model(features_tensor)
            attack_probs = F.softmax(attack_outputs, dim=1).cpu().numpy()
            predictions = np.argmax(attack_probs, axis=1)
            member_probs = attack_probs[:, 1]  # Probability of being a member
        
        # Compute metrics
        accuracy = accuracy_score(true_membership, predictions)
        
        try:
            auc = roc_auc_score(true_membership, member_probs)
        except ValueError:
            auc = 0.5
        
        true_positives = np.sum((predictions == 1) & (true_membership == 1))
        false_positives = np.sum((predictions == 1) & (true_membership == 0))
        false_negatives = np.sum((predictions == 0) & (true_membership == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        advantage = (accuracy - 0.5) * 2
        vulnerability_score = max(0, auc - 0.5) * 2
        
        result = MIAResult(
            attack_type="shadow_model",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc=auc,
            advantage=advantage,
            vulnerability_score=vulnerability_score,
            num_members=len(target_member_features),
            num_non_members=len(target_non_member_features)
        )
        
        logger.info(f"Shadow model attack results: accuracy={accuracy:.4f}, AUC={auc:.4f}")
        
        return result
    
    def run_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        model_class: Optional[type] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> MIAResult:
        """
        Run the configured membership inference attack.
        
        Args:
            member_loader: DataLoader with training samples
            non_member_loader: DataLoader with non-training samples
            model_class: For shadow attack, the model class
            model_kwargs: For shadow attack, model arguments
            
        Returns:
            Attack results
        """
        if self.config.attack_type == "threshold":
            return self.threshold_attack(member_loader, non_member_loader)
        elif self.config.attack_type == "loss_based":
            return self.loss_based_attack(member_loader, non_member_loader)
        elif self.config.attack_type == "shadow":
            if model_class is None or model_kwargs is None:
                raise ValueError("Shadow attack requires model_class and model_kwargs")
            return self.shadow_model_attack(
                member_loader, non_member_loader, model_class, model_kwargs
            )
        else:
            raise ValueError(f"Unknown attack type: {self.config.attack_type}")


def run_privacy_audit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    attack_types: List[str] = None
) -> Dict[str, MIAResult]:
    """
    Run a comprehensive privacy audit using multiple MIA attacks.
    
    Args:
        model: Target model to audit
        train_loader: Training data (members)
        test_loader: Test data (non-members)
        device: Device for computation
        attack_types: List of attack types to run
        
    Returns:
        Dictionary of attack type to results
    """
    if attack_types is None:
        attack_types = ["threshold", "loss_based"]
    
    results = {}
    
    for attack_type in attack_types:
        config = MIAConfig(attack_type=attack_type)
        attack = MembershipInferenceAttack(config, model, device)
        
        try:
            result = attack.run_attack(train_loader, test_loader)
            results[attack_type] = result
        except Exception as e:
            logger.error(f"Attack {attack_type} failed: {e}")
            continue
    
    return results


def print_privacy_report(results: Dict[str, MIAResult]):
    """Print a formatted privacy audit report."""
    print("\n" + "=" * 70)
    print("PRIVACY AUDIT REPORT - Membership Inference Attack Results")
    print("=" * 70)
    
    for attack_type, result in results.items():
        print(f"\n--- {attack_type.upper()} Attack ---")
        print(f"  Accuracy:           {result.accuracy:.4f}")
        print(f"  AUC-ROC:            {result.auc:.4f}")
        print(f"  Precision:          {result.precision:.4f}")
        print(f"  Recall:             {result.recall:.4f}")
        print(f"  Advantage:          {result.advantage:.4f}")
        print(f"  Vulnerability:      {result.vulnerability_score:.4f}")
        print(f"  Privacy Grade:      {result.get_privacy_grade()}")
        print(f"  Is Vulnerable:      {'Yes' if result.is_vulnerable() else 'No'}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE:")
    print("  - AUC ≈ 0.5: Model is well protected (attacker no better than random)")
    print("  - AUC > 0.6: Model may be vulnerable to membership inference")
    print("  - AUC > 0.8: Model is highly vulnerable")
    print("  - DP training should reduce AUC towards 0.5")
    print("=" * 70)
