"""Privacy module for MedFedAgent."""

from .dp_trainer import (
    PrivacyConfig,
    PrivacyMetrics,
    DPTrainer,
    compute_dp_sgd_privacy,
    validate_model_for_dp,
    fix_model_for_dp
)

from .mia_attack import (
    MIAConfig,
    MIAResult,
    MembershipInferenceAttack,
    run_privacy_audit,
    print_privacy_report
)

from .secure_aggregation import (
    SecureAggConfig,
    SecureAggregator,
    SecureAggregationPhase,
    SimulatedHomomorphicEncryption,
    create_secure_aggregator,
    demonstrate_secure_aggregation
)

__all__ = [
    # DP Trainer
    "PrivacyConfig",
    "PrivacyMetrics",
    "DPTrainer",
    "compute_dp_sgd_privacy",
    "validate_model_for_dp",
    "fix_model_for_dp",
    # Membership Inference Attack
    "MIAConfig",
    "MIAResult",
    "MembershipInferenceAttack",
    "run_privacy_audit",
    "print_privacy_report",
    # Secure Aggregation
    "SecureAggConfig",
    "SecureAggregator",
    "SecureAggregationPhase",
    "SimulatedHomomorphicEncryption",
    "create_secure_aggregator",
    "demonstrate_secure_aggregation"
]
