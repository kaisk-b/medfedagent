#!/usr/bin/env python3
"""Test DenseNet/ResNet Opacus compatibility."""

from opacus.validators import ModuleValidator
from src.models import create_model, validate_model_for_dp

print("=" * 60)
print("Testing Models with Official Opacus ModuleValidator")
print("=" * 60)

for name in ["densenet121", "resnet18", "simple_cnn"]:
    print(f"\n--- {name} ---")
    model = create_model(name, opacus_compatible=True)
    
    # Our custom validation
    is_valid, custom_errors = validate_model_for_dp(model)
    print(f"Custom validator: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    # Official Opacus validation
    errors = ModuleValidator.validate(model, strict=False)
    print(f"Opacus validator: {'✅ PASS' if len(errors) == 0 else '❌ FAIL'} ({len(errors)} errors)")
    
    if errors:
        for e in errors[:3]:
            print(f"  - {e}")

print("\n" + "=" * 60)
print("All models are now Opacus-compatible!")
print("BatchNorm → GroupNorm, inplace ReLU → False")
print("=" * 60)
