"""
Model Registry - Versioned Model Management
Tracks, saves, and loads versioned sklearn/joblib models.
Applies to: domain_classifier, engine_selector (NOT to SemanticIntentClassifier
which uses pre-trained MiniLM embeddings that never change).
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Registry lives in training/models/
MODELS_DIR = Path(__file__).parent.parent / "training" / "models"
REGISTRY_FILE = MODELS_DIR / "model_registry.json"


def _load_registry() -> Dict[str, Any]:
    """Load the registry JSON, creating it if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"models": {}, "version_history": []}


def _save_registry(registry: Dict[str, Any]) -> None:
    """Persist the registry to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def save_model(model: Any, model_name: str, metadata: Optional[Dict] = None) -> str:
    """
    Save a model with version tracking.

    Creates a timestamped archive copy and updates the registry
    so the canonical path (e.g. domain_classifier.joblib) always
    points to the latest model.

    Args:
        model:       Trained sklearn / any joblib-serialisable object.
        model_name:  Logical name, e.g. "domain_classifier", "engine_selector".
        metadata:    Optional dict of metrics / notes to store in the registry.

    Returns:
        Path to the versioned archive file.
    """
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required for model_registry.save_model()")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = _load_registry()

    # Determine next version number
    history_for_model = [
        e for e in registry["version_history"] if e["model_name"] == model_name
    ]
    version_num = len(history_for_model) + 1

    # Build versioned filename
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    versioned_filename = f"{model_name}_v{version_num}_{ts}.joblib"
    versioned_path = MODELS_DIR / versioned_filename

    # Save versioned copy
    joblib.dump(model, versioned_path)

    # Save / overwrite the canonical path
    canonical_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, canonical_path)

    # Update registry
    entry = {
        "model_name": model_name,
        "version": version_num,
        "timestamp": datetime.now().isoformat(),
        "versioned_file": versioned_filename,
        "canonical_file": f"{model_name}.joblib",
        "metadata": metadata or {},
    }
    registry["version_history"].append(entry)
    registry["models"][model_name] = entry  # latest pointer

    _save_registry(registry)

    print(f"✓ Saved {model_name} v{version_num} → {versioned_path}")
    return str(versioned_path)


def load_model(model_name: str, version: Optional[int] = None) -> Any:
    """
    Load a model by name, optionally pinning to a specific version.

    Args:
        model_name: Logical name, e.g. "domain_classifier".
        version:    If None, loads the latest canonical file.

    Returns:
        Loaded model object.
    """
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required for model_registry.load_model()")

    if version is None:
        canonical_path = MODELS_DIR / f"{model_name}.joblib"
        if not canonical_path.exists():
            raise FileNotFoundError(
                f"No canonical model found at {canonical_path}. "
                "Train the model first."
            )
        model = joblib.load(canonical_path)
        print(f"✓ Loaded {model_name} (latest) from {canonical_path}")
        return model

    # Load specific version
    registry = _load_registry()
    history = [
        e for e in registry["version_history"] if e["model_name"] == model_name
    ]
    for entry in history:
        if entry["version"] == version:
            versioned_path = MODELS_DIR / entry["versioned_file"]
            if not versioned_path.exists():
                raise FileNotFoundError(
                    f"Versioned file not found: {versioned_path}"
                )
            model = joblib.load(versioned_path)
            print(f"✓ Loaded {model_name} v{version} from {versioned_path}")
            return model

    raise ValueError(
        f"Version {version} of '{model_name}' not found in registry."
    )


def list_versions(model_name: Optional[str] = None) -> List[Dict]:
    """
    List all registered model versions.

    Args:
        model_name: Filter by model name, or None for all models.

    Returns:
        List of version entry dicts.
    """
    registry = _load_registry()
    history = registry.get("version_history", [])
    if model_name:
        history = [e for e in history if e["model_name"] == model_name]
    return history


def get_latest_version_info(model_name: str) -> Optional[Dict]:
    """Return the registry entry for the latest version of a model."""
    registry = _load_registry()
    return registry["models"].get(model_name)


def rollback(model_name: str, version: int) -> bool:
    """
    Roll back the canonical model file to a specific older version.

    Args:
        model_name: Logical name.
        version:    Version number to restore.

    Returns:
        True on success.
    """
    registry = _load_registry()
    history = [
        e for e in registry["version_history"] if e["model_name"] == model_name
    ]
    for entry in history:
        if entry["version"] == version:
            versioned_path = MODELS_DIR / entry["versioned_file"]
            canonical_path = MODELS_DIR / f"{model_name}.joblib"
            if not versioned_path.exists():
                print(f"✗ Versioned file missing: {versioned_path}")
                return False
            shutil.copy2(versioned_path, canonical_path)
            registry["models"][model_name] = {**entry, "note": f"rolled_back_from_v{version}"}
            _save_registry(registry)
            print(f"✓ Rolled back {model_name} to v{version}")
            return True
    print(f"✗ Version {version} not found for '{model_name}'")
    return False


def get_registry_summary() -> Dict[str, Any]:
    """Return a human-readable summary of all tracked models."""
    registry = _load_registry()
    summary = {}
    for name, entry in registry.get("models", {}).items():
        summary[name] = {
            "latest_version": entry["version"],
            "last_trained": entry["timestamp"],
            "versioned_file": entry["versioned_file"],
            "metadata": entry.get("metadata", {}),
        }
    return summary
