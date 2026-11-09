"""Small helpers for experiment reproducibility.

Every training and evaluation manifest should carry: feature_version,
feature_file_sha256, git_commit_hash, seed and date (UTC, ISO 8601). This
module produces them, and each script obtains them in one call through
`manifest_entry()`.
"""
import datetime
import subprocess

from src.data.featurize import get_feature_provenance


def get_git_commit() -> str:
    """The git commit hash, or 'unknown' if there is no repository or no commit."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


BASELINE_HIDDEN_DIM = 128


def active_components(config: dict) -> list[str]:
    """A readable list of the config's `ablation` flags that differ from the
    baseline, for example ["gin_layers=5", "use_asl=True"]. An empty list means
    this is the plain baseline.
    """
    defaults = {"use_asl": False, "use_rdkit_descriptors": False,
                "use_pretrained_weights": False, "gin_layers": 3,
                "task_weighting": "none", "use_swa": False}
    ablation = config.get("ablation", {})
    comps = [f"{k}={ablation[k]}" for k, default in defaults.items()
             if k in ablation and ablation[k] != default]
    hidden_dim = config.get("model", {}).get("hidden_dim", BASELINE_HIDDEN_DIM)
    if hidden_dim != BASELINE_HIDDEN_DIM:
        comps.append(f"hidden_dim={hidden_dim}")
    return comps


def manifest_entry(seed: int, config: dict | None = None, model_name: str | None = None,
                    extra: dict | None = None) -> dict:
    """The common provenance fields to add to every manifest record.

    If `config` is given, `active_components` (the ablation flags that differ
    from the baseline) is included. Without `model_name`, the name is
    "gin-v2-baseline" when no flag is set, or derived from the active
    components otherwise.

    Usage: `manifest.append(manifest_entry(seed, config=config, extra={"path": ckpt_path, ...}))`

    Note: manifests written before 2026-11-05 carry the same two fields under
    the names `model_adi` and `aktif_bilesenler`.
    """
    fp = get_feature_provenance()
    comps = active_components(config) if config is not None else []
    entry = {
        "model_name": model_name or ("gin-v2-baseline" if not comps else "gin-v2+" + "+".join(comps)),
        "active_components": comps,
        "feature_version": fp.get("version"),
        "feature_file_sha256": fp.get("sha256"),
        "git_commit_hash": get_git_commit(),
        "seed": seed,
        "date": utc_now_iso(),
    }
    if extra:
        entry.update(extra)
    return entry
