"""Production runtime choices for native MLX WordVoice."""

PRODUCTION_FLOW_STEPS = 8
QUALITY_REFERENCE_FLOW_STEPS = 10


def resolve_flow_steps(override: int | None) -> int:
    """Return the production step count unless a valid override is supplied."""
    if override is not None and override < 1:
        raise ValueError(
            "MLX WordVoice flow_steps must be at least 1; "
            f"actual={override}, safe_recovery=omit-flow-steps-for-production-default"
        )
    return PRODUCTION_FLOW_STEPS if override is None else override
