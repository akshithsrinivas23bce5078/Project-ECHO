from pydantic import BaseModel
from typing import List, Literal, Optional


class NeuromotorState(BaseModel):
    """Phase 1: IDeLog-derived neuromotor state carried through the pipeline."""
    neuromotor_load: float = 0.0       # Unified neuromotor load [0.0, 1.0]
    sigma_mean: float = 0.40           # Mean σ (log response time)
    mu_mean: float = -0.80             # Mean μ (log time delay)
    sigma_deviation: float = 0.0       # σ deviation from baseline
    mu_deviation: float = 0.0          # μ deviation from baseline
    snr_quality: Literal[
        "unsatisfactory", "acceptable", "good", "excellent"
    ] = "acceptable"
    snr_confidence: float = 0.3        # SNR-derived confidence weight [0.0, 1.0]
    command_count: int = 0             # Number of IDeLog commands in window


class BeliefState(BaseModel):
    trust_mean: float
    trust_variance: float
    cognitive_load: float
    clc_raw: float
    emotion_label: str
    emotion_confidence: float
    neuromotor_state: Optional[NeuromotorState] = None

class TaskContext(BaseModel):
    current_action: str
    risk_level: Literal["low", "med", "high"]
    domain: Literal["enterprise", "gaming"] = "enterprise"

class EchoInputPayload(BaseModel):
    belief_state: BeliefState
    task_context: TaskContext
    interaction_history: List[dict]

class HcxaiOutput(BaseModel):
    response_verbosity: Literal["minimal", "standard", "detailed"]
    surface_uncertainty: bool
    suggested_ui_action: Literal[
        "none", "offer_break", "show_confidence_band", "simplify_view",
        "spawn_powerup", "decrease_difficulty", "increase_difficulty"
    ]
    explanation: str
    granular_adjustments: Optional[dict] = None
    opt_in_prompt: Optional[str] = None
    hud_declutter: bool = False
    highlight_targets: bool = False
