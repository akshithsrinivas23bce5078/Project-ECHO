# Project ECHO: Empathic Cognitive Hook Orchestrator

> A software-only, closed-loop interface system that dynamically calibrates user trust and manages cognitive load in human-agent teaming through non-intrusive behavioral biometrics.

---

## 1. Context and Problem Statement

The integration of agentic artificial intelligence into high-stakes, real-world operational environments introduces fundamental challenges for Human-Computer Interaction (HCI). The primary issues are:

- **Dynamic calibration of user trust** — current systems treat trust as a static variable, leading to *automation bias* (over-trust) or *algorithm aversion* (under-trust).
- **Real-time management of cognitive load** — existing systems cannot sense, quantify, and adapt to fluctuating psychological and neuromotor states without intrusive hardware (EEG, pupillometry).

Project ECHO solves both problems with a **software-only, closed-loop** approach using behavioral biometrics derived purely from keyboard and mouse input.

---

## 2. Project Objectives

| Objective | Approach |
|---|---|
| Dynamically calibrate user trust | Bayesian POMDP belief state estimation |
| Estimate cognitive load non-intrusively | Keystroke digraph latency + CLC metric |
| Reduce automation bias | Trust-calibrated explanation depth |
| Detect emotional states | DSTER (Dual-Stream Transformer) on keystroke dynamics |
| Governed extensibility | Google Antigravity SDK lifecycle hooks |

---

## 3. Methodology and Mathematical Modeling

### 3.1 Partially Observable Markov Decision Process (POMDP)

The system models human-agent interaction as a POMDP with:

- **State space**: `S = (trust_level × cognitive_load_level)` — 15 composite states
- **Observation space**: `O = (clc_raw, emotion_label, iki_mean, snr_db)`
- **Belief update**: `b'(s') = η · O(o|s') · Σ_s T(s'|s) · b(s)`

The true psychological state remains hidden; the system maintains a probability distribution (belief state) over all possible states, continuously updated via Bayesian inference.

### 3.2 Sigma-Lognormal Model (Neuromotor Kinematics)

Decomposes continuous mouse movement into time-overlapped lognormal primitives:

```
v(t) = Σ_i  D_i / (σ_i · (t - t0_i) · √(2π)) · exp(-(ln(t - t0_i) - μ_i)² / (2σ_i²))
```

Parameters per stroke: amplitude `D_i`, onset time `t0_i`, log-time mean `μ_i`, variance `σ_i`.

### 3.3 Cognitive Load Correlation (CLC)

Non-intrusive cognitive load estimation via keystroke digraph latencies:

```
CLC = ρ_s(complexity_ranks, iki_ranks)
```

- `ρ_s`: Spearman rank correlation
- **Nominal range**: CLC ∈ [0.35, 0.55] for genuine text composition
- CLC < 0.35 → possible pasted/non-generative text
- CLC > 0.55 → high deliberation or unfamiliar content

### 3.4 Dual-Stream Transformer Emotion Recognition (DSTER)

Extracts emotional states from keystroke dynamics via two parallel transformer streams:

| Stream | Axis | Captures |
|---|---|---|
| Temporal-over-Channel | Time | Rhythmic latency patterns |
| Channel-over-Temporal | Features | Cross-feature contextual relationships |

The streams are fused via an MLP into a unified emotion embedding, classified into: `frustrated`, `focused`, `fatigued`, `neutral`, `engaged`.

---

## 4. System Architecture

```
pynput listeners → Feature Extraction → DSTER Model → Bayesian POMDP
                                                           ↓
                                                   Belief State
                                                           ↓
                                              Adaptation Engine (Middleware)
                                                           ↓
                                                  HCXAI Formatter
                                                           ↓
                                                Adaptive React UI
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React / Next.js |
| Input Handling | pynput |
| Mathematical Processing | NumPy, SciPy, Pandas |
| AI Models | PyTorch, Stable Baselines3 |
| Agent SDK | Google Antigravity SDK |
| LLM Backend | Gemini Flash, MCP |
| API Server | FastAPI, Uvicorn |
| Data Contracts | Pydantic |

---

## 5. Project Structure

```
project-echo/
├── config/
│   └── echo_config.yaml              # Centralized thresholds
├── input_capture/
│   ├── keystroke_listener.py          # pynput keystroke capture
│   └── mouse_listener.py             # pynput mouse trajectory capture
├── feature_extraction/
│   ├── digraph_latency.py             # Digraph IKI extraction
│   ├── clc_estimator.py               # CLC (Spearman correlation)
│   └── sigma_lognormal.py             # Sigma-Lognormal neuromotor model
├── dster/
│   ├── temporal_over_channel.py       # Stream 1: rhythmic latency
│   ├── channel_over_temporal.py       # Stream 2: contextual features
│   ├── dster_model.py                 # Dual-stream fusion
│   └── emotion_classifier.py          # Emotion label + confidence
├── estimator/
│   └── src/
│       ├── pomdp_model.py             # Bayesian POMDP (15-state)
│       ├── belief_updater.py          # Observation fusion + belief update
│       └── sim_api.py                 # Simulation API
├── middleware/
│   └── src/
│       ├── models.py                  # Pydantic data contracts
│       ├── adaptation_engine.py       # Core behavioral rules
│       ├── server.py                  # FastAPI server
│       └── hcxai_formatter.py         # HCXAI output templating
├── frontend/
│   ├── public/index.html
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       ├── index.css
│       └── components/
│           ├── AdaptiveMessage.jsx
│           ├── ConfidenceBand.jsx
│           ├── BreakModal.jsx
│           ├── TrustGauge.jsx
│           └── CognitiveLoadBar.jsx
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_dster_accuracy.py         # F1 > 0.95
│   ├── test_neuromotor_snr.py         # SNR > 24 dB
│   ├── test_latency.py               # < 40 ms
│   ├── test_clc_range.py             # CLC ∈ [0.35, 0.55]
│   └── test_adaptation_engine.py      # All 5 behavioral rules
├── requirements.txt
└── README.md
```

---

## 6. Behavioral Rules (Adaptation Layer)

### Rule 1: Trust Calibration (Not Persuasion)
- **High trust + High risk** → increase explanation depth, surface uncertainty (counter automation bias)
- **Low trust + Low risk** → concise justification (counter algorithm aversion)
- **Never** manufacture trust through tone, urgency, or persuasive framing

### Rule 2: Cognitive Load Response
- `cognitive_load > 0.7` → shorten responses, defer detail, suggest break, simplify view
- `cognitive_load < 0.3` with `clc_raw` outside [0.35, 0.55] → flag measurement noise

### Rule 3: Emotion-Adaptive Tone
- Use `emotion_label` only for pacing/warmth adjustment
- If `emotion_confidence < 0.6`, ignore emotion signal entirely

### Rule 4: HCXAI Output Format
```json
{
  "response_verbosity": "minimal" | "standard" | "detailed",
  "surface_uncertainty": true | false,
  "suggested_ui_action": "none" | "offer_break" | "show_confidence_band" | "simplify_view",
  "explanation": "<plain language, no jargon>"
}
```

### Rule 5: Guardrails
- **NEVER** make task decisions or override agent actions
- **NEVER** disclose raw biometric values, DSTER scores, or POMDP internals
- Default to "standard" verbosity when confidence is low across the board

---

## 7. Testing and Validation

| Metric | Target | Test File |
|---|---|---|
| DSTER Accuracy | F1 > 0.95 | `tests/test_dster_accuracy.py` |
| Neuromotor Fluidity | SNR > 24 dB | `tests/test_neuromotor_snr.py` |
| Processing Latency | < 40 ms | `tests/test_latency.py` |
| Cognitive Load (CLC) | ∈ [0.35, 0.55] | `tests/test_clc_range.py` |
| Behavioral Rules | All 5 rules | `tests/test_adaptation_engine.py` |

### Running Tests
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## 8. Running the System

### Start the Middleware API
```bash
cd middleware/src
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Frontend
```bash
cd frontend
npm install
npm start
```

---

## 9. API Endpoint

### `POST /api/modulate`

**Request:**
```json
{
  "belief_state": {
    "trust_mean": 0.85,
    "trust_variance": 0.05,
    "cognitive_load": 0.4,
    "clc_raw": 0.45,
    "emotion_label": "focused",
    "emotion_confidence": 0.7
  },
  "task_context": {
    "current_action": "deploy to production",
    "risk_level": "high"
  },
  "interaction_history": []
}
```

**Response:**
```json
{
  "response_verbosity": "detailed",
  "surface_uncertainty": true,
  "suggested_ui_action": "show_confidence_band",
  "explanation": "I am preparing to deploy to production. Because this is a high-risk action, I want to highlight that there is some uncertainty in the current projection. Please review the underlying data carefully."
}
```

---

## 10. Dynamic Difficulty Adjustment (DDA) Integrations

Project ECHO's cognitive/emotional fusion engine is actively being applied to model the **Dynamic Difficulty Adjustment** systems seen in highly successful real-world games. The `game_framework` module includes real-time telemetry extraction for:

### 1. Left 4 Dead (The AI Director & Pacing Modulation)
Valve's "AI Director" tracks the "Emotional Intensity" of players based on damage taken and items used. Project ECHO expands this by sensing actual panic via facial movement and keystroke mashing.
- **Modulated Parameters**: `pacing_intensity`, `horde_spawn_frequency`, `health_item_scarcity`, `special_infected_cap`

### 2. Alien: Isolation (The Menace Gauge & Tension Budget)
The Xenomorph operates on a "Tension Budget" where the game ensures the player is stressed but not continuously overwhelmed. ECHO reads exact psychological states to modulate the Alien's behavior tree.
- **Modulated Parameters**: `alien_menace_gauge`, `tension_budget`, `alien_search_radius`, `noise_sensitivity`

### 3. Resident Evil 4 (Action Scoring & Hidden Metrics)
RE4 features a hidden 1-10 "Action Score" that dynamically affects enemy aggressiveness and item drops based on player deaths and accuracy. ECHO ties this directly to cognitive load.
- **Modulated Parameters**: `action_score`, `enemy_aggression`, `item_drop_quality`, `damage_taken_multiplier`

### 4. Silent Hill: Shattered Memories (Psychological Profiling)
The game actively profiles the player's psychology to alter monster appearances and environmental aesthetics. ECHO utilizes the emotion classifier (`joy`, `panic`, `focused`) to seamlessly warp the world states.
- **Modulated Parameters**: `world_decay_speed`, `monster_type_bias`, `puzzle_obfuscation`, `environment_lighting`

You can test these real-world DDA profiles using the high-performance visual analyzer demo:
```bash
python run_embedded_demo.py --genre left_4_dead
```
