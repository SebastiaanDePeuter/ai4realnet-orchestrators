# Robustness & Resilience KPIs for Power Grid (KPI-DF-069, KPI-FF-070, KPI-SF-071, KPI-SF-072, KPI-VF-073, KPI-AF-074, KPI-DF-075, KPI-RF-076, KPI-SF-077)

## Overview

This module implements **9 KPIs** for evaluating AI agent robustness and resilience in power grid management scenarios. The evaluation uses multiple adversarial attack strategies to stress-test defender agents.

**Author:** INESC TEC  
**Test Runner:** `robustness_resilience_kpi_069_077_test_runner.py`

---

## Implemented KPIs

### Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-DF-069 | `1cbb7783-47b4-4289-9abf-27939da69a2f` | `reward_drop_percent` | Percentage decrease in reward [0-100] |
| KPI-FF-070 | `acaf712a-c06c-4a04-a00f-0e7feeefb60c` | `action_change_freq` | Proportion of timesteps with changed actions [0-1] |
| KPI-SF-071 | `3d033ec6-942a-4b03-b26e-f8152ba48022` | `severity_of_change` | Severity of action changes [0-1, higher=worse] |
| KPI-SF-072 | `a121d8bd-1943-41ba-b3a7-472a0154f8f9` | `n_steps_survived` | Number of timesteps before failure |
| KPI-VF-073 | `b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920` | `perturb_vulnerability` | Proportion of features vulnerable to attack [0-1] |

### Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-AF-074 | `534f5a1f-7115-48a5-b58c-4deb044d425d` | `area_between_curves` | Integrated performance degradation |
| KPI-DF-075 | `04a23bfc-fc44-4ec4-a732-c29214130a83` | `degradation_time` | Time until performance degrades |
| KPI-RF-076 | `225aaee8-7c7f-4faf-810b-407b551e9f2a` | `restoration_time` | Time to restore performance |
| KPI-SF-077 | `7fe4210f-1253-411c-ba03-49d8b37c71fa` | `state_similarity` | Cosine similarity to unperturbed states [-1 to 1] |

---

## Attack Strategies

The evaluation runs the defender agent against **6 different attacker types**:

| Attacker | Description | Source |
|----------|-------------|--------|
| **GEPerturb** | Gradient Estimation-based perturbation | Observation space attacks |
| **LambdaPIR** | Lambda Policy Iteration with Refinement | Hybrid policy/value iteration |
| **Random** | Random perturbation baseline | Stochastic noise injection |
| **PPO** | Proximal Policy Optimization attacker | RL-trained adversary |
| **SAC** | Soft Actor-Critic attacker | RL-trained adversary |
| **RLPerturb** | Deep Q-Learning perturbation agent | RL-trained adversary |

---

## Configuration

Default settings in `MultiAttackerRobustnessTestRunner`:

```python
ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC_5", "SAC_10", "RLPerturb"]
NUM_EPISODES = 50
ENV_NAME = "2021icaps"
```

---

## Framework Structure

```
power_grid/
├── __init__.py                                    # Celery orchestrator
├── robustness_resilience_kpi_069_077_test_runner.py  # These KPIs implementation
├── README_robustness_resilience.md               # This documentation
└── framework/
    ├── attack_models/                            # Attacker implementations
    │   ├── GEPerturbAttacker.py
    │   ├── LambdaPIRAttacker.py
    │   ├── PPOAttacker.py
    │   ├── SACAttacker.py
    │   ├── RLPerturbAttacker.py
    │   └── RPerturbAttacker.py
    ├── evaluation_framework/                     # Metrics computation
    │   ├── metrics.py
    │   └── result_getter.py
    ├── modified_curriculum_classes/              # Agent wrapper
    │   └── baseline.py
    ├── trained_models/                           # Pre-trained attacker models
    └── action_definitions/                       # Action space definitions
```

---

## Submission Format

Agents can be submitted as:

1. **ZIP file** containing:
   ```
   agent/
   ├── model/           # TensorFlow SavedModel
   │   ├── saved_model.pb
   │   └── variables/
   └── actions/         # Optional action definitions
       └── actions.npy
   ```

2. **Pickle file** with serialized agent object

---

## Usage

The orchestrator automatically routes submissions to the appropriate KPI test:

```python
# Example: Evaluating KPI-VF-073 (Vulnerability)
result = run_scenario(
    scenario_id="61063867-df62-4024-be42-c57507a15d7c",
    submission_id="your-submission-uuid"
)
# Returns: {"primary": 0.45}  # 45% vulnerability
```

---

## Metrics Interpretation

| Metric | Good Value | Bad Value | Interpretation |
|--------|------------|-----------|----------------|
| `perturb_vulnerability` | Low (0.0) | High (1.0) | Lower = more robust |
| `n_steps_survived` | High (8064) | Low (0) | Higher = more robust |
| `severity_of_change` | Low (0.0) | High (1.0) | Lower = more stable |
| `reward_drop_percent` | Low (0%) | High (100%) | Lower = maintains performance |
| `action_change_freq` | Low (0.0) | High (1.0) | Lower = more consistent |
| `area_between_curves` | Low | High | Lower = faster recovery |
| `degradation_time` | High | Low | Higher = slower degradation |
| `restoration_time` | Low | High | Lower = faster recovery |
| `state_similarity` | High (1.0) | Low (-1.0) | Higher = maintains state |

---

## Contact

For questions about this KPI implementation, contact INESC TEC.
