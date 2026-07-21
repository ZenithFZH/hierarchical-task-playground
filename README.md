# Hierarchical Task Playground

This repository is a reproducible playground for studying how agents reuse structure across a hand-designed family of four 10×10 gridworlds. The tasks cross two experimental factors:

| Transition dynamics | Gain maximization | Loss avoidance |
|---|---|---|
| Deterministic | forest foraging | safe navigation |
| Stochastic | windy foraging | slippery hazard avoidance |

The audited default tasks expose the same Gymnasium-style agent interface and share the grid size, start, exit, action set, state encoding, horizon, and one-time collection rule. Their wall layouts differ, while outcome layouts are paired where the default factorial design calls for it. The editor does not enforce that coupling: each scene can be changed independently. The agent receives states, rewards, and termination signals—not task names or factor labels.

The repository contains the softmax fresh-start, persistent, and reference DINER agents used in `day2_track1_recovery.ipynb`, plus `StochasticSoftmaxCRPDyna`, which learns stochastic transition distributions and uses graded context assignment.

## Repository map

- `hierarchical_task_playground.ipynb`: main interactive tutorial and experiment playground.
- `four_task_family.json`: working four-task stimulus definition.
- `four_task_family_default_acceptable.json`: unchanged audited default to restore or compare against.
- `hierarchical_gridworld.py`: task schema, Gymnasium environment, plotting, validation, difficulty audit, and JSON I/O.
- `task_designer.py`: Jupyter widget and widget-independent editing logic.
- `stochastic_crp_dyna_agent.py`: stochastic softmax CRP-Dyna agent.
- `recovery.py`, `models.py`, `environments.py`: reference tutorial agents and supporting code.
- `day2_track1_recovery.ipynb`: the original parameter/model-recovery tutorial for provenance and further analysis.
- `build_hierarchical_playground_notebook.py`: reproducibly rebuilds the main notebook.
- `tests/`: environment, widget logic, reference-agent, and stochastic-agent tests.

## Install

Python 3.11 or newer is recommended.

```bash
git clone <YOUR-REPOSITORY-URL>
cd hierarchical-task-playground
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name hierarchical-task-playground \
  --display-name "Python (hierarchical-task-playground)"
jupyter lab hierarchical_task_playground.ipynb
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. In Jupyter, select the `Python (hierarchical-task-playground)` kernel. The helper modules are already included locally, so any Colab-only download/setup cells in the reference notebook can be skipped.

If widgets appear as text or do not respond, verify that the notebook is trusted, the kernel above is active, and `ipywidgets` is installed in that same kernel:

```bash
jupyter trust hierarchical_task_playground.ipynb
python -c "import ipywidgets; print(ipywidgets.__version__)"
```

## Load and initialize the gym

```python
from hierarchical_gridworld import (
    load_task_family,
    make_task_environments,
    validate_task_family,
)

specs = load_task_family("four_task_family.json")
print(validate_task_family(specs))

envs = make_task_environments(specs, seed=2026)
env = envs["stochastic_gain"]
state, info = env.reset(seed=2026)
next_state, reward, terminated, truncated, info = env.step(1)
```

Each named environment follows the familiar Gymnasium contract:

- `reset()` returns `(state, info)`.
- `step(action)` returns `(next_state, reward, terminated, truncated, info)`.
- actions are `0=up`, `1=right`, `2=down`, and `3=left`.
- an outcome is collected at most once per episode.
- episodes end at the exit or at `max_steps`.
- ordinary steps contribute that task's `step_cost`.

The state includes the agent position and the collected-outcome inventory. This keeps the process Markov after one-time collection and makes all four tasks share identical state numbering.

### Change slippery motion

There are three supported levels of control.

To change the saved design, edit each stochastic `TaskSpec` and save a new JSON:

```python
from hierarchical_gridworld import save_task_family

specs["stochastic_gain"].slip_prob = 0.35
specs["stochastic_loss"].slip_prob = 0.35
save_task_family(
    specs,
    "four_task_family_slip035.json",
    compute_difficulty_audit=False,  # fast design-iteration save
)
```

To manipulate slip for one run without changing the JSON:

```python
envs = make_task_environments(
    specs,
    seed=2026,
    slip_prob_overrides={
        "stochastic_gain": 0.35,
        "stochastic_loss": 0.35,
    },
)
```

To initialize one environment directly:

```python
from hierarchical_gridworld import HierarchicalGridWorld

env = HierarchicalGridWorld(specs["stochastic_gain"], slip_prob=0.35)
```

With `slip_prob=p`, the intended action is executed with probability `1-p`; each of the four actions, including the intended action, receives `p/4` from uniform substitution. Thus the intended action's total probability is `1 - 3p/4`.

## Use the interactive task designer

Run this in a Jupyter cell:

```python
from hierarchical_gridworld import load_task_family
from task_designer import TaskDesigner

specs = load_task_family("four_task_family.json")
designer = TaskDesigner(specs, export_path="four_task_family_edited.json")
designer.display()
```

Recommended workflow:

1. Choose a scene with the **Task** dropdown.
2. Select `inspect`, `start`, `exit`, `wall`, `outcome`, or `clear outcome`.
3. Click a grid cell. Clicking an existing wall with the `wall` tool toggles it back to an ordinary cell.
4. Set **Outcome value** before placing an outcome. Outcome edits affect only the active scene by default. Enable **Link outcome edits by objective** only when the same edit is intentional in both transition variants of an objective frame.
5. Start, exit, wall, slip, and step-cost edits are also active-scene-only. Their adjacent linking checkboxes are explicit opt-ins; simply switching scenes never copies values.
6. In a stochastic scene, move **Slip probability** to change that scene. Enable **Link slip by transition type** to copy it to the other stochastic scene. The slider is disabled in deterministic scenes.
7. Set an export filename and click **Validate and save (fast)**. This checks the common state encoding and complete 2×2 factor crossing, then writes the JSON immediately with `difficulty_audit: null`. It permits different starts, exits, walls, outcomes, horizons, slip probabilities, and step costs.
8. Click **Audit difficulty** only when you need the expensive known-MDP check. It is independent from saving and computes worst, random, and optimal returns, optimal exit probability, and the comparable-difficulty flag. The preview then adds the optimal path; for a slippery task it shows the nominal intended-action trace because the true policy branches after slips.

The left preview is the designer view with hidden outcomes revealed. The participant view conceals them. To use unsaved in-memory edits immediately:

```python
edited_specs = designer.editor.specs
envs = make_task_environments(edited_specs, seed=2026)
```

For a reproducible experiment, save the JSON and reload it before simulation rather than relying on widget state.

The main notebook also contains an explicit **Optional pre-agent audit and best-route visualization** cell immediately before the agent sections. It reads the current in-memory widget design, recomputes the audit, plots return ranges and all four best routes, and can be skipped during quick interface smoke tests. Run and review it before freezing stimuli, comparing agents substantively, or collecting human data.

To persist a completed audit alongside a task family:

```python
from hierarchical_gridworld import audit_task_family_difficulty, save_task_family

audit = audit_task_family_difficulty(edited_specs, tolerance=0.15)
save_task_family(
    edited_specs,
    "four_task_family_audited.json",
    difficulty_audit=audit,
)
```

Outcome maps are behaviorally independent, but their coordinate union is maintained as a shared inventory index. This is necessary so a state integer has the same meaning in every task. Consequently, adding many unique outcome locations increases the state space exponentially: with `m` one-time outcome coordinates, the default encoding has `size² × 2^m` states. Reuse a modest coordinate pool unless the larger state space is intentional.

## Build an episode schedule

A smoke schedule is intentionally tiny and only checks interoperability:

```python
task_names = list(specs)
smoke_sequence = task_names * 2
```

A sticky schedule produces recurrent blocks without revealing the current task to the agent:

```python
from environments import generate_sticky_env_sequence

long_sequence = generate_sticky_env_sequence(
    task_names,
    episodes=200,
    stay_prob=0.80,
    seed=2026,
)
```

The expected run length is `1 / (1 - stay_prob)`, so `stay_prob=0.80` gives runs of about five episodes on average. For confirmatory comparisons, generate the schedule once, save it, and give the identical sequence to every agent. Check the realized task counts and transition counts; a sticky generator does not guarantee exact balance in a finite sample.

## Run the reference softmax agents

The three reference simulators use the API from `recovery.py`. Construct a fresh environment family for every model so one model's RNG and episode state cannot affect another.

```python
from recovery import simulate_diner, simulate_fresh_start, simulate_persistent

def fresh_envs(seed):
    return make_task_environments(specs, seed=seed)

common = dict(eta=0.30, gamma=0.95, tau=0.50, planning_steps=5)

fresh_returns, fresh_trajectories = simulate_fresh_start(
    smoke_sequence, fresh_envs(101), **common, seed=101
)

persistent_returns, persistent_trajectories = simulate_persistent(
    smoke_sequence, fresh_envs(101), **common, seed=101
)

reference_returns, reference_assignments, reference_trajectories = simulate_diner(
    smoke_sequence,
    fresh_envs(101),
    crp_alpha=0.50,
    n_probe=20,
    **common,
    seed=101,
)
```

Model interpretation:

- **Fresh-start Dyna** discards its Q-table and transition model at every episode.
- **Persistent Dyna** uses one Q-table and model across every task.
- **Reference DINER** reuses a separate Dyna system for each inferred recurrent context.

The reference DINER's probe comparison and Dyna model assume deterministic transitions. It is retained as a deliberate comparison/misspecification condition on slippery tasks; a chance slip can look like a hard contradiction, and its model stores only one next state per state-action pair. Do not treat it as the fully stochastic model.

## Run stochastic softmax CRP-Dyna

```python
from stochastic_crp_dyna_agent import AgentConfig, StochasticSoftmaxCRPDyna

envs_stochastic = fresh_envs(101)
first_env = envs_stochastic[smoke_sequence[0]]

config = AgentConfig(
    eta=0.30,                       # Q-learning rate
    gamma=0.95,                     # discount factor
    tau=0.50,                       # softmax temperature
    crp_alpha=0.50,                 # prior mass for a new context
    planning_steps=5,
    n_probe=12,                     # probed state-action keys/steps
    probe_samples_per_step=5,       # repeated samples reveal stochasticity
    probe_eta=0.30,                 # delta rule for probe distributions
    model_eta=0.30,                 # delta rule for the Dyna transition model
    fingerprint_update_eta=0.30,    # across-episode context fingerprint update
    assignment_mode="sample",      # "sample" or deterministic "map"
)

agent = StochasticSoftmaxCRPDyna(
    n_states=first_env.observation_space.n,
    n_actions=first_env.action_space.n,
    config=config,
    seed=101,
)
result = agent.play_sequence(smoke_sequence, envs_stochastic)

print(result.returns)
print(result.assignments)
print(result.posteriors)
print(result.probe_distances)
```

This agent implements the three stochastic DINER features motivated in the recovery tutorial:

1. repeated probe transitions are converted into probability fingerprints by a delta rule;
2. Jensen–Shannon divergence provides a graded diagnostic distance, while a CRP prior plus graded categorical likelihood produces Bayesian context-assignment probabilities;
3. the Dyna model learns `P(next_state | state, action)` by a delta rule, and planning samples from that distribution.

The probe does not read the task name, transition factor, objective factor, walls, or outcome map. It learns only from sampled transitions. Repeated probes are additional observations: in a later human experiment, they must correspond to explicit, time-costed micro-trials rather than privileged access to the transition kernel.

## Smoke test versus planned long run

Use a smoke test after changing the JSON, environment, or agent. It answers “does every component run and return correctly shaped data?” It does not establish learning or model superiority.

For a planned comparison:

```python
import matplotlib.pyplot as plt
import numpy as np

sequence = long_sequence
seeds = range(20)

# Run each model with the same frozen sequence and matched seed set.
# Store one return vector per seed, then summarize across seeds.

def moving_average(values, width=10):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(width) / width, mode="valid")

plt.plot(moving_average(result.returns, width=10))
plt.xlabel("Episode")
plt.ylabel("Moving-average return")
plt.title("Stochastic CRP-Dyna learning curve")
plt.show()
```

A defensible long run should:

- freeze and version the task JSON and schedule;
- use the same episode sequence and matched seed set for all agents;
- rebuild fresh environments for each model and seed;
- report distributions or uncertainty across seeds, not only one average curve;
- plot return by episode and by task, exit rate, context assignments, posterior entropy, and probe JSD;
- separate probe transitions from rewarded episode transitions when reporting sample efficiency;
- rerun the difficulty audit after the final maze/outcome revision and before interpreting model comparisons.

The stochastic agent is computationally heavier because each probe step contains repeated transition samples. Start with 8 episodes, `planning_steps=1`, and `probe_samples_per_step=2` for debugging; use the planned values only after the smoke test passes.

## Parameter and model recovery

`day2_track1_recovery.ipynb` explains the original likelihood and recovery workflow. The functions `loglik_fresh_start`, `loglik_persistent`, `loglik_diner`, and `fit_model` remain available in `recovery.py` for the reference agents.

The new stochastic agent currently supplies simulation diagnostics, not a finished trajectory-likelihood function. Fitting it requires teacher-forced replay of the stochastic context posterior, transition-distribution updates, and softmax choices. Do not pass its trajectories to the deterministic reference DINER likelihood and interpret the result as valid stochastic-agent recovery.

## Test and rebuild

```bash
python -m pytest -q
python build_hierarchical_playground_notebook.py
jupyter nbconvert \
  --to notebook \
  --execute hierarchical_task_playground.ipynb \
  --output /tmp/hierarchical_task_playground.executed.ipynb \
  --ExecutePreprocessor.timeout=600
```

The notebook is generated from the build script. Make durable structural edits in `build_hierarchical_playground_notebook.py`, rebuild, and commit both files so collaborators can inspect the notebook normally and verify how it was produced.

Before sharing a result, record the Git commit, task JSON filename, schedule or schedule seed, agent configuration, simulation seeds, dependency versions, and whether reported interactions include probe samples.
