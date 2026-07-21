"""Factorized gridworld tasks for hierarchical structure-learning experiments.

The task contract deliberately separates five things that are often conflated:

1. the shared interface (grid size, start, exits, horizon, and state indexing),
2. task-specific wall geometry,
3. transition dynamics (deterministic or action slip),
4. hidden signed outcomes attached to cells, and
5. gain versus loss-avoidance framing.

Gymnasium agents always maximize the signed return.  A loss-avoidance task is
therefore represented by negative outcomes and a safe exit, not by changing the
optimizer from maximization to minimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


Position = Tuple[int, int]
ACTION_DELTAS: Dict[int, Position] = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


def _position(value: Sequence[int]) -> Position:
    if len(value) != 2:
        raise ValueError(f"positions require two coordinates, received {value!r}")
    return int(value[0]), int(value[1])


@dataclass
class TaskSpec:
    """Serializable definition of one member of a hierarchical task family."""

    name: str
    size: int
    start: Position
    terminals: Set[Position]
    walls: Set[Position] = field(default_factory=set)
    outcomes: Dict[Position, float] = field(default_factory=dict)
    inventory_positions: Set[Position] = field(default_factory=set)
    transition_mode: str = "deterministic"
    objective_mode: str = "gain"
    slip_prob: float = 0.0
    step_cost: float = -0.01
    max_steps: int = 80
    collect_once: bool = True
    inventory_state: bool = True
    transition_context: str = "deterministic"
    outcome_context: str = "gain"
    visible_theme: str = "abstract"
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.size = int(self.size)
        self.start = _position(self.start)
        self.terminals = {_position(position) for position in self.terminals}
        self.walls = {_position(position) for position in self.walls}
        self.outcomes = {
            _position(position): float(value) for position, value in self.outcomes.items()
        }
        self.inventory_positions = {
            _position(position) for position in self.inventory_positions
        } or set(self.outcomes)
        self.slip_prob = float(self.slip_prob)
        self.step_cost = float(self.step_cost)
        self.max_steps = int(self.max_steps)
        self.validate()

    def validate(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2")
        if self.transition_mode not in {"deterministic", "stochastic"}:
            raise ValueError("transition_mode must be 'deterministic' or 'stochastic'")
        if self.objective_mode not in {"gain", "loss_avoidance"}:
            raise ValueError("objective_mode must be 'gain' or 'loss_avoidance'")
        if not 0.0 <= self.slip_prob < 1.0:
            raise ValueError("slip_prob must lie in [0, 1)")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.terminals:
            raise ValueError("at least one terminal cell is required")

        positions = (
            {self.start} | self.terminals | self.walls
            | set(self.outcomes) | self.inventory_positions
        )
        outside = [position for position in positions if not self.in_bounds(position)]
        if outside:
            raise ValueError(f"positions outside the grid: {outside}")
        if self.start in self.walls or self.start in self.terminals:
            raise ValueError("the start cannot be a wall or terminal")
        if self.terminals & self.walls:
            raise ValueError("terminal cells cannot be walls")
        if set(self.outcomes) & self.walls:
            raise ValueError("outcome cells cannot be walls")
        if not set(self.outcomes).issubset(self.inventory_positions):
            raise ValueError("every outcome cell must be included in inventory_positions")

    def in_bounds(self, position: Position) -> bool:
        row, column = position
        return 0 <= row < self.size and 0 <= column < self.size

    def geometry_signature(self) -> Tuple[object, ...]:
        return (
            self.size,
            self.start,
            tuple(sorted(self.terminals)),
            tuple(sorted(self.walls)),
        )

    def interface_signature(self) -> Tuple[object, ...]:
        """State-encoding properties that must match across a task family.

        Start, exits, walls, rewards, horizon, and transition probabilities may
        all differ between tasks.  The size and ordered inventory positions must
        remain shared because they determine the meaning of every discrete state
        index used by a reusable Q-table.
        """

        return (
            self.size,
            self.collect_once,
            self.inventory_state,
            tuple(sorted(self.inventory_positions)),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "start": list(self.start),
            "terminals": [list(position) for position in sorted(self.terminals)],
            "walls": [list(position) for position in sorted(self.walls)],
            "outcomes": [
                {"position": list(position), "value": value}
                for position, value in sorted(self.outcomes.items())
            ],
            "inventory_positions": [
                list(position) for position in sorted(self.inventory_positions)
            ],
            "transition_mode": self.transition_mode,
            "objective_mode": self.objective_mode,
            "slip_prob": self.slip_prob,
            "step_cost": self.step_cost,
            "max_steps": self.max_steps,
            "collect_once": self.collect_once,
            "inventory_state": self.inventory_state,
            "transition_context": self.transition_context,
            "outcome_context": self.outcome_context,
            "visible_theme": self.visible_theme,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TaskSpec":
        outcomes = {
            _position(item["position"]): float(item["value"])
            for item in payload.get("outcomes", [])
        }
        return cls(
            name=str(payload["name"]),
            size=int(payload["size"]),
            start=_position(payload["start"]),
            terminals={_position(position) for position in payload["terminals"]},
            walls={_position(position) for position in payload.get("walls", [])},
            outcomes=outcomes,
            inventory_positions={
                _position(position)
                for position in payload.get("inventory_positions", outcomes)
            },
            transition_mode=str(payload.get("transition_mode", "deterministic")),
            objective_mode=str(payload.get("objective_mode", "gain")),
            slip_prob=float(payload.get("slip_prob", 0.0)),
            step_cost=float(payload.get("step_cost", -0.01)),
            max_steps=int(payload.get("max_steps", 80)),
            collect_once=bool(payload.get("collect_once", True)),
            inventory_state=bool(payload.get("inventory_state", True)),
            transition_context=str(payload.get("transition_context", "deterministic")),
            outcome_context=str(payload.get("outcome_context", "gain")),
            visible_theme=str(payload.get("visible_theme", "abstract")),
            metadata=dict(payload.get("metadata", {})),
        )


class HierarchicalGridWorld(gym.Env):
    """Gymnasium environment with hidden cell outcomes and configurable action slip."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        spec: TaskSpec,
        render_mode: Optional[str] = None,
        *,
        include_debug_labels: bool = False,
        slip_prob: Optional[float] = None,
    ):
        super().__init__()
        self.spec = spec
        self.render_mode = render_mode
        self.include_debug_labels = bool(include_debug_labels)
        self.slip_prob = spec.slip_prob if slip_prob is None else float(slip_prob)
        if not 0.0 <= self.slip_prob < 1.0:
            raise ValueError("slip_prob override must lie in [0, 1)")
        # All tasks in a family use the same ordered inventory positions, even
        # when only some scenes attach a nonzero outcome to a position.  This
        # keeps Discrete state indices interoperable across reward-map edits.
        self._outcome_positions = tuple(sorted(spec.inventory_positions))
        self._outcome_index = {
            position: index for index, position in enumerate(self._outcome_positions)
        }
        self._inventory_states = (
            2 ** len(self._outcome_positions)
            if spec.collect_once and spec.inventory_state
            else 1
        )
        self.observation_space = spaces.Discrete(
            spec.size * spec.size * self._inventory_states
        )
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        self.state = self.position_to_state(spec.start)
        self.step_count = 0
        self.cumulative_return = 0.0
        self.collected: Set[Position] = set()

    # Generic aliases used by the supplied reference agents and plotting helpers.
    @property
    def size(self) -> int:
        return self.spec.size

    @property
    def start_pos(self) -> Position:
        return self.spec.start

    @property
    def goal_pos(self) -> Position:
        if len(self.spec.terminals) != 1:
            raise AttributeError("goal_pos is defined only when there is one terminal")
        return next(iter(self.spec.terminals))

    @property
    def obstacles(self) -> Set[Position]:
        return self.spec.walls

    @property
    def max_steps(self) -> int:
        return self.spec.max_steps

    @property
    def n_states(self) -> int:
        return self.observation_space.n

    @property
    def n_actions(self) -> int:
        return self.action_space.n

    def _inventory_mask(self, collected: Iterable[Position]) -> int:
        mask = 0
        for position in collected:
            index = self._outcome_index.get(position)
            if index is not None:
                mask |= 1 << index
        return mask

    def position_to_state(
        self, position: Position, collected: Optional[Iterable[Position]] = None
    ) -> int:
        position_state = int(position[0] * self.spec.size + position[1])
        if self._inventory_states == 1:
            return position_state
        if collected is None:
            collected = getattr(self, "collected", set())
        return position_state * self._inventory_states + self._inventory_mask(collected)

    def _pos_to_state(self, position: Position) -> int:
        return self.position_to_state(position)

    def state_to_position(self, state: int) -> Position:
        position_state = int(state) // self._inventory_states
        return divmod(position_state, self.spec.size)

    def state_to_collected(self, state: int) -> Set[Position]:
        if self._inventory_states == 1:
            return set()
        mask = int(state) % self._inventory_states
        return {
            position
            for index, position in enumerate(self._outcome_positions)
            if mask & (1 << index)
        }

    def _state_to_pos(self, state: int) -> Position:
        return self.state_to_position(state)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        requested_state = None if options is None else options.get("state")
        if requested_state is None:
            self.state = self.position_to_state(self.spec.start, collected=set())
        else:
            requested_state = int(requested_state)
            if not self.observation_space.contains(requested_state):
                raise ValueError(f"probe reset state is invalid: {requested_state}")
            position = self.state_to_position(requested_state)
            if position in self.spec.walls or position in self.spec.terminals:
                raise ValueError(f"probe reset state is not traversable: {position}")
            self.state = requested_state
        self.step_count = 0
        self.cumulative_return = 0.0
        self.collected = self.state_to_collected(self.state)
        return self.state, self._info(intended_action=None, executed_action=None, outcome=0.0)

    def action_execution_probs(self, intended_action: int) -> np.ndarray:
        if intended_action not in ACTION_DELTAS:
            raise ValueError(f"invalid action {intended_action}")
        probabilities = np.zeros(self.n_actions, dtype=float)
        if self.slip_prob == 0.0:
            probabilities[intended_action] = 1.0
            return probabilities
        probabilities.fill(self.slip_prob / (self.n_actions - 1))
        probabilities[intended_action] = 1.0 - self.slip_prob
        return probabilities

    def move(self, state: int, executed_action: int) -> int:
        row, column = self.state_to_position(state)
        delta_row, delta_column = ACTION_DELTAS[int(executed_action)]
        candidate = row + delta_row, column + delta_column
        if not self.spec.in_bounds(candidate) or candidate in self.spec.walls:
            candidate = row, column
        return self.position_to_state(candidate, collected=self.state_to_collected(state))

    def arrival_state(self, state: int) -> int:
        """Mark an uncollected outcome at `state` as collected in the next state."""

        position = self.state_to_position(state)
        collected = self.state_to_collected(state)
        if self.spec.collect_once and position in self.spec.outcomes:
            collected.add(position)
        return self.position_to_state(position, collected=collected)

    def transition_distribution(self, state: int, intended_action: int) -> Dict[int, float]:
        distribution: Dict[int, float] = {}
        for executed_action, probability in enumerate(
            self.action_execution_probs(intended_action)
        ):
            if probability == 0.0:
                continue
            next_state = self.arrival_state(self.move(state, executed_action))
            distribution[next_state] = distribution.get(next_state, 0.0) + float(probability)
        return distribution

    def transition_outcome_distribution(
        self, state: int, intended_action: int
    ) -> Dict[Tuple[int, float, bool], float]:
        """Exact distribution over next state, signed feedback, and termination."""

        distribution: Dict[Tuple[int, float, bool], float] = {}
        collected_before = self.state_to_collected(state)
        for executed_action, probability in enumerate(
            self.action_execution_probs(intended_action)
        ):
            if probability == 0.0:
                continue
            moved_state = self.move(state, executed_action)
            position = self.state_to_position(moved_state)
            available = not self.spec.collect_once or position not in collected_before
            outcome = float(self.spec.outcomes.get(position, 0.0)) if available else 0.0
            next_state = self.arrival_state(moved_state)
            key = (next_state, self.spec.step_cost + outcome, position in self.spec.terminals)
            distribution[key] = distribution.get(key, 0.0) + float(probability)
        return distribution

    def transition(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Nominal inspection transition without sampling action slip.

        The reference `GridWorldEnv.transition` method is deterministic and is used
        mainly for inspection. Real interaction must call `step`, which samples the
        configured stochastic kernel.
        """

        moved_state = self.move(state, int(action))
        position = self.state_to_position(moved_state)
        collected_before = self.state_to_collected(state)
        available = not self.spec.collect_once or position not in collected_before
        outcome = float(self.spec.outcomes.get(position, 0.0)) if available else 0.0
        return self.arrival_state(moved_state), self.spec.step_cost + outcome, (
            position in self.spec.terminals
        )

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        probabilities = self.action_execution_probs(int(action))
        executed_action = int(self.np_random.choice(self.n_actions, p=probabilities))
        moved_state = self.move(self.state, executed_action)
        next_position = self.state_to_position(moved_state)

        reward = self.spec.step_cost
        outcome = 0.0
        available = not self.spec.collect_once or next_position not in self.collected
        if next_position in self.spec.outcomes and available:
            outcome = float(self.spec.outcomes[next_position])
            reward += outcome
            if self.spec.collect_once:
                self.collected.add(next_position)

        self.state = self.arrival_state(moved_state)
        self.step_count += 1
        self.cumulative_return += reward
        terminated = next_position in self.spec.terminals
        truncated = self.step_count >= self.spec.max_steps and not terminated
        info = self._info(
            intended_action=int(action), executed_action=executed_action, outcome=outcome
        )
        return self.state, float(reward), bool(terminated), bool(truncated), info

    def _info(
        self,
        *,
        intended_action: Optional[int],
        executed_action: Optional[int],
        outcome: float,
    ) -> dict:
        info = {
            "position": self.state_to_position(self.state),
            "intended_action": intended_action,
            "executed_action": executed_action,
            "slipped": (
                intended_action is not None
                and executed_action is not None
                and intended_action != executed_action
            ),
            "revealed_outcome": float(outcome),
            "collected": tuple(sorted(self.collected)),
            "cumulative_return": float(self.cumulative_return),
        }
        if self.include_debug_labels:
            info.update({
                "task_name": self.spec.name,
                "transition_context": self.spec.transition_context,
                "outcome_context": self.spec.outcome_context,
            })
        return info

    def render(self, reveal_outcomes: bool = False) -> str:
        rows: List[List[str]] = [
            ["·" for _ in range(self.spec.size)] for _ in range(self.spec.size)
        ]
        for row, column in self.spec.walls:
            rows[row][column] = "#"
        for row, column in self.spec.terminals:
            rows[row][column] = "E"
        start_row, start_column = self.spec.start
        rows[start_row][start_column] = "S"
        if reveal_outcomes:
            for (row, column), value in self.spec.outcomes.items():
                if (row, column) not in self.spec.terminals:
                    rows[row][column] = "+" if value > 0 else "−" if value < 0 else "0"
        agent_row, agent_column = self.state_to_position(self.state)
        rows[agent_row][agent_column] = "A"
        return "\n".join(" ".join(row) for row in rows)


def validate_task_family(
    specs: Mapping[str, TaskSpec], *, require_crossed_factorial: bool = True
) -> dict:
    """Validate agent interoperability and the transition × objective crossing.

    Every experimental layout field may differ.  Only the discrete state encoding
    remains matched, allowing one agent/Q-table implementation to play every task.
    Factor labels are validated as design metadata but are never exposed to agents.
    """

    if len(specs) != 4:
        raise ValueError(f"the primary task family requires four tasks, received {len(specs)}")
    values = list(specs.values())
    reference_interface = values[0].interface_signature()
    mismatched = [
        spec.name for spec in values
        if spec.interface_signature() != reference_interface
    ]
    if mismatched:
        raise ValueError(f"task interfaces differ for: {mismatched}")

    factor_pairs = {(spec.transition_mode, spec.objective_mode) for spec in values}
    expected_pairs = {
        (transition, objective)
        for transition in ["deterministic", "stochastic"]
        for objective in ["gain", "loss_avoidance"]
    }
    if require_crossed_factorial and factor_pairs != expected_pairs:
        raise ValueError(f"factor crossing is incomplete: {sorted(factor_pairs)}")

    wall_signatures = {tuple(sorted(spec.walls)) for spec in values}
    outcome_signatures = {
        tuple(sorted(spec.outcomes.items())) for spec in values
    }
    start_matched = len({spec.start for spec in values}) == 1
    terminal_signatures = {tuple(sorted(spec.terminals)) for spec in values}
    horizon_matched = len({spec.max_steps for spec in values}) == 1
    return {
        "n_tasks": len(values),
        "interface_matched": True,
        # Kept for backward compatibility with earlier notebook cells/tests.
        "geometry_matched": len(wall_signatures) == 1,
        "walls_matched": len(wall_signatures) == 1,
        "n_distinct_wall_layouts": len(wall_signatures),
        "outcomes_matched": len(outcome_signatures) == 1,
        "n_distinct_outcome_layouts": len(outcome_signatures),
        "start_matched": start_matched,
        "terminals_matched": len(terminal_signatures) == 1,
        "horizon_matched": horizon_matched,
        "factor_pairs": sorted(factor_pairs),
        "size": values[0].size,
        "start": values[0].start if start_matched else None,
        "starts": {spec.name: spec.start for spec in values},
        "terminals": (
            sorted(values[0].terminals) if len(terminal_signatures) == 1 else None
        ),
        "terminals_by_task": {
            spec.name: sorted(spec.terminals) for spec in values
        },
        "n_walls": {spec.name: len(spec.walls) for spec in values},
        "n_outcomes": {spec.name: len(spec.outcomes) for spec in values},
    }


def finite_horizon_policy_solution(
    spec: TaskSpec,
    *,
    policy: str = "optimal",
    horizon: Optional[int] = None,
) -> dict:
    """Solve a known finite-horizon MDP and retain its time-dependent policy.

    Index ``remaining`` in the returned arrays means exactly that many actions
    remain.  This matters when outcomes are consumable: position alone is not a
    sufficient state, and a stationary shortest-path arrow field can be wrong.
    The solution is an experimenter-side audit and is never exposed to agents.
    """

    if policy not in {"optimal", "random", "worst"}:
        raise ValueError("policy must be 'optimal', 'random', or 'worst'")
    env = HierarchicalGridWorld(spec)
    horizon = spec.max_steps if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")

    transition_table = [
        [env.transition_outcome_distribution(state, action) for action in range(env.n_actions)]
        for state in range(env.n_states)
    ]
    values_by_remaining = [np.zeros(env.n_states, dtype=float)]
    exits_by_remaining = [np.zeros(env.n_states, dtype=float)]
    actions_by_remaining = [np.full(env.n_states, -1, dtype=int)]

    for _remaining in range(1, horizon + 1):
        previous_values = values_by_remaining[-1]
        previous_exits = exits_by_remaining[-1]
        values = np.zeros(env.n_states, dtype=float)
        exits = np.zeros(env.n_states, dtype=float)
        actions = np.full(env.n_states, -1, dtype=int)
        for state in range(env.n_states):
            if env.state_to_position(state) in spec.terminals:
                continue
            action_values = np.zeros(env.n_actions, dtype=float)
            action_exits = np.zeros(env.n_actions, dtype=float)
            for action, outcomes in enumerate(transition_table[state]):
                for (next_state, reward, terminal), probability in outcomes.items():
                    action_values[action] += probability * (
                        reward + (0.0 if terminal else previous_values[next_state])
                    )
                    action_exits[action] += probability * (
                        1.0 if terminal else previous_exits[next_state]
                    )
            if policy == "random":
                values[state] = float(action_values.mean())
                exits[state] = float(action_exits.mean())
            else:
                chooser = np.argmax if policy == "optimal" else np.argmin
                action = int(chooser(action_values))
                values[state] = action_values[action]
                exits[state] = action_exits[action]
                actions[state] = action
        values_by_remaining.append(values)
        exits_by_remaining.append(exits)
        actions_by_remaining.append(actions)

    initial_state = env.position_to_state(spec.start, collected=set())
    return {
        "policy": policy,
        "horizon": horizon,
        "initial_state": initial_state,
        "expected_return": float(values_by_remaining[horizon][initial_state]),
        "exit_probability": float(exits_by_remaining[horizon][initial_state]),
        "values_by_remaining": values_by_remaining,
        "exit_probabilities_by_remaining": exits_by_remaining,
        "actions_by_remaining": actions_by_remaining,
    }


def finite_horizon_policy_metrics(
    spec: TaskSpec,
    *,
    policy: str = "optimal",
    horizon: Optional[int] = None,
) -> dict:
    """Compute exact return and exit probability for a known finite-horizon MDP.

    `policy='optimal'` and `policy='worst'` select actions by expected signed
    return. `policy='random'` averages uniformly across actions. These are design
    diagnostics using the known task, never information supplied to an agent.
    """

    solution = finite_horizon_policy_solution(spec, policy=policy, horizon=horizon)
    return {
        "policy": policy,
        "horizon": solution["horizon"],
        "expected_return": solution["expected_return"],
        "exit_probability": solution["exit_probability"],
    }


def optimal_policy_trace(spec: TaskSpec, *, horizon: Optional[int] = None) -> dict:
    """Trace intended actions from the exact optimal finite-horizon policy.

    For a deterministic task this is the actual optimal path.  For a slippery
    task it is the *nominal* path obtained when each intended action executes as
    requested.  The stochastic optimum is a branching policy, so the audit also
    retains the exact expected return and exit probability; the line is a visual
    summary, not a claim that every stochastic episode follows it.
    """

    solution = finite_horizon_policy_solution(spec, policy="optimal", horizon=horizon)
    env = HierarchicalGridWorld(spec)
    state = int(solution["initial_state"])
    states = [state]
    positions = [env.state_to_position(state)]
    actions: List[int] = []
    rewards: List[float] = []
    terminated = False

    for remaining in range(solution["horizon"], 0, -1):
        action = int(solution["actions_by_remaining"][remaining][state])
        if action < 0:
            break
        next_state, reward, terminal = env.transition(state, action)
        actions.append(action)
        rewards.append(float(reward))
        states.append(next_state)
        positions.append(env.state_to_position(next_state))
        state = next_state
        terminated = bool(terminal)
        if terminated:
            break

    return {
        "trace_kind": (
            "deterministic_optimal_path"
            if spec.transition_mode == "deterministic" or spec.slip_prob == 0.0
            else "nominal_intended_path_under_stochastic_optimal_policy"
        ),
        "positions": positions,
        "states": states,
        "actions": actions,
        "action_names": [ACTION_NAMES[action] for action in actions],
        "nominal_return": float(sum(rewards)),
        "reaches_exit": terminated,
        "expected_return": solution["expected_return"],
        "exit_probability": solution["exit_probability"],
    }


def audit_task_family_difficulty(
    specs: Mapping[str, TaskSpec], *, tolerance: float = 0.15
) -> dict:
    """Audit known-MDP return ranges and normalized policy difficulty.

    Difficulty is summarized as the random-to-optimal gap divided by the full
    worst-to-optimal return range. This scale-free quantity permits gain/loss
    comparisons without pretending that raw signed returns have the same origin.
    """

    validate_task_family(specs)
    task_metrics: Dict[str, dict] = {}
    for name, spec in specs.items():
        optimal = finite_horizon_policy_metrics(spec, policy="optimal")
        trace = optimal_policy_trace(spec)
        random_policy = finite_horizon_policy_metrics(spec, policy="random")
        worst = finite_horizon_policy_metrics(spec, policy="worst")
        return_range = optimal["expected_return"] - worst["expected_return"]
        normalized_gap = (
            (optimal["expected_return"] - random_policy["expected_return"])
            / max(return_range, 1e-12)
        )
        task_metrics[name] = {
            "transition_mode": spec.transition_mode,
            "objective_mode": spec.objective_mode,
            "worst_return": worst["expected_return"],
            "random_return": random_policy["expected_return"],
            "optimal_return": optimal["expected_return"],
            "optimal_exit_probability": optimal["exit_probability"],
            "return_range": return_range,
            "normalized_random_to_optimal_gap": normalized_gap,
            "optimal_trace": trace,
        }

    comparisons = []
    for factor, levels, match_on in [
        ("transition", ["deterministic", "stochastic"], "objective_mode"),
        ("objective", ["gain", "loss_avoidance"], "transition_mode"),
    ]:
        matching_values = sorted({metrics[match_on] for metrics in task_metrics.values()})
        for matching_value in matching_values:
            selected = [
                (name, metrics)
                for name, metrics in task_metrics.items()
                if metrics[match_on] == matching_value
            ]
            if factor == "transition":
                selected.sort(key=lambda item: levels.index(item[1]["transition_mode"]))
            else:
                selected.sort(key=lambda item: levels.index(item[1]["objective_mode"]))
            left, right = selected
            gap_difference = abs(
                left[1]["normalized_random_to_optimal_gap"]
                - right[1]["normalized_random_to_optimal_gap"]
            )
            success_difference = abs(
                left[1]["optimal_exit_probability"]
                - right[1]["optimal_exit_probability"]
            )
            comparisons.append({
                "factor": factor,
                "matched_on": matching_value,
                "tasks": [left[0], right[0]],
                "normalized_gap_difference": gap_difference,
                "optimal_exit_difference": success_difference,
                "comparable": bool(
                    gap_difference <= tolerance and success_difference <= tolerance
                ),
            })

    acceptable = all(item["comparable"] for item in comparisons) and all(
        metrics["optimal_exit_probability"] >= 0.80
        for metrics in task_metrics.values()
    )
    return {
        "tolerance": float(tolerance),
        "tasks": task_metrics,
        "comparisons": comparisons,
        "all_optimal_exit_probability_at_least_0.80": all(
            metrics["optimal_exit_probability"] >= 0.80
            for metrics in task_metrics.values()
        ),
        "acceptable": bool(acceptable),
    }


def save_task_family(
    specs: Mapping[str, TaskSpec],
    path: Path | str,
    *,
    difficulty_audit: Optional[dict] = None,
    compute_difficulty_audit: bool = False,
) -> Path:
    """Validate and serialize a task family.

    Structural validation is always fast and mandatory. Difficulty auditing is
    independent and off by default, so the JSON records ``difficulty_audit: null``.
    Pass a completed ``difficulty_audit`` to store it, or explicitly set
    ``compute_difficulty_audit=True`` when a one-call audited export is desired.
    """

    validate_task_family(specs)
    target = Path(path)
    stored_audit = difficulty_audit
    if stored_audit is None and compute_difficulty_audit:
        stored_audit = audit_task_family_difficulty(specs)
    payload = {
        "schema_version": 1,
        "tasks": [spec.to_dict() for spec in specs.values()],
        "difficulty_audit": stored_audit,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def load_task_family(path: Path | str) -> Dict[str, TaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported schema version: {payload.get('schema_version')}")
    specs = {item["name"]: TaskSpec.from_dict(item) for item in payload["tasks"]}
    # Backward-compatible migration: combine every declared/resolved inventory
    # coordinate family-wide. Task-specific reward maps remain independent, but
    # a given state integer retains the same bit meanings in every environment.
    shared_inventory = (
        set().union(*(set(spec.outcomes) for spec in specs.values()))
        | set().union(*(set(spec.inventory_positions) for spec in specs.values()))
    )
    for spec in specs.values():
        spec.inventory_positions = set(shared_inventory)
        spec.validate()
    validate_task_family(specs)
    return specs


def make_default_task_family() -> Dict[str, TaskSpec]:
    """Create a hand-crafted 10×10 family with controlled wall differences.

    Outcome locations and magnitudes are intentionally paired across deterministic
    and stochastic variants.  The group can edit these maps in the designer before
    treating them as the fixed experimental stimuli.
    """

    size = 10
    start = (9, 5)
    terminals = {(0, 5)}
    # Alternating barrier gaps create a long, serpentine route while leaving
    # side branches for optional outcomes.  The two distal walls are exchanged
    # for near-start gates below, keeping wall count fixed at 23 in every task.
    base_walls = (
        {(2, column) for column in range(1, 9) if column != 2}
        | {(4, column) for column in range(1, 9) if column != 7}
        | {(6, column) for column in range(1, 9) if column != 2}
        | {(8, 1), (8, 8)}
    )
    gain_outcomes = {
        (6, 2): 1.00,
        (7, 2): 0.50,
        (8, 4): 0.20,
        (8, 7): 0.30,
    }
    loss_outcomes = {position: -value for position, value in gain_outcomes.items()}
    inventory_positions = set(gain_outcomes) | set(loss_outcomes)

    # The two near-start gate cells provide compositional probe evidence:
    #   left gate (9, 4)  -> stochastic transition context
    #   right gate (9, 6) -> loss-avoidance outcome context
    # Each added gate replaces a distal wall so all tasks retain 23 walls.  This
    # is a designed context cue through experienced dynamics, not a label in the
    # observation.  Remove these gates to test inference without structural cues.
    wall_layouts = {
        "deterministic_gain": set(base_walls),
        "deterministic_loss": (set(base_walls) - {(8, 8)}) | {(9, 6)},
        "stochastic_gain": (set(base_walls) - {(8, 1)}) | {(9, 4)},
        "stochastic_loss": (
            set(base_walls) - {(8, 1), (8, 8)}
        ) | {(9, 4), (9, 6)},
    }

    definitions = [
        ("deterministic_gain", "deterministic", "gain", 0.0, "forest_foraging"),
        ("deterministic_loss", "deterministic", "loss_avoidance", 0.0, "safe_navigation"),
        ("stochastic_gain", "stochastic", "gain", 0.20, "windy_foraging"),
        ("stochastic_loss", "stochastic", "loss_avoidance", 0.20, "ice_hazards"),
    ]
    specs: Dict[str, TaskSpec] = {}
    for name, transition, objective, slip_prob, theme in definitions:
        outcomes = gain_outcomes if objective == "gain" else loss_outcomes
        specs[name] = TaskSpec(
            name=name,
            size=size,
            start=start,
            terminals=set(terminals),
            walls=set(wall_layouts[name]),
            outcomes=dict(outcomes),
            inventory_positions=set(inventory_positions),
            transition_mode=transition,
            objective_mode=objective,
            slip_prob=slip_prob,
            step_cost=-0.01,
            max_steps=120,
            collect_once=True,
            inventory_state=True,
            transition_context=transition,
            outcome_context=objective,
            visible_theme=theme,
            metadata={
                "template": True,
                "outcomes_hidden_from_agent": True,
                "wall_probe_cue": {
                    "transition_gate_left": transition == "stochastic",
                    "loss_gate_right": objective == "loss_avoidance",
                },
            },
        )
    validate_task_family(specs)
    return specs


def make_task_environments(
    specs: Mapping[str, TaskSpec],
    *,
    seed: Optional[int] = None,
    slip_prob_overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, HierarchicalGridWorld]:
    """Build a named environment family for any compatible discrete-state agent.

    ``slip_prob_overrides`` permits runtime manipulation without rewriting the
    saved task specifications, for example
    ``{"stochastic_gain": 0.35, "stochastic_loss": 0.35}``.  A nonzero override
    makes movement stochastic even if the stored task has zero slip. Debug task
    labels remain withheld from ``info``.
    """

    overrides = dict(slip_prob_overrides or {})
    unknown = set(overrides) - set(specs)
    if unknown:
        raise KeyError(f"slip overrides reference unknown tasks: {sorted(unknown)}")
    envs = {
        name: HierarchicalGridWorld(
            spec,
            render_mode="ansi",
            include_debug_labels=False,
            slip_prob=overrides.get(name),
        )
        for name, spec in specs.items()
    }
    if seed is not None:
        for offset, env in enumerate(envs.values()):
            env.reset(seed=int(seed) + offset)
    return envs


def plot_task(
    spec: TaskSpec,
    ax,
    *,
    reveal_outcomes: bool = True,
    agent_position: Optional[Position] = None,
    optimal_path: Optional[Sequence[Position]] = None,
    title: Optional[str] = None,
) -> None:
    """Matplotlib rendering shared by the notebook and interactive designer."""

    image = np.ones((spec.size, spec.size, 3), dtype=float)
    for row, column in spec.walls:
        image[row, column] = [0.20, 0.20, 0.20]
    ax.imshow(image, interpolation="nearest")
    for gridline in np.arange(-0.5, spec.size, 1):
        ax.axhline(gridline, color="0.72", linewidth=0.5)
        ax.axvline(gridline, color="0.72", linewidth=0.5)

    if reveal_outcomes:
        for (row, column), value in spec.outcomes.items():
            color = "forestgreen" if value > 0 else "firebrick" if value < 0 else "0.4"
            ax.scatter(column, row, s=390, marker="o", facecolors="white",
                       edgecolors=color, linewidths=1.7)
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    start_row, start_column = spec.start
    ax.text(start_column, start_row, "S", ha="center", va="center",
            color="royalblue", fontweight="bold", fontsize=12)
    for row, column in spec.terminals:
        ax.text(column, row, "E", ha="center", va="center",
                color="darkorange", fontweight="bold", fontsize=12)
    if agent_position is not None:
        row, column = agent_position
        ax.scatter(column, row, s=120, color="royalblue", marker="o", zorder=4)

    if optimal_path:
        path = [tuple(position) for position in optimal_path]
        for step, (source, target) in enumerate(zip(path, path[1:]), start=1):
            source_row, source_column = source
            target_row, target_column = target
            if source == target:
                continue
            ax.annotate(
                "",
                xy=(target_column, target_row),
                xytext=(source_column, source_row),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "deepskyblue",
                    "linewidth": 2.0,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
                zorder=5,
            )
        if len(path) > 1:
            rows, columns = zip(*path)
            ax.plot(columns, rows, color="deepskyblue", linewidth=1.0,
                    alpha=0.55, zorder=4)

    dynamics = "det" if spec.transition_mode == "deterministic" else f"slip={spec.slip_prob:.2f}"
    ax.set(
        xticks=[],
        yticks=[],
        title=title or f"{spec.name}\n{dynamics}; {spec.objective_mode}",
    )
