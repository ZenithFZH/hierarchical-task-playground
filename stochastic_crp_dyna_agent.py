"""Stochastic softmax DINER with graded probe fingerprints and Dyna replay.

The implementation completes the stochastic-environment scaffold from the
recovery tutorial:

1. repeated probe transitions form probability fingerprints by a delta rule;
2. Jensen--Shannon divergence gives a graded context-distance diagnostic;
3. a CRP prior and graded transition likelihood give Bayesian cluster weights;
4. the replay model learns transition probabilities by the same delta rule;
5. planning samples from the learned transition distribution.

No task name, transition label, or outcome label is read by the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ProbeKey = Tuple[int, int]
ProbeObservation = Tuple[int, int, int, float, bool]
TrajectoryStep = Tuple[int, int, int, float, bool]
Distribution = Dict[int, float]
Fingerprint = Dict[ProbeKey, Distribution]


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def action_softmax(q_values: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return stable_softmax(np.asarray(q_values, dtype=float) / float(tau))


def normalized_distribution(distribution: Mapping[int, float]) -> Distribution:
    """Return a valid categorical distribution without mutating its source."""

    positive = {int(state): max(0.0, float(value)) for state, value in distribution.items()}
    total = float(sum(positive.values()))
    if total <= 0.0:
        return {}
    return {state: value / total for state, value in positive.items() if value > 0.0}


def delta_rule_distribution_update(
    distribution: Distribution,
    observed_state: int,
    eta_m: float,
) -> None:
    """Move a transition vector toward the observed one-hot outcome.

    This is the vector form of ``P <- P + eta_m * (delta - P)``. Existing
    alternatives decay and the observed next state receives the prediction-error
    increment. Values are normalized only when queried, matching the tutorial's
    stochastic model scaffold.
    """

    if not 0.0 < eta_m <= 1.0:
        raise ValueError("eta_m must lie in (0, 1]")
    observed_state = int(observed_state)
    for state in list(distribution):
        distribution[state] *= 1.0 - eta_m
    distribution[observed_state] = distribution.get(observed_state, 0.0) + eta_m


def _distribution_delta_update(
    distribution: Distribution,
    target: Mapping[int, float],
    eta_m: float,
) -> None:
    """Delta update toward a full probe distribution rather than one sample."""

    target = normalized_distribution(target)
    states = set(distribution) | set(target)
    for state in states:
        old = distribution.get(state, 0.0)
        distribution[state] = old + eta_m * (target.get(state, 0.0) - old)


def jensen_shannon_divergence(
    left: Mapping[int, float], right: Mapping[int, float]
) -> float:
    """Base-2 Jensen--Shannon divergence in [0, 1]."""

    left = normalized_distribution(left)
    right = normalized_distribution(right)
    states = sorted(set(left) | set(right))
    if not states:
        return 0.0
    p = np.asarray([left.get(state, 0.0) for state in states], dtype=float)
    q = np.asarray([right.get(state, 0.0) for state in states], dtype=float)
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return float(np.clip(0.5 * kl(p) + 0.5 * kl(q), 0.0, 1.0))


def probe_distance_stochastic(
    fingerprint: Mapping[ProbeKey, Mapping[int, float]],
    probe: Mapping[ProbeKey, Mapping[int, float]],
) -> float:
    """Mean JSD across state-action keys observed in both fingerprints.

    Zero means identical transition distributions and one means maximally
    separated distributions. No shared probe keys are treated as maximally
    uninformative/distant rather than as an accidental match.
    """

    shared = sorted(set(fingerprint) & set(probe))
    if not shared:
        return 1.0
    return float(np.mean([
        jensen_shannon_divergence(fingerprint[key], probe[key]) for key in shared
    ]))


@dataclass
class ProbeFingerprint:
    distributions: Fingerprint
    observations: List[ProbeObservation]


def collect_probe_transitions_stochastic(
    env,
    n_probe: int,
    *,
    n_samples_per_step: int = 5,
    eta_m: float = 0.30,
    rng: Optional[np.random.Generator] = None,
) -> ProbeFingerprint:
    """Build a label-free stochastic transition fingerprint by delta learning.

    Every probe state-action pair is sampled repeatedly from the same state using
    Gymnasium's reset ``options={'state': ...}`` protocol. The first four keys
    test every action at the common start, making near-start wall cues observable.
    Later keys follow modal transitions and can reveal more distant geometry.

    In a human task these repeated micro-trials must be implemented explicitly;
    they are not cost-free observations of the transition kernel.
    """

    if n_probe < env.action_space.n:
        raise ValueError("n_probe must cover every start action at least once")
    if n_samples_per_step < 1:
        raise ValueError("n_samples_per_step must be positive")
    rng = np.random.default_rng() if rng is None else rng
    start_state, _ = env.reset()
    state = int(start_state)
    raw_distributions: Fingerprint = {}
    observations: List[ProbeObservation] = []

    for probe_step in range(int(n_probe)):
        if probe_step < env.action_space.n:
            state = int(start_state)
            action = probe_step
        else:
            action = int(rng.integers(env.action_space.n))

        key = (state, action)
        distribution = raw_distributions.setdefault(key, {})
        terminal_votes = 0
        step_next_states: List[int] = []
        for _ in range(int(n_samples_per_step)):
            sampled_state, _ = env.reset(options={"state": state})
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            observation = (
                int(sampled_state), action, int(next_state), float(reward), done
            )
            observations.append(observation)
            delta_rule_distribution_update(distribution, int(next_state), eta_m)
            step_next_states.append(int(next_state))
            terminal_votes += int(done)

        modal_next = max(
            normalized_distribution(distribution),
            key=normalized_distribution(distribution).get,
        )
        state = int(start_state) if terminal_votes > n_samples_per_step / 2 else int(modal_next)

    fingerprint = {
        key: normalized_distribution(distribution)
        for key, distribution in raw_distributions.items()
    }
    return ProbeFingerprint(fingerprint, observations)


def update_model_stochastic(
    model_transitions: Dict[ProbeKey, Distribution],
    model_rewards: Dict[ProbeKey, float],
    reward_counts: Dict[ProbeKey, int],
    state: int,
    action: int,
    next_state: int,
    reward: float,
    eta_m: float,
) -> None:
    """Delta-update ``P(s'|s,a)`` and running-mean reward ``R(s,a)``."""

    key = (int(state), int(action))
    distribution = model_transitions.setdefault(key, {})
    delta_rule_distribution_update(distribution, int(next_state), eta_m)
    count = reward_counts.get(key, 0) + 1
    old_reward = model_rewards.get(key, 0.0)
    model_rewards[key] = old_reward + (float(reward) - old_reward) / count
    reward_counts[key] = count


@dataclass
class StochasticContext:
    q_values: np.ndarray
    fingerprint: Fingerprint = field(default_factory=dict)
    model_transitions: Dict[ProbeKey, Distribution] = field(default_factory=dict)
    model_rewards: Dict[ProbeKey, float] = field(default_factory=dict)
    reward_counts: Dict[ProbeKey, int] = field(default_factory=dict)
    terminal_values: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    episodes: int = 0


@dataclass
class AgentConfig:
    eta: float = 0.30
    gamma: float = 0.95
    tau: float = 0.15
    crp_alpha: float = 0.50
    planning_steps: int = 5
    n_probe: int = 12
    probe_samples_per_step: int = 5
    probe_eta: float = 0.30
    model_eta: float = 0.30
    fingerprint_update_eta: float = 0.30
    transition_support: int = 5
    transition_floor: float = 1e-4
    assignment_mode: str = "sample"

    def validate(self) -> None:
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must lie in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.tau <= 0.0 or self.crp_alpha <= 0.0:
            raise ValueError("tau and crp_alpha must be positive")
        if self.planning_steps < 0 or self.n_probe < 4:
            raise ValueError("planning_steps must be nonnegative and n_probe at least 4")
        if self.probe_samples_per_step < 1:
            raise ValueError("probe_samples_per_step must be positive")
        for name, value in [
            ("probe_eta", self.probe_eta),
            ("model_eta", self.model_eta),
            ("fingerprint_update_eta", self.fingerprint_update_eta),
        ]:
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.transition_support < 2:
            raise ValueError("transition_support must be at least 2")
        if not 0.0 < self.transition_floor < 1.0:
            raise ValueError("transition_floor must lie in (0, 1)")
        if self.assignment_mode not in {"sample", "map"}:
            raise ValueError("assignment_mode must be 'sample' or 'map'")


@dataclass
class SequenceResult:
    returns: List[float]
    assignments: List[int]
    posteriors: List[np.ndarray]
    probe_distances: List[np.ndarray]
    probes: List[ProbeFingerprint]
    trajectories: List[List[TrajectoryStep]]


class StochasticSoftmaxCRPDyna:
    """Stochastic DINER with softmax choice and probabilistic Dyna planning."""

    def __init__(self, n_states: int, n_actions: int, config: AgentConfig, seed: int = 0):
        config.validate()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.contexts: List[StochasticContext] = []

    def new_context(self) -> StochasticContext:
        return StochasticContext(
            q_values=np.zeros((self.n_states, self.n_actions), dtype=float)
        )

    def collect_probe(self, env) -> ProbeFingerprint:
        return collect_probe_transitions_stochastic(
            env,
            self.config.n_probe,
            n_samples_per_step=self.config.probe_samples_per_step,
            eta_m=self.config.probe_eta,
            rng=self.rng,
        )

    def _probe_loglik(self, context: StochasticContext, probe: ProbeFingerprint) -> float:
        """Categorical probe likelihood used with the CRP prior.

        JSD and this likelihood are complementary: JSD is a scale-free distance
        diagnostic, while the likelihood supplies probabilities for Bayesian
        assignment. Unseen next states retain a small floor probability, so one
        slippery sample cannot create a hard contradiction.
        """

        loglik = 0.0
        samples = float(self.config.probe_samples_per_step)
        for key, observed in probe.distributions.items():
            learned = normalized_distribution(context.fingerprint.get(key, {}))
            if not learned:
                loglik -= samples * np.log(self.config.transition_support)
                continue
            states = set(learned) | set(observed)
            support = max(self.config.transition_support, len(states))
            for next_state, probability in observed.items():
                predictive = (
                    (1.0 - self.config.transition_floor) * learned.get(next_state, 0.0)
                    + self.config.transition_floor / support
                )
                loglik += samples * probability * np.log(predictive + 1e-300)
        return float(loglik)

    def _new_context_probe_loglik(self, probe: ProbeFingerprint) -> float:
        observations = len(probe.distributions) * self.config.probe_samples_per_step
        return float(-observations * np.log(self.config.transition_support))

    def context_posterior(
        self, probe: ProbeFingerprint
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return CRP posterior and graded JSD to every existing context."""

        distances = np.asarray([
            probe_distance_stochastic(context.fingerprint, probe.distributions)
            for context in self.contexts
        ], dtype=float)
        if not self.contexts:
            return np.ones(1, dtype=float), distances
        log_weights = [
            np.log(max(context.episodes, 1e-12)) + self._probe_loglik(context, probe)
            for context in self.contexts
        ]
        log_weights.append(
            np.log(self.config.crp_alpha) + self._new_context_probe_loglik(probe)
        )
        return stable_softmax(np.asarray(log_weights, dtype=float)), distances

    def _update_context_fingerprint(
        self, context: StochasticContext, probe: ProbeFingerprint
    ) -> None:
        for key, target in probe.distributions.items():
            if key not in context.fingerprint:
                context.fingerprint[key] = dict(target)
            else:
                _distribution_delta_update(
                    context.fingerprint[key],
                    target,
                    self.config.fingerprint_update_eta,
                )
                context.fingerprint[key] = normalized_distribution(context.fingerprint[key])

    def select_context(
        self, probe: ProbeFingerprint
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        posterior, distances = self.context_posterior(probe)
        if not self.contexts:
            selected = 0
        elif self.config.assignment_mode == "map":
            selected = int(np.argmax(posterior))
        else:
            selected = int(self.rng.choice(len(posterior), p=posterior))
        if selected == len(self.contexts):
            self.contexts.append(self.new_context())
        context = self.contexts[selected]
        context.episodes += 1
        self._update_context_fingerprint(context, probe)
        return selected, posterior, distances

    def _update_model(
        self,
        context: StochasticContext,
        state: int,
        action: int,
        next_state: int,
        reward: float,
        terminal: bool,
    ) -> None:
        update_model_stochastic(
            context.model_transitions,
            context.model_rewards,
            context.reward_counts,
            state,
            action,
            next_state,
            reward,
            self.config.model_eta,
        )
        terminal_key = (int(state), int(action), int(next_state))
        previous = context.terminal_values.get(terminal_key, 0.0)
        context.terminal_values[terminal_key] = previous + self.config.model_eta * (
            float(terminal) - previous
        )

    def _q_update(
        self,
        context: StochasticContext,
        state: int,
        action: int,
        next_state: int,
        reward: float,
        terminal: bool,
    ) -> None:
        continuation = 0.0 if terminal else self.config.gamma * np.max(
            context.q_values[next_state]
        )
        target = reward + continuation
        context.q_values[state, action] += self.config.eta * (
            target - context.q_values[state, action]
        )

    def _planning(self, context: StochasticContext) -> None:
        keys = list(context.model_transitions)
        if not keys:
            return
        for _ in range(self.config.planning_steps):
            state, action = keys[int(self.rng.integers(len(keys)))]
            distribution = normalized_distribution(context.model_transitions[(state, action)])
            next_states = np.asarray(list(distribution), dtype=int)
            probabilities = np.asarray([distribution[item] for item in next_states], dtype=float)
            next_state = int(self.rng.choice(next_states, p=probabilities))
            reward = context.model_rewards.get((state, action), 0.0)
            terminal_probability = context.terminal_values.get((state, action, next_state), 0.0)
            self._q_update(
                context,
                state,
                action,
                next_state,
                reward,
                bool(self.rng.random() < terminal_probability),
            )

    def play_episode(self, env):
        if env.observation_space.n != self.n_states or env.action_space.n != self.n_actions:
            raise ValueError("all tasks must share the agent's state and action spaces")
        probe = self.collect_probe(env)
        assignment, posterior, distances = self.select_context(probe)
        context = self.contexts[assignment]
        for observation in probe.observations:
            self._update_model(context, *observation)

        state, _ = env.reset()
        trajectory: List[TrajectoryStep] = []
        total_return = 0.0
        done = False
        while not done:
            probabilities = action_softmax(context.q_values[state], self.config.tau)
            action = int(self.rng.choice(self.n_actions, p=probabilities))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            self._q_update(context, state, action, next_state, reward, done)
            self._update_model(context, state, action, next_state, reward, done)
            self._planning(context)
            trajectory.append((state, action, next_state, reward, done))
            total_return += reward
            state = next_state
        return float(total_return), assignment, posterior, distances, probe, trajectory

    def play_sequence(
        self, env_sequence: Sequence[str], envs: Mapping[str, object]
    ) -> SequenceResult:
        result = SequenceResult([], [], [], [], [], [])
        for task_name in env_sequence:
            episode = self.play_episode(envs[task_name])
            total_return, assignment, posterior, distances, probe, trajectory = episode
            result.returns.append(total_return)
            result.assignments.append(assignment)
            result.posteriors.append(posterior)
            result.probe_distances.append(distances)
            result.probes.append(probe)
            result.trajectories.append(trajectory)
        return result
