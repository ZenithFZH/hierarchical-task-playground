import numpy as np

from hierarchical_gridworld import make_task_environments, make_default_task_family
from stochastic_crp_dyna_agent import (
    AgentConfig,
    ProbeFingerprint,
    StochasticSoftmaxCRPDyna,
    action_softmax,
    delta_rule_distribution_update,
    jensen_shannon_divergence,
    normalized_distribution,
    probe_distance_stochastic,
)


def test_action_softmax_is_normalized_and_temperature_sensitive():
    q_values = np.array([0.0, 0.5, 1.0, -0.5])
    cold = action_softmax(q_values, 0.1)
    warm = action_softmax(q_values, 1.0)
    assert np.isclose(cold.sum(), 1.0)
    assert np.argmax(cold) == 2
    assert cold[2] > warm[2]


def test_stochastic_context_posterior_keeps_old_and_new_hypotheses_alive():
    config = AgentConfig(n_probe=4, assignment_mode="map")
    agent = StochasticSoftmaxCRPDyna(20, 4, config, seed=1)
    context = agent.new_context()
    context.episodes = 3
    agent.contexts.append(context)
    context.fingerprint[(0, 0)] = {1: 1.0}
    probe = ProbeFingerprint({(0, 0): {2: 1.0}}, [(0, 0, 2, -0.01, False)])
    posterior, distances = agent.context_posterior(probe)
    assert len(posterior) == 2
    assert np.isclose(posterior.sum(), 1.0)
    assert np.all(posterior > 0.0)
    assert np.allclose(distances, [1.0])


def test_probe_jsd_and_delta_rule_are_graded():
    learned = {}
    delta_rule_distribution_update(learned, 1, eta_m=0.5)
    delta_rule_distribution_update(learned, 2, eta_m=0.5)
    probabilities = normalized_distribution(learned)
    assert np.allclose([probabilities[1], probabilities[2]], [1 / 3, 2 / 3])
    assert jensen_shannon_divergence({1: 1.0}, {1: 1.0}) == 0.0
    assert np.isclose(jensen_shannon_divergence({1: 1.0}, {2: 1.0}), 1.0)
    distance = probe_distance_stochastic(
        {(0, 0): {1: 0.8, 2: 0.2}},
        {(0, 0): {1: 0.6, 2: 0.4}},
    )
    assert 0.0 < distance < 1.0


def test_crp_alpha_changes_new_cluster_posterior_weight():
    probe = ProbeFingerprint({(0, 0): {2: 1.0}}, [])
    new_weights = []
    for alpha in [0.1, 10.0]:
        agent = StochasticSoftmaxCRPDyna(
            20, 4, AgentConfig(n_probe=4, crp_alpha=alpha, assignment_mode="map"), seed=2
        )
        context = agent.new_context()
        context.episodes = 2
        context.fingerprint[(0, 0)] = {state: 0.2 for state in range(5)}
        agent.contexts.append(context)
        posterior, _ = agent.context_posterior(probe)
        new_weights.append(posterior[-1])
    assert new_weights[1] > new_weights[0]


def test_agent_plays_json_compatible_hierarchical_tasks():
    specs = make_default_task_family()
    for spec in specs.values():
        spec.max_steps = 15
    envs = make_task_environments(specs, seed=30)
    first_env = next(iter(envs.values()))
    config = AgentConfig(
        eta=0.3, gamma=0.9, tau=0.2, crp_alpha=0.5,
        planning_steps=1, n_probe=4, probe_samples_per_step=2,
        assignment_mode="map",
    )
    agent = StochasticSoftmaxCRPDyna(
        first_env.observation_space.n, first_env.action_space.n, config, seed=31
    )
    sequence = list(specs) * 2
    result = agent.play_sequence(sequence, envs)
    assert len(result.returns) == len(sequence)
    assert len(result.assignments) == len(sequence)
    assert len(result.posteriors) == len(sequence)
    assert len(result.probe_distances) == len(sequence)
    assert all(
        np.isclose(sum(distribution.values()), 1.0)
        for probe in result.probes
        for distribution in probe.distributions.values()
    )
    assert all(np.isfinite(value) for value in result.returns)
    assert len(agent.contexts) >= 1
