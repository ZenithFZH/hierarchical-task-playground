import numpy as np

from environments import create_custom_environment
from recovery import (
    compute_bic,
    compute_cluster_assignments,
    simulate_diner,
    simulate_fresh_start,
    simulate_persistent,
    softmax_probs,
)


ENV_CONFIGS = {
    "right_goal": """
    A . . G
    # # . #
    . . . .
    . . . .
    """,
    "down_goal": """
    A # . .
    . # . .
    . . . .
    G . . .
    """,
}


def build_envs():
    return {
        name: create_custom_environment(ascii_map=layout, max_steps=40).build()
        for name, layout in ENV_CONFIGS.items()
    }


def test_softmax_is_a_probability_distribution():
    probs = softmax_probs(np.array([0.2, 0.8, 0.3, 0.5]), tau=0.5)
    assert probs.shape == (4,)
    assert np.all(probs > 0)
    assert np.isclose(probs.sum(), 1.0)


def test_cluster_assignment_and_all_simulators_run():
    envs = build_envs()
    sequence = ["right_goal", "down_goal", "right_goal", "down_goal"]

    assignments = compute_cluster_assignments(
        sequence, envs, crp_alpha=1.0, n_probe=4, seed=7
    )
    assert len(assignments) == len(sequence)
    assert len(set(assignments)) == 2

    diner_returns, diner_assignments, diner_trajs = simulate_diner(
        sequence,
        envs,
        crp_alpha=1.0,
        eta=0.3,
        gamma=0.95,
        tau=0.5,
        planning_steps=1,
        n_probe=4,
        seed=7,
    )
    fresh_returns, fresh_trajs = simulate_fresh_start(
        sequence, envs, eta=0.3, gamma=0.95, tau=0.5, planning_steps=1, seed=7
    )
    persistent_returns, persistent_trajs = simulate_persistent(
        sequence, envs, eta=0.3, gamma=0.95, tau=0.5, planning_steps=1, seed=7
    )

    assert len(diner_returns) == len(diner_assignments) == len(diner_trajs) == 4
    assert len(fresh_returns) == len(fresh_trajs) == 4
    assert len(persistent_returns) == len(persistent_trajs) == 4
    assert all(len(traj) > 0 for traj in diner_trajs + fresh_trajs + persistent_trajs)


def test_bic_penalizes_an_extra_parameter_at_equal_fit():
    bic_three = compute_bic(negloglik=100.0, n_params=3, n_steps=500)
    bic_four = compute_bic(negloglik=100.0, n_params=4, n_steps=500)
    assert bic_four > bic_three

