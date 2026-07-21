import json
import numpy as np
import hierarchical_gridworld as gridworld_module

from hierarchical_gridworld import (
    HierarchicalGridWorld,
    TaskSpec,
    audit_task_family_difficulty,
    load_task_family,
    make_task_environments,
    make_default_task_family,
    optimal_policy_trace,
    save_task_family,
    validate_task_family,
)
from task_designer import TaskFamilyEditor
from recovery import simulate_diner, simulate_fresh_start, simulate_persistent


def test_default_family_has_matched_interface_distinct_walls_and_crossed_factors():
    specs = make_default_task_family()
    summary = validate_task_family(specs)
    assert summary["n_tasks"] == 4
    assert summary["interface_matched"]
    assert not summary["walls_matched"]
    assert summary["n_distinct_wall_layouts"] == 4
    assert summary["n_distinct_outcome_layouts"] == 2
    assert summary["size"] == 10
    assert all(count == 23 for count in summary["n_walls"].values())
    assert {spec.transition_mode for spec in specs.values()} == {
        "deterministic", "stochastic"
    }
    assert {spec.objective_mode for spec in specs.values()} == {
        "gain", "loss_avoidance"
    }


def test_deterministic_and_stochastic_transition_kernels():
    specs = make_default_task_family()
    deterministic = HierarchicalGridWorld(specs["deterministic_gain"])
    stochastic = HierarchicalGridWorld(specs["stochastic_gain"])
    state, _ = deterministic.reset(seed=1)
    stochastic.reset(seed=1)

    deterministic_probs = deterministic.action_execution_probs(0)
    stochastic_probs = stochastic.action_execution_probs(0)
    assert np.allclose(deterministic_probs, [1.0, 0.0, 0.0, 0.0])
    assert np.isclose(stochastic_probs.sum(), 1.0)
    assert np.isclose(stochastic_probs[0], 0.8)
    state_distribution = stochastic.transition_distribution(state, 0)
    intended_next_state = stochastic.move(state, 0)
    assert np.isclose(sum(state_distribution.values()), 1.0)
    assert np.isclose(state_distribution[intended_next_state], 0.8)


def test_slip_probability_can_be_overridden_at_environment_initialization():
    spec = make_default_task_family()["deterministic_gain"]
    env = HierarchicalGridWorld(spec, slip_prob=0.30)
    assert np.allclose(env.action_execution_probs(0), [0.70, 0.10, 0.10, 0.10])
    envs = make_task_environments(
        {"task": spec}, slip_prob_overrides={"task": 0.15}
    )
    assert np.isclose(envs["task"].action_execution_probs(0)[0], 0.85)


def test_hidden_outcome_is_collected_once():
    spec = TaskSpec(
        name="tiny",
        size=3,
        start=(2, 1),
        terminals={(0, 1)},
        outcomes={(2, 2): 0.5},
        transition_mode="deterministic",
        objective_mode="gain",
        step_cost=-0.01,
        collect_once=True,
    )
    env = HierarchicalGridWorld(spec)
    start_state, _ = env.reset(seed=2)
    first_state, first_reward, _, _, first_info = env.step(1)
    env.step(3)
    _, second_reward, _, _, second_info = env.step(1)
    assert np.isclose(first_reward, 0.49)
    assert np.isclose(second_reward, -0.01)
    assert np.isclose(first_info["revealed_outcome"], 0.5)
    assert np.isclose(second_info["revealed_outcome"], 0.0)
    assert env.n_states == 3 * 3 * 2
    assert first_state != env.position_to_state((2, 2), collected=set())
    assert start_state < env.n_states


def test_task_family_round_trip(tmp_path):
    specs = make_default_task_family()
    path = save_task_family(
        specs, tmp_path / "family.json", compute_difficulty_audit=True
    )
    loaded = load_task_family(path)
    payload = json.loads(path.read_text())
    assert list(loaded) == list(specs)
    assert all(loaded[name].to_dict() == specs[name].to_dict() for name in specs)
    assert payload["difficulty_audit"]["acceptable"]


def test_editor_fast_save_is_independent_from_difficulty_audit(tmp_path, monkeypatch):
    editor = TaskFamilyEditor(make_default_task_family(), "deterministic_gain")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("difficulty audit must not run during fast save")

    monkeypatch.setattr(
        gridworld_module, "audit_task_family_difficulty", fail_if_called
    )
    path = editor.save(tmp_path / "fast_family.json")
    payload = json.loads(path.read_text())
    assert payload["difficulty_audit"] is None
    assert len(payload["tasks"]) == 4


def test_validator_allows_independent_layouts_with_shared_state_encoding():
    specs = make_default_task_family()
    specs["stochastic_gain"].start = (9, 9)
    specs["stochastic_gain"].terminals = {(0, 4)}
    specs["stochastic_gain"].max_steps = 90
    specs["stochastic_gain"].outcomes = {(5, 7): 0.4}
    shared_inventory = set().union(*(set(spec.outcomes) for spec in specs.values()))
    for spec in specs.values():
        spec.inventory_positions = set(shared_inventory)
        spec.validate()

    summary = validate_task_family(specs)
    assert summary["interface_matched"]
    assert not summary["start_matched"]
    assert not summary["terminals_matched"]
    assert not summary["horizon_matched"]
    assert summary["n_distinct_outcome_layouts"] == 3
    envs = make_task_environments(specs, seed=12)
    assert len({env.observation_space.n for env in envs.values()}) == 1
    assert len({env.action_space.n for env in envs.values()}) == 1


def test_default_known_mdp_difficulty_is_comparable():
    audit = audit_task_family_difficulty(make_default_task_family())
    assert audit["acceptable"]
    assert all(item["comparable"] for item in audit["comparisons"])
    assert all(
        metrics["optimal_exit_probability"] >= 0.8
        for metrics in audit["tasks"].values()
    )
    assert all(metrics["optimal_trace"]["reaches_exit"] for metrics in audit["tasks"].values())


def test_optimal_trace_distinguishes_deterministic_path_from_stochastic_summary():
    specs = make_default_task_family()
    deterministic = optimal_policy_trace(specs["deterministic_gain"])
    stochastic = optimal_policy_trace(specs["stochastic_gain"])
    assert deterministic["trace_kind"] == "deterministic_optimal_path"
    assert stochastic["trace_kind"].startswith("nominal_intended_path")
    assert deterministic["positions"][0] == specs["deterministic_gain"].start
    assert stochastic["positions"][-1] in specs["stochastic_gain"].terminals


def test_editor_edits_every_layout_field_only_in_active_task_by_default():
    editor = TaskFamilyEditor(make_default_task_family(), "deterministic_gain")

    editor.apply_tool((9, 9), "start")
    assert editor.specs["deterministic_gain"].start == (9, 9)
    assert all(
        spec.start == (9, 5)
        for name, spec in editor.specs.items()
        if name != "deterministic_gain"
    )

    editor.apply_tool((0, 4), "exit")
    assert editor.specs["deterministic_gain"].terminals == {(0, 4)}
    assert all(
        spec.terminals == {(0, 5)}
        for name, spec in editor.specs.items()
        if name != "deterministic_gain"
    )

    editor.apply_tool((9, 0), "wall")
    assert (9, 0) in editor.specs["deterministic_gain"].walls
    assert all(
        (9, 0) not in spec.walls
        for name, spec in editor.specs.items()
        if name != "deterministic_gain"
    )

    editor.apply_tool((5, 3), "outcome", 0.9)
    assert editor.specs["deterministic_gain"].outcomes[(5, 3)] == 0.9
    assert all(
        (5, 3) not in spec.outcomes
        for name, spec in editor.specs.items()
        if name != "deterministic_gain"
    )
    assert all((5, 3) in spec.inventory_positions for spec in editor.specs.values())

    editor.set_step_cost(-0.03)
    assert editor.specs["deterministic_gain"].step_cost == -0.03
    assert all(
        spec.step_cost == -0.01
        for name, spec in editor.specs.items()
        if name != "deterministic_gain"
    )

    editor.active_name = "stochastic_gain"
    editor.set_stochastic_slip(0.35)
    assert editor.specs["stochastic_gain"].slip_prob == 0.35
    assert editor.specs["stochastic_loss"].slip_prob == 0.20

    assert validate_task_family(editor.specs)["interface_matched"]


def test_editor_links_only_when_requested_and_keeps_shared_state_encoding():
    editor = TaskFamilyEditor(make_default_task_family(), "deterministic_gain")

    editor.link_positions_across_tasks = True
    editor.apply_tool((9, 9), "start")
    assert all(spec.start == (9, 9) for spec in editor.specs.values())

    editor.link_walls_across_tasks = True
    editor.apply_tool((9, 1), "wall")
    assert all((9, 1) in spec.walls for spec in editor.specs.values())

    editor.link_outcomes_by_objective = True
    editor.apply_tool((5, 3), "outcome", 0.9)
    gain_specs = [
        spec for spec in editor.specs.values() if spec.objective_mode == "gain"
    ]
    loss_specs = [
        spec for spec in editor.specs.values()
        if spec.objective_mode == "loss_avoidance"
    ]
    assert all(spec.outcomes[(5, 3)] == 0.9 for spec in gain_specs)
    assert all((5, 3) not in spec.outcomes for spec in loss_specs)
    assert all((5, 3) in spec.inventory_positions for spec in editor.specs.values())

    editor.active_name = "stochastic_gain"
    editor.link_slip_by_transition = True
    editor.set_stochastic_slip(0.35)
    assert editor.specs["stochastic_gain"].slip_prob == 0.35
    assert editor.specs["stochastic_loss"].slip_prob == 0.35

    editor.link_step_cost_across_tasks = True
    editor.set_step_cost(-0.03)
    assert all(spec.step_cost == -0.03 for spec in editor.specs.values())

    editor.link_outcomes_by_objective = False
    editor.active_name = "deterministic_gain"
    editor.apply_tool((5, 3), "clear outcome")
    assert (5, 3) not in editor.specs["deterministic_gain"].outcomes
    assert (5, 3) in editor.specs["stochastic_gain"].outcomes
    assert all((5, 3) in spec.inventory_positions for spec in editor.specs.values())


def test_reference_agents_can_run_without_interface_changes():
    specs = make_default_task_family()
    for spec in specs.values():
        spec.max_steps = 8
    envs = make_task_environments(specs, seed=20)
    sequence = list(specs)

    fresh, _ = simulate_fresh_start(
        sequence, envs, eta=0.3, gamma=0.9, tau=0.5, planning_steps=1, seed=21
    )
    persistent, _ = simulate_persistent(
        sequence, envs, eta=0.3, gamma=0.9, tau=0.5, planning_steps=1, seed=21
    )
    diner, assignments, _ = simulate_diner(
        sequence, envs, crp_alpha=0.5, eta=0.3, gamma=0.9,
        tau=0.5, planning_steps=1, n_probe=4, seed=21,
    )

    assert len(fresh) == len(persistent) == len(diner) == len(sequence)
    assert len(assignments) == len(sequence)
    _, info = envs[sequence[0]].reset(seed=22)
    assert "task_name" not in info
    assert "transition_context" not in info
