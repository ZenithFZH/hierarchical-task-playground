"""Interactive Jupyter editor for a controlled hierarchical gridworld family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import matplotlib.pyplot as plt

from hierarchical_gridworld import (
    Position,
    TaskSpec,
    audit_task_family_difficulty,
    plot_task,
    save_task_family,
    validate_task_family,
)


def clone_specs(specs: Mapping[str, TaskSpec]) -> Dict[str, TaskSpec]:
    return {name: TaskSpec.from_dict(spec.to_dict()) for name, spec in specs.items()}


@dataclass
class TaskFamilyEditor:
    """Widget-independent editing logic, kept separate for testing and reuse."""

    specs: Dict[str, TaskSpec]
    active_name: str
    link_positions_across_tasks: bool = False
    link_outcomes_by_objective: bool = False
    link_walls_across_tasks: bool = False
    link_slip_by_transition: bool = False
    link_step_cost_across_tasks: bool = False

    def __init__(
        self,
        specs: Mapping[str, TaskSpec],
        active_name: Optional[str] = None,
        link_positions_across_tasks: bool = False,
        link_outcomes_by_objective: bool = False,
        link_walls_across_tasks: bool = False,
        link_slip_by_transition: bool = False,
        link_step_cost_across_tasks: bool = False,
    ) -> None:
        self.specs = clone_specs(specs)
        self.active_name = active_name or next(iter(self.specs))
        self.link_positions_across_tasks = bool(link_positions_across_tasks)
        self.link_outcomes_by_objective = bool(link_outcomes_by_objective)
        self.link_walls_across_tasks = bool(link_walls_across_tasks)
        self.link_slip_by_transition = bool(link_slip_by_transition)
        self.link_step_cost_across_tasks = bool(link_step_cost_across_tasks)
        if self.active_name not in self.specs:
            raise KeyError(self.active_name)

    @property
    def active(self) -> TaskSpec:
        return self.specs[self.active_name]

    def position_targets(self):
        """Edit only the active task unless position linking is explicit."""

        return self.specs.values() if self.link_positions_across_tasks else [self.active]

    def wall_targets(self):
        """Edit only the active maze unless the user explicitly links walls."""

        return self.specs.values() if self.link_walls_across_tasks else [self.active]

    def outcome_targets(self):
        if not self.link_outcomes_by_objective:
            return [self.active]
        objective = self.active.objective_mode
        return [spec for spec in self.specs.values() if spec.objective_mode == objective]

    def set_start(self, position: Position) -> None:
        for spec in self.position_targets():
            spec.walls.discard(position)
            spec.terminals.discard(position)
            spec.start = position
            spec.validate()

    def set_single_exit(self, position: Position) -> None:
        for spec in self.position_targets():
            spec.walls.discard(position)
            spec.terminals = {position}
            if spec.start == position:
                raise ValueError("the exit cannot overlap the start")
            spec.validate()

    def toggle_wall(self, position: Position) -> None:
        if position == self.active.start or position in self.active.terminals:
            raise ValueError("start and exit cells cannot be walls")
        targets = list(self.wall_targets())
        make_wall = position not in self.active.walls
        if make_wall:
            conflicts = [spec.name for spec in targets if position in spec.outcomes]
            if conflicts:
                raise ValueError(
                    "clear the outcome before adding a wall in: " + ", ".join(conflicts)
                )
        for spec in targets:
            if make_wall:
                spec.walls.add(position)
            else:
                spec.walls.discard(position)
            spec.validate()

    def set_outcome(self, position: Position, value: float) -> None:
        targets = list(self.outcome_targets())
        conflicts = [spec.name for spec in targets if position in spec.walls]
        if conflicts:
            raise ValueError(
                "the linked outcome cell is a wall in: " + ", ".join(conflicts)
            )
        for spec in targets:
            spec.outcomes[position] = float(value)
        self._synchronize_inventory_positions()

    def clear_outcome(self, position: Position) -> None:
        for spec in self.outcome_targets():
            spec.outcomes.pop(position, None)
        self._synchronize_inventory_positions()

    def _synchronize_inventory_positions(self) -> None:
        """Keep state indices compatible without coupling actual reward maps."""

        shared_inventory = set().union(
            *(set(spec.outcomes) for spec in self.specs.values())
        )
        for spec in self.specs.values():
            spec.inventory_positions = set(shared_inventory)
            spec.validate()

    def set_stochastic_slip(self, slip_prob: float) -> None:
        if not 0.0 <= slip_prob < 1.0:
            raise ValueError("slip_prob must lie in [0, 1)")
        if self.active.transition_mode != "stochastic":
            raise ValueError("select a stochastic task before changing slip probability")
        targets = [self.active]
        if self.link_slip_by_transition:
            transition_context = self.active.transition_context
            targets = [
                spec for spec in self.specs.values()
                if spec.transition_context == transition_context
            ]
        for spec in targets:
            spec.slip_prob = float(slip_prob)
            spec.validate()

    def set_step_cost(self, step_cost: float) -> None:
        targets = (
            self.specs.values() if self.link_step_cost_across_tasks else [self.active]
        )
        for spec in targets:
            spec.step_cost = float(step_cost)
            spec.validate()

    def set_shared_step_cost(self, step_cost: float) -> None:
        """Explicit compatibility helper for callers that want global coupling."""

        for spec in self.specs.values():
            spec.step_cost = float(step_cost)
            spec.validate()

    def apply_tool(self, position: Position, tool: str, value: float = 0.0) -> None:
        if not self.active.in_bounds(position):
            raise ValueError(f"position outside grid: {position}")
        if tool == "inspect":
            return
        if tool == "start":
            self.set_start(position)
        elif tool == "exit":
            self.set_single_exit(position)
        elif tool == "wall":
            self.toggle_wall(position)
        elif tool == "outcome":
            self.set_outcome(position, value)
        elif tool == "clear outcome":
            self.clear_outcome(position)
        else:
            raise ValueError(f"unknown tool: {tool}")

    def validate(self) -> dict:
        return validate_task_family(self.specs)

    def save(self, path: str | Path) -> Path:
        """Run fast structural validation and save without a difficulty audit."""

        return save_task_family(
            self.specs, path, compute_difficulty_audit=False
        )

    def difficulty_audit(self, tolerance: float = 0.15) -> dict:
        return audit_task_family_difficulty(self.specs, tolerance=tolerance)

    def save_with_audit(self, path: str | Path, audit: dict) -> Path:
        return save_task_family(self.specs, path, difficulty_audit=audit)


class TaskDesigner:
    """An ipywidgets grid editor for four factor-crossed tasks.

    Every edit affects only the active scene by default. Optional checkboxes permit
    deliberate propagation of positions, walls, outcomes, slip, or step cost. The
    common inventory index is synchronized invisibly so edited tasks remain usable
    by agents that carry one discrete state encoding across the family.
    """

    def __init__(
        self,
        specs: Mapping[str, TaskSpec],
        *,
        export_path: str = "four_task_family.json",
    ) -> None:
        try:
            import ipywidgets as widgets
        except ImportError as error:  # pragma: no cover - exercised in notebooks
            raise ImportError(
                "TaskDesigner requires ipywidgets. Install it with: pip install ipywidgets"
            ) from error

        self.widgets = widgets
        self.editor = TaskFamilyEditor(specs)
        self._buttons: Dict[Position, object] = {}
        self._last_audit: Optional[dict] = None
        self._refreshing = False

        self.task_dropdown = widgets.Dropdown(
            options=list(self.editor.specs),
            value=self.editor.active_name,
            description="Task",
            layout=widgets.Layout(width="360px"),
        )
        self.factor_label = widgets.HTML()
        self.tool = widgets.ToggleButtons(
            options=["inspect", "start", "exit", "wall", "outcome", "clear outcome"],
            value="inspect",
            description="Tool",
            style={"button_width": "110px"},
        )
        self.outcome_value = widgets.FloatText(
            value=0.5, description="Outcome value", layout=widgets.Layout(width="220px")
        )
        self.link_outcomes = widgets.Checkbox(
            value=False,
            description="Link outcome edits by objective",
            indent=False,
            layout=widgets.Layout(width="300px"),
        )
        self.link_positions = widgets.Checkbox(
            value=False,
            description="Link start/exit across all tasks",
            indent=False,
            layout=widgets.Layout(width="300px"),
        )
        self.link_walls = widgets.Checkbox(
            value=False,
            description="Link wall edits across all tasks",
            indent=False,
            layout=widgets.Layout(width="300px"),
        )
        self.link_slip = widgets.Checkbox(
            value=False,
            description="Link slip by transition type",
            indent=False,
            layout=widgets.Layout(width="300px"),
        )
        self.slip_slider = widgets.FloatSlider(
            value=self.editor.active.slip_prob,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Slip probability",
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="360px"),
        )
        self.step_cost = widgets.FloatText(
            value=self.editor.active.step_cost, description="Step cost",
            layout=widgets.Layout(width="220px"),
        )
        self.link_step_cost = widgets.Checkbox(
            value=False,
            description="Link step cost across all tasks",
            indent=False,
            layout=widgets.Layout(width="300px"),
        )
        self.export_path = widgets.Text(
            value=export_path,
            description="Export JSON",
            layout=widgets.Layout(width="520px"),
        )
        self.save_button = widgets.Button(
            description="Validate and save (fast)", button_style="success"
        )
        self.preview_button = widgets.Button(description="Refresh preview")
        self.audit_tolerance = widgets.FloatSlider(
            value=0.15, min=0.02, max=0.40, step=0.01,
            description="Difficulty tolerance", readout_format=".2f",
            continuous_update=False, layout=widgets.Layout(width="360px"),
        )
        self.audit_button = widgets.Button(description="Audit difficulty")
        self.audit_result = widgets.HTML(
            value="<i>Difficulty audit not yet run for the current edits.</i>"
        )
        self.status = widgets.HTML(value="")
        self.preview = widgets.Output()

        self.grid = self._make_grid()
        self.root = widgets.VBox([
            widgets.HBox([self.task_dropdown, self.factor_label]),
            self.tool,
            widgets.HBox([self.outcome_value, self.link_outcomes]),
            widgets.HBox([self.link_positions, self.link_walls]),
            widgets.HBox([self.slip_slider, self.link_slip]),
            widgets.HBox([self.step_cost, self.link_step_cost]),
            self.grid,
            widgets.HBox([self.audit_tolerance, self.audit_button]),
            self.audit_result,
            widgets.HBox([self.export_path, self.save_button, self.preview_button]),
            self.status,
            self.preview,
        ])

        self.task_dropdown.observe(self._on_task_change, names="value")
        self.link_positions.observe(self._on_position_link_change, names="value")
        self.link_outcomes.observe(self._on_link_change, names="value")
        self.link_walls.observe(self._on_wall_link_change, names="value")
        self.link_slip.observe(self._on_slip_link_change, names="value")
        self.link_step_cost.observe(self._on_step_cost_link_change, names="value")
        self.slip_slider.observe(self._on_slip_change, names="value")
        self.step_cost.observe(self._on_step_cost_change, names="value")
        self.save_button.on_click(self._on_save)
        self.audit_button.on_click(self._on_audit)
        self.preview_button.on_click(lambda _: self.refresh_preview())
        self.refresh()

    def _make_grid(self):
        widgets = self.widgets
        children = []
        size = self.editor.active.size
        cell_size = max(36, min(54, 420 // size))
        for row in range(size):
            for column in range(size):
                position = (row, column)
                button = widgets.Button(
                    description="·",
                    layout=widgets.Layout(
                        width=f"{cell_size}px", height=f"{cell_size}px", padding="0"
                    ),
                    tooltip=str(position),
                )
                button.on_click(lambda _, pos=position: self._on_cell(pos))
                self._buttons[position] = button
                children.append(button)
        return widgets.GridBox(
            children,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({size}, {cell_size}px)",
                grid_gap="2px",
                width=f"{size * (cell_size + 2)}px",
            ),
        )

    def _cell_text(self, position: Position) -> str:
        spec = self.editor.active
        if position == spec.start:
            return "S"
        if position in spec.terminals:
            return "E"
        if position in spec.walls:
            return "#"
        if position in spec.outcomes:
            value = spec.outcomes[position]
            return f"{value:+.2g}"
        return "·"

    def refresh(self) -> None:
        spec = self.editor.active
        self._refreshing = True
        try:
            for position, button in self._buttons.items():
                button.description = self._cell_text(position)
                button.tooltip = f"{position}: {button.description}"
            self.slip_slider.disabled = spec.transition_mode != "stochastic"
            self.slip_slider.value = spec.slip_prob
            self.step_cost.value = spec.step_cost
            self.factor_label.value = (
                f"<b>{spec.transition_mode}</b> × <b>{spec.objective_mode}</b> "
                f"({spec.visible_theme})"
            )
        finally:
            self._refreshing = False
        self.refresh_preview()

    def refresh_preview(self) -> None:
        with self.preview:
            self.preview.clear_output(wait=True)
            show_path = self._last_audit is not None
            n_panels = 3 if show_path else 2
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
            plot_task(self.editor.active, axes[0], reveal_outcomes=True, title="Designer view")
            plot_task(self.editor.active, axes[1], reveal_outcomes=False, title="Participant view")
            if show_path:
                metrics = self._last_audit["tasks"][self.editor.active_name]
                trace = metrics["optimal_trace"]
                plot_task(
                    self.editor.active,
                    axes[2],
                    reveal_outcomes=True,
                    optimal_path=trace["positions"],
                    title=(
                        "Audited optimal path"
                        if self.editor.active.transition_mode == "deterministic"
                        else "Audited nominal optimal path"
                    ),
                )
            plt.tight_layout()
            plt.show()

    def _set_status(self, message: str, *, error: bool = False) -> None:
        color = "#b91c1c" if error else "#166534"
        self.status.value = f'<span style="color:{color}">{message}</span>'

    def _on_cell(self, position: Position) -> None:
        try:
            self.editor.apply_tool(position, self.tool.value, self.outcome_value.value)
            if self.tool.value != "inspect":
                self._mark_audit_stale()
            self._set_status(f"Applied {self.tool.value} at {position}.")
            self.refresh()
        except Exception as error:
            self._set_status(str(error), error=True)

    def _on_task_change(self, change) -> None:
        self.editor.active_name = change["new"]
        self.refresh()

    def _on_link_change(self, change) -> None:
        self.editor.link_outcomes_by_objective = bool(change["new"])
        scope = "same-objective tasks" if change["new"] else "only the active task"
        self._set_status(f"Outcome edits now affect {scope}.")

    def _on_position_link_change(self, change) -> None:
        self.editor.link_positions_across_tasks = bool(change["new"])
        scope = "all tasks" if change["new"] else "only the active task"
        self._set_status(f"Start/exit edits now affect {scope}.")

    def _on_wall_link_change(self, change) -> None:
        self.editor.link_walls_across_tasks = bool(change["new"])
        scope = "all tasks" if change["new"] else "only the active task"
        self._set_status(f"Wall edits now affect {scope}.")

    def _on_slip_link_change(self, change) -> None:
        self.editor.link_slip_by_transition = bool(change["new"])
        scope = "the same transition type" if change["new"] else "only the active task"
        self._set_status(f"Slip edits now affect {scope}.")

    def _on_step_cost_link_change(self, change) -> None:
        self.editor.link_step_cost_across_tasks = bool(change["new"])
        scope = "all tasks" if change["new"] else "only the active task"
        self._set_status(f"Step-cost edits now affect {scope}.")

    def _on_slip_change(self, change) -> None:
        if self._refreshing or self.editor.active.transition_mode != "stochastic":
            return
        try:
            self.editor.set_stochastic_slip(float(change["new"]))
            self._mark_audit_stale()
            self._set_status("Updated slip for tasks sharing this transition context.")
        except Exception as error:
            self._set_status(str(error), error=True)

    def _on_step_cost_change(self, change) -> None:
        if self._refreshing:
            return
        try:
            self.editor.set_step_cost(float(change["new"]))
            self._mark_audit_stale()
            scope = (
                "all tasks" if self.editor.link_step_cost_across_tasks
                else "the active task"
            )
            self._set_status(f"Updated step cost for {scope}.")
        except Exception as error:
            self._set_status(str(error), error=True)

    def _on_save(self, _) -> None:
        try:
            summary = self.editor.validate()
            path = self.editor.save(self.export_path.value)
            self._set_status(
                f"Saved {summary['n_tasks']} validated tasks to <code>{path}</code>; "
                "difficulty audit not run."
            )
        except Exception as error:
            self._set_status(str(error), error=True)

    def _on_audit(self, _) -> None:
        try:
            self._show_audit(
                self.editor.difficulty_audit(self.audit_tolerance.value)
            )
        except Exception as error:
            self._set_status(str(error), error=True)

    def _mark_audit_stale(self) -> None:
        self._last_audit = None
        self.audit_result.value = "<i>Difficulty audit is stale; run it again before saving.</i>"

    def _show_audit(self, audit: dict) -> None:
        self._last_audit = audit
        rows = []
        for name, metrics in audit["tasks"].items():
            rows.append(
                "<tr>"
                f"<td>{name}</td>"
                f"<td>{metrics['worst_return']:+.2f}</td>"
                f"<td>{metrics['random_return']:+.2f}</td>"
                f"<td>{metrics['optimal_return']:+.2f}</td>"
                f"<td>{metrics['optimal_exit_probability']:.1%}</td>"
                f"<td>{metrics['normalized_random_to_optimal_gap']:.2f}</td>"
                "</tr>"
            )
        failed = [
            f"{item['factor']} within {item['matched_on']}"
            for item in audit["comparisons"] if not item["comparable"]
        ]
        label = "ACCEPTABLE" if audit["acceptable"] else "REVISE"
        detail = "all matched comparisons pass" if not failed else "failed: " + ", ".join(failed)
        self.audit_result.value = (
            f"<p><b>Difficulty flag: {label}</b> ({detail}; tolerance={audit['tolerance']:.2f})</p>"
            "<table><thead><tr><th>Task</th><th>Worst</th><th>Random</th>"
            "<th>Optimal</th><th>Optimal exit</th><th>Normalized gap</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        self.refresh_preview()

    def display(self) -> None:
        from IPython.display import display

        display(self.root)
