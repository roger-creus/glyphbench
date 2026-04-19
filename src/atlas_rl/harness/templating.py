"""Jinja2 environment for per-env system prompt templates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from atlas_rl.core.action import ActionSpec

_TEMPLATE_DIR = Path(__file__).parent / "system_prompts"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    undefined=StrictUndefined,
    autoescape=select_autoescape(),
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_system_prompt(
    template_stem: str,
    *,
    env_name: str,
    task_description: str,
    action_spec: ActionSpec,
    reward_description: str,
    symbol_legend_summary: str,
    physics_notes: str,
) -> str:
    """Render the template at system_prompts/<template_stem>.j2.

    `template_stem` is the gym env id without the "atlas_rl/" prefix;
    e.g. "__dummy-v0" for DummyEnv.
    """
    template = _env.get_template(f"{template_stem}.j2")
    return template.render(
        env_name=env_name,
        task_description=task_description,
        action_table=action_spec.render_for_prompt(),
        reward_description=reward_description,
        symbol_legend_summary=symbol_legend_summary,
        physics_notes=physics_notes,
    )
