#!/usr/bin/env python3
"""Generate environment catalog from the registry."""

import glyphbench  # noqa: F401

from glyphbench.core import all_glyphbench_env_ids, make_env


def main() -> None:
    envs = sorted(all_glyphbench_env_ids())

    suites: dict[str, list[str]] = {}
    for eid in envs:
        suite = eid.split("/")[1].split("-")[0]
        suites.setdefault(suite, []).append(eid)

    lines = [
        "# GlyphBench Environment Catalog",
        "",
        f"Total environments: {len(envs)}",
        "",
    ]

    for suite_name, suite_envs in sorted(suites.items()):
        lines.append(f"## {suite_name.title()} ({len(suite_envs)} environments)")
        lines.append("")
        lines.append("| Env ID | Actions |")
        lines.append("|--------|---------|")
        for eid in suite_envs:
            try:
                env = make_env(eid, max_turns=10)
                n_actions = env.action_spec.n if hasattr(env, 'action_spec') else '?'
                lines.append(f"| `{eid}` | {n_actions} |")
            except Exception:
                lines.append(f"| `{eid}` | ? |")
        lines.append("")

    output = "\n".join(lines)
    with open("docs/ENVIRONMENTS.md", "w") as f:
        f.write(output)
    print(f"Generated docs/ENVIRONMENTS.md with {len(envs)} environments")


if __name__ == "__main__":
    main()
