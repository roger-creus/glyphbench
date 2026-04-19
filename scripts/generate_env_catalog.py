#!/usr/bin/env python3
"""Generate environment catalog from the registry."""

import atlas_rl  # noqa: F401
import gymnasium as gym


def main() -> None:
    envs = sorted(
        eid for eid in gym.registry
        if isinstance(eid, str) and eid.startswith("atlas_rl/")
    )

    suites: dict[str, list[str]] = {}
    for eid in envs:
        suite = eid.split("/")[1].split("-")[0]
        suites.setdefault(suite, []).append(eid)

    lines = [
        "# ATLAS Environment Catalog",
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
                env = gym.make(eid, max_turns=10)
                unwrapped = env.unwrapped
                n_actions = unwrapped.action_spec.n if hasattr(unwrapped, 'action_spec') else '?'
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
