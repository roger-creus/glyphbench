"""Glue between glyphbench and prime-rl: custom advantage, custom loss,
and an orchestrator-side monkey-patch that exposes per-env metadata to
the advantage step.

See ``specs/2026-04-30-glyphbench-rl-training-design.md`` for design notes.
"""
