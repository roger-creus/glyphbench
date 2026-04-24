"""Verifiers integration — entry point for vf-eval and prime-rl."""

from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser

# Uncommented in Task 1.11 after the dependent modules exist:
# from glyphbench.verifiers_integration.env import (
#     GlyphbenchMultiTurnEnv,
#     load_environment,
# )
# from glyphbench.verifiers_integration.prompting import (
#     build_system_prompt,
#     render_user_turn,
# )
# from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

__all__ = ["GlyphbenchXMLParser"]
