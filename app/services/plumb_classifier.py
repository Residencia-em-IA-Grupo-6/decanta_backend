"""
[LEGADO / FACHADA DE RETROCOMPATIBILIDADE]
Módulo de compatibilidade para código e testes existentes que importam de plumb_classifier.
Todo o motor epistêmico foi migrado oficialmente para o JevK5-Lite em app/services/decision_classifier.py.
"""
from app.services.decision_classifier import (
    DecisionBinaryResult,
    DecisionClaimResult,
    DecisionFusedResult,
    DecisionTopicResult,
    JevDecisionClassifier,
    PlumbBinaryResult,
    PlumbClaimResult,
    PlumbFusedResult,
    PlumbTopicClassifier,
    PlumbTopicResult,
    get_decision_classifier,
    get_plumb_classifier,
    get_plumb_classifier_engine,
)

__all__ = [
    "DecisionBinaryResult",
    "DecisionClaimResult",
    "DecisionFusedResult",
    "DecisionTopicResult",
    "JevDecisionClassifier",
    "PlumbBinaryResult",
    "PlumbClaimResult",
    "PlumbFusedResult",
    "PlumbTopicClassifier",
    "PlumbTopicResult",
    "get_decision_classifier",
    "get_plumb_classifier",
    "get_plumb_classifier_engine",
]
