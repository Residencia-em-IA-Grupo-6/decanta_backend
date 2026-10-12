"""
Serviço de decisão epistêmica e checagem factual baseado no modelo JevK5-Lite (alibiserikbay/JevK5-Lite).
Executa inferência determinística calibrada ("System 1") em CPU sem geração de texto livre e sem alucinações.
Avalia a veracidade de alegações confrontando com evidências jornalísticas e regulatórias checadas.
"""
import asyncio
import logging
import os
from typing import Any
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas.analysis import Verdict
from app.services.onnx_classifier import get_onnx_classifier, OnnxTopicResult

logger = logging.getLogger("factchkbr.services.decision_classifier")

TOPIC_MAP: dict[str, str] = {
    "health": "Saúde",
    "politics": "Política",
    "entertainment": "Entretenimento",
    "sports": "Esportes",
    "economy": "Economia",
    "other": "Outros",
}


class DecisionTopicResult(BaseModel):
    """Resultado da classificação de tópico (delegado ao classificador ONNX INT8 de alta performance)."""
    topic: str = Field(..., description="Tema classificado em português")
    raw_topic: str = Field(..., description="Chave original predita")
    confidence: float = Field(..., description="Confiança na predição do tópico dominante (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades normalizada")
    is_health: bool = Field(..., description="True se o tema for classificado como Saúde")
    health_probability: float = Field(..., description="Probabilidade atribuída ao tema Saúde")


class DecisionClaimResult(BaseModel):
    """Resultado da checagem epistêmica de uma alegação confrontada com evidências via JevK5-Lite."""
    statement: str = Field(..., description="Alegação factual avaliada")
    verdict: Verdict = Field(..., description="Veredito epistêmico emitido pelo JevK5-Lite")
    confidence: float = Field(..., description="Grau de certeza da decisão (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades por veredito")
    rationale_hint: str = Field(default="", description="Pista ou resumo do critério orientador da decisão")


class DecisionBinaryResult(BaseModel):
    """Resultado da checagem binária estrita (V ou F) emitida pelo JevK5-Lite."""
    statement: str = Field(..., description="Alegação factual avaliada")
    verdict: str = Field(..., description="Veredito binário: 'V' ou 'F'")
    confidence: float = Field(..., description="Grau de certeza da decisão (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Distribuição de probabilidades entre V e F")
    prob_true: float = Field(..., description="Probabilidade de ser Verdadeiro (V)")
    prob_false: float = Field(..., description="Probabilidade de ser Falso (F)")


class DecisionFusedResult(BaseModel):
    """Resultado unificado combinando Tema (ONNX) + Veredito (JevK5-Lite)."""
    statement: str = Field(..., description="Alegação factual avaliada")
    topic: str = Field(..., description="Tema dominante: Saúde, Política, Entretenimento, Esportes, Economia, Outros")
    topic_confidence: float = Field(..., description="Confiança na predição do tema")
    verdict: str = Field(..., description="Veredito binário dominante: 'V' ou 'F'")
    verdict_confidence: float = Field(..., description="Confiança no veredito binário")
    prob_v: float = Field(..., description="Probabilidade de ser Verdadeiro (V)")
    prob_f: float = Field(..., description="Probabilidade de ser Falso (F)")
    topic_probabilities: dict[str, float] = Field(default_factory=dict)
    raw_probabilities: dict[str, float] = Field(default_factory=dict)
    raw_choice: str = Field(default="outros")


# Aliases para retrocompatibilidade total com classes legadas (Plumb-4B / Laya)
PlumbTopicResult = DecisionTopicResult
PlumbClaimResult = DecisionClaimResult
PlumbBinaryResult = DecisionBinaryResult
PlumbFusedResult = DecisionFusedResult


class JevDecisionClassifier:
    """
    Motor Epistêmico de Decisão JevK5-Lite.
    Substituto determinístico do Plumb-4B (~437M parâmetros, ~323 MB RAM em CPU).
    Avalia afirmações fáticas contra registros jornalísticos e fontes regulatórias sem alucinações.
    """
    _model: Any = None
    _load_attempted: bool = False

    def __init__(
        self,
        model_path_or_name: str | None = None,
        threads: int | None = None,
        preload: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.model_path = model_path_or_name or self.settings.DECISION_MODEL_PATH
        self.threads = threads or self.settings.DECISION_THREADS
        if preload:
            self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        if JevDecisionClassifier._model is not None:
            return JevDecisionClassifier._model
        if JevDecisionClassifier._load_attempted and JevDecisionClassifier._model is None:
            return None

        JevDecisionClassifier._load_attempted = True
        try:
            from jevk5 import JevK5Lite

            target_path = self.model_path
            # Verifica se o modelo existe localmente na pasta models/jevk5_lite
            if not os.path.exists(target_path):
                alt_local = os.path.join(os.path.dirname(__file__), "..", "..", "models", "jevk5_lite")
                if os.path.exists(alt_local):
                    target_path = alt_local
                else:
                    # Fallback para o identificador no Hugging Face Hub
                    target_path = self.settings.DECISION_MODEL_NAME

            logger.info("Carregando motor epistêmico JevK5-Lite (%s, threads=%d)...", target_path, self.threads)
            JevDecisionClassifier._model = JevK5Lite.from_pretrained(target_path, threads=self.threads)
            logger.info("Motor epistêmico JevK5-Lite carregado com sucesso.")
            return JevDecisionClassifier._model
        except Exception as exc:
            logger.warning("Não foi possível carregar JevK5-Lite (%s): %s", type(exc).__name__, exc)
            JevDecisionClassifier._model = None
            return None

    def evaluate_claim(
        self,
        claim: str,
        evidences: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> DecisionClaimResult | None:
        """
        Avalia a veracidade epistêmica de uma alegação confrontando-a com as evidências coletadas.
        Retorna DecisionClaimResult ou None se o modelo estiver indisponível.
        """
        clean_claim = claim.strip()
        if not clean_claim:
            return None

        model = self._ensure_loaded()
        if model is None:
            return None

        evidence_lines: list[str] = []
        for e in (evidences or [])[:5]:
            if isinstance(e, dict):
                src = e.get("source_name") or e.get("source") or "Fonte"
                title = e.get("title", "")
                snippet = e.get("snippet") or e.get("finding") or ""
                rating = e.get("rating") or e.get("stance") or ""
                evidence_lines.append(f"- {src}: {title}. {snippet} (Classificação: {rating})")
            elif hasattr(e, "model_dump"):
                d = e.model_dump()
                src = d.get("source_name", "Fonte")
                title = d.get("title", "")
                snippet = d.get("snippet") or d.get("finding") or ""
                rating = d.get("rating") or d.get("stance") or ""
                evidence_lines.append(f"- {src}: {title}. {snippet} (Classificação: {rating})")

        if evidence_lines:
            text_input = f"Alegação: {clean_claim}\nEvidências fáticas e registros checados:\n" + "\n".join(evidence_lines)
        else:
            text_input = f"Alegação: {clean_claim}"

        try:
            tasks = {"veredicto": ["falso", "verdadeiro", "suspeito"]}
            res = model.classify(text_input, tasks)
            ver_data = res.get("veredicto", {})
            labels = ver_data.get("labels", ["suspeito"])
            raw_choice = labels[0].lower() if labels else "suspeito"
            raw_probs = ver_data.get("probabilities", {})

            verdict_map = {
                "falso": Verdict.FAKE,
                "verdadeiro": Verdict.VERDADEIRO,
                "suspeito": Verdict.SUSPEITO,
            }
            verdict = verdict_map.get(raw_choice, Verdict.SUSPEITO)
            confidence = float(raw_probs.get(raw_choice, 0.70))

            probs = {
                "fake": round(float(raw_probs.get("falso", 0.0)), 4),
                "verdadeiro": round(float(raw_probs.get("verdadeiro", 0.0)), 4),
                "suspeito": round(float(raw_probs.get("suspeito", 0.0)), 4),
            }

            return DecisionClaimResult(
                statement=clean_claim,
                verdict=verdict,
                confidence=round(confidence, 4),
                probabilities=probs,
                rationale_hint=f"Decisão JevK5-Lite ({raw_choice}, conf: {confidence:.2f}) com {len(evidence_lines)} evidência(s).",
            )
        except Exception as exc:
            logger.warning("Falha na inferência do JevK5-Lite (%s): %s", type(exc).__name__, exc)
            return None

    async def evaluate_claim_async(
        self,
        claim: str,
        evidences: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> DecisionClaimResult | None:
        """Executa a checagem epistêmica de forma assíncrona em threadpool (sem travar event loop)."""
        return await asyncio.to_thread(self.evaluate_claim, claim, evidences, context)

    def evaluate_claim_binary(
        self,
        claim: str,
        context: str | None = None,
    ) -> DecisionBinaryResult | None:
        """Avalia a veracidade de forma estritamente binária (V ou F) com JevK5-Lite."""
        clean_claim = claim.strip()
        if not clean_claim:
            return None

        model = self._ensure_loaded()
        if model is None:
            return None

        text_input = f"Alegação: {clean_claim}" + (f"\nContexto: {context.strip()[:600]}" if context else "")

        try:
            tasks = {"veredicto": ["falso", "verdadeiro"]}
            res = model.classify(text_input, tasks)
            ver_data = res.get("veredicto", {})
            labels = ver_data.get("labels", ["falso"])
            raw_choice = labels[0].lower() if labels else "falso"
            raw_probs = ver_data.get("probabilities", {})

            p_f = round(float(raw_probs.get("falso", 0.5)), 4)
            p_v = round(float(raw_probs.get("verdadeiro", 0.5)), 4)

            verdict_str = "V" if raw_choice == "verdadeiro" else "F"
            confidence = p_v if verdict_str == "V" else p_f

            return DecisionBinaryResult(
                statement=clean_claim,
                verdict=verdict_str,
                confidence=round(confidence, 4),
                prob_true=p_v,
                prob_false=p_f,
                probabilities={"V": p_v, "F": p_f},
            )
        except Exception as exc:
            logger.warning("Falha na avaliação binária do JevK5-Lite (%s): %s", type(exc).__name__, exc)
            return None

    async def evaluate_claim_binary_async(
        self,
        claim: str,
        context: str | None = None,
    ) -> DecisionBinaryResult | None:
        return await asyncio.to_thread(self.evaluate_claim_binary, claim, context)

    def classify(self, text: str) -> DecisionTopicResult | None:
        """Classifica o tema utilizando o classificador ONNX INT8 de alta performance (~19ms)."""
        clean_text = text.strip()
        if not clean_text:
            return None

        try:
            onnx_clf = get_onnx_classifier()
            res: OnnxTopicResult | None = onnx_clf.classify(clean_text)
            if res is not None:
                return DecisionTopicResult(
                    topic=res.topic,
                    raw_topic=res.raw_topic,
                    confidence=res.confidence,
                    probabilities=res.probabilities,
                    is_health=res.is_health,
                    health_probability=res.health_probability,
                )
        except Exception as e:
            logger.debug("Falha na classificação temática ONNX: %s", e)
        return None

    async def classify_async(self, text: str) -> DecisionTopicResult | None:
        return await asyncio.to_thread(self.classify, text)

    def decide_fused(self, claim: str) -> DecisionFusedResult | None:
        """Combina predição de Tema (ONNX) e Veredito V/F (JevK5-Lite)."""
        topic_res = self.classify(claim)
        bin_res = self.evaluate_claim_binary(claim)

        topic = topic_res.topic if topic_res else "Outros"
        topic_conf = topic_res.confidence if topic_res else 0.5
        topic_probs = topic_res.probabilities if topic_res else {}

        verdict = bin_res.verdict if bin_res else "F"
        verdict_conf = bin_res.confidence if bin_res else 0.5
        p_v = bin_res.prob_true if bin_res else 0.5
        p_f = bin_res.prob_false if bin_res else 0.5

        return DecisionFusedResult(
            statement=claim,
            topic=topic,
            topic_confidence=topic_conf,
            verdict=verdict,
            verdict_confidence=verdict_conf,
            prob_v=p_v,
            prob_f=p_f,
            topic_probabilities=topic_probs,
            raw_probabilities={"v": p_v, "f": p_f},
            raw_choice=f"{topic.lower()}_{verdict.lower()}",
        )

    def batch_decide_fused(self, claims: list[str]) -> list[DecisionFusedResult | None]:
        """Classifica lote de alegações combinando predição de Tema (ONNX) e Veredito V/F (JevK5-Lite)."""
        return [self.decide_fused(c) for c in claims]

    def batch_evaluate_claim_binary(
        self,
        claims: list[str],
        contexts: list[str | None] | None = None,
    ) -> list[DecisionBinaryResult | None]:
        """Avalia lote com veredito binário (V ou F) via JevK5-Lite."""
        ctx_list = contexts or ([None] * len(claims))
        return [self.evaluate_claim_binary(c, ctx) for c, ctx in zip(claims, ctx_list)]

    def decompose_news(self, text: str, max_claims: int = 5) -> list[str]:
        """Decompõe o texto de uma notícia em alegações atômicas diretas."""
        clean = text.strip()[:2500]
        if not clean:
            return []
        import re
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        claims = [s.strip() for s in sentences if len(s.strip()) >= 20 and not s.strip().endswith("?")]
        return claims[:max_claims] if claims else [clean[:200]]


# Singleton
_default_decision_classifier: JevDecisionClassifier | None = None


def get_decision_classifier() -> JevDecisionClassifier:
    """Retorna a instância singleton do classificador de decisão JevK5-Lite."""
    global _default_decision_classifier
    if _default_decision_classifier is None:
        _default_decision_classifier = JevDecisionClassifier()
    return _default_decision_classifier


# Retrocompatibilidade com referências a get_plumb_classifier
PlumbTopicClassifier = JevDecisionClassifier
get_plumb_classifier = get_decision_classifier


def get_plumb_classifier_engine(engine: str = "auto") -> Any:
    return get_decision_classifier()
