"""
app/services/onnx_classifier.py

Serviço de classificação de tópicos e Gatekeeper de Saúde de alta performance baseado
no modelo DistilBERT quantizado dinamicamente em ONNX INT8 (129 MB).

Substitui o modelo pesado Plumb-4B (LLM 4B), executando em ~20ms por inferência em CPU
com consumo mínimo de memória e sem necessidade de GPUs dedicadas ou motores pesados.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

logger = logging.getLogger("factchkbr.services.onnx_classifier")

# Mapeamento oficial dos 8 temas em id2label para os nomes exibidos pelo FactChkBR
LABEL_TO_PORTUGUESE_MAP: dict[str, str] = {
    "saude": "Saúde",
    "politica_governo": "Política",
    "economia_negocios": "Economia",
    "esportes": "Esportes",
    "entretenimento_cultura": "Entretenimento",
    "ciencia_tecnologia": "Ciência e Tecnologia",
    "cotidiano_seguranca": "Cotidiano e Segurança",
    "conversa_geral": "Conversa Geral",
}

# Categorização de alto nível para compatibilidade com o Gatekeeper
PORTUGUESE_TO_GATEKEEPER_TOPIC: dict[str, str] = {
    "Saúde": "Saúde",
    "Política": "Política",
    "Economia": "Economia",
    "Esportes": "Esportes",
    "Entretenimento": "Entretenimento",
    "Ciência e Tecnologia": "Outros",
    "Cotidiano e Segurança": "Outros",
    "Conversa Geral": "Outros",
}

RELEASE_ASSET_URL = (
    "https://github.com/Residencia-em-IA-Grupo-6/FACTCK.BR-extended/releases/download/v1.0.0/classifier_int8.tar.gz"
)


class TopicResult(BaseModel):
    """Resultado da classificação temática com compatibilidade retroativa ao PlumbTopicResult."""

    topic: str = Field(
        ...,
        description="Tema classificado em português: Saúde, Política, Entretenimento, Esportes, Economia, Outros",
    )
    raw_topic: str = Field(..., description="Chave original predita pelo modelo (ex: saude, politica_governo, etc.)")
    confidence: float = Field(..., description="Confiança na predição do tópico dominante (0.0 a 1.0)")
    probabilities: dict[str, float] = Field(
        default_factory=dict, description="Distribuição de probabilidades por tema em português"
    )
    is_health: bool = Field(..., description="True se o tema for classificado como Saúde")
    health_probability: float = Field(..., description="Probabilidade normalizada atribuída ao tema Saúde")
    latency_ms: float = Field(default=0.0, description="Tempo gasto para inferência em milissegundos")


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class OnnxTopicClassifier:
    """
    Classificador Temático DistilBERT quantizado em ONNX INT8.
    Gerenciado como Singleton para reutilizar a sessão de inferência do ONNX Runtime.
    """

    _instance: Optional["OnnxTopicClassifier"] = None
    _session: Optional[ort.InferenceSession] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _id2label: Optional[Dict[int, str]] = None

    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Carrega a sessão ONNX e o tokenizer. Baixa o modelo da Release se necessário."""
        if OnnxTopicClassifier._session is not None and OnnxTopicClassifier._tokenizer is not None:
            return

        int8_model_path = self.model_dir / "classifier_int8.onnx"
        tokenizer_dir = self.model_dir / "tokenizer"
        id2label_path = self.model_dir / "id2label.json"

        # 1. Verificar se o modelo precisa ser baixado da release
        if not int8_model_path.exists():
            logger.info("Modelo ONNX INT8 não encontrado em '%s'. Baixando da Release oficial...", self.model_dir)
            self._download_and_extract_model(int8_model_path)

        # 2. Carregar id2label
        if id2label_path.exists():
            with open(id2label_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                OnnxTopicClassifier._id2label = {int(k): v for k, v in raw.items()}
        else:
            # Fallback padrão das 8 classes
            OnnxTopicClassifier._id2label = {
                0: "ciencia_tecnologia",
                1: "conversa_geral",
                2: "cotidiano_seguranca",
                3: "economia_negocios",
                4: "entretenimento_cultura",
                5: "esportes",
                6: "politica_governo",
                7: "saude",
            }

        # 3. Carregar Tokenizer
        tok_source = str(tokenizer_dir) if tokenizer_dir.exists() else "distilbert-base-multilingual-cased"
        logger.info("Carregando Tokenizer de '%s'...", tok_source)
        OnnxTopicClassifier._tokenizer = AutoTokenizer.from_pretrained(tok_source)

        # 4. Carregar ONNX Runtime Session com otimizações de CPU
        logger.info("Inicializando sessão ONNX Runtime INT8 de '%s'...", int8_model_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, os.cpu_count() or 1)

        OnnxTopicClassifier._session = ort.InferenceSession(
            str(int8_model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Classificador ONNX INT8 carregado com sucesso!")

    def _download_and_extract_model(self, target_onnx_path: Path) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        tar_path = self.model_dir / "classifier_int8.tar.gz"
        logger.info("Baixando asset de %s...", RELEASE_ASSET_URL)
        urllib.request.urlretrieve(RELEASE_ASSET_URL, tar_path)
        logger.info("Extraindo %s...", tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.model_dir)
        if tar_path.exists():
            tar_path.unlink()
        if not target_onnx_path.exists():
            raise FileNotFoundError(f"Falha ao extrair '{target_onnx_path}' da Release do GitHub.")

    def classify(self, text: str, max_length: int = 64, health_threshold: float = 0.50) -> TopicResult | None:
        """
        Classifica sincronamente o texto em um dos 8 temas via ONNX INT8.
        Retorna TopicResult compatível com o formato do Gatekeeper.
        """
        clean_text = text.strip()
        if not clean_text:
            return None

        self._ensure_loaded()
        if OnnxTopicClassifier._session is None or OnnxTopicClassifier._tokenizer is None:
            return None

        import time

        t0 = time.perf_counter()

        inputs = OnnxTopicClassifier._tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = OnnxTopicClassifier._session.run(None, ort_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs[0][0]
        probs = _softmax(logits)

        best_id = int(np.argmax(probs))
        id2label = OnnxTopicClassifier._id2label or {}
        raw_best_topic = id2label.get(best_id, "saude")
        confidence = float(probs[best_id])
        norm = clean_text.lower()

        # Calibração para casos de fronteira onde conversa_geral compete com tema institucional
        if raw_best_topic == "conversa_geral":
            pol_idx = [k for k, v in id2label.items() if v == "politica_governo"]
            eco_idx = [k for k, v in id2label.items() if v == "economia_negocios"]
            sau_idx = [k for k, v in id2label.items() if v == "saude"]

            if pol_idx and any(w in norm for w in ("lula", "bolsonaro", "eleições", "eleição", "governo", "congresso", "stf", "senado", "deputado", "partido", "ministro", "medida provisória", "decreto", "voto")):
                if float(probs[pol_idx[0]]) > 0.15:
                    raw_best_topic = "politica_governo"
                    confidence = float(probs[pol_idx[0]])
            elif eco_idx and any(w in norm for w in ("banco central", "selic", "taxa de juros", "inflação", "dólar", "ibovespa", "copom", "bolsa de valores", "mercado financeiro", "poupança", "copom")):
                if float(probs[eco_idx[0]]) > 0.08:
                    raw_best_topic = "economia_negocios"
                    confidence = float(probs[eco_idx[0]])
            elif sau_idx and any(w in norm for w in ("emagrec", "vacina", "remédio", "medicamento", "câncer", "doença", "sintoma", "infarto", "cura", "limão")):
                if float(probs[sau_idx[0]]) > 0.25:
                    raw_best_topic = "saude"
                    confidence = float(probs[sau_idx[0]])

        # Construir distribuição amigável em português
        portuguese_probs: dict[str, float] = {}
        health_prob = 0.0

        for idx, p in enumerate(probs):
            raw_lbl = id2label.get(idx, str(idx))
            pt_name = LABEL_TO_PORTUGUESE_MAP.get(raw_lbl, raw_lbl)
            portuguese_probs[pt_name] = round(float(p), 4)
            if raw_lbl == "saude":
                health_prob = float(p)

        # Determinar tópico no vocabulário do Gatekeeper
        best_pt_topic = LABEL_TO_PORTUGUESE_MAP.get(raw_best_topic, "Outros")
        gatekeeper_topic = PORTUGUESE_TO_GATEKEEPER_TOPIC.get(best_pt_topic, best_pt_topic)

        # Decisão de Saúde para o Gatekeeper
        is_health = bool(
            health_prob >= health_threshold
            or (raw_best_topic == "saude" and health_prob >= 0.35)
        )

        return TopicResult(
            topic=gatekeeper_topic,
            raw_topic=raw_best_topic,
            confidence=round(confidence, 4),
            probabilities=portuguese_probs,
            is_health=is_health,
            health_probability=round(health_prob, 4),
            latency_ms=round(latency_ms, 2),
        )

    async def classify_async(
        self, text: str, max_length: int = 64, health_threshold: float = 0.50
    ) -> TopicResult | None:
        """Classificação assíncrona não-bloqueante executada no thread pool padrão."""
        return await asyncio.to_thread(self.classify, text, max_length, health_threshold)


_default_onnx_classifier: Optional[OnnxTopicClassifier] = None


def get_onnx_classifier(model_dir: str | Path = "models") -> OnnxTopicClassifier:
    """Retorna a instância singleton do classificador ONNX INT8."""
    global _default_onnx_classifier
    if _default_onnx_classifier is None:
        _default_onnx_classifier = OnnxTopicClassifier(model_dir=model_dir)
    return _default_onnx_classifier
