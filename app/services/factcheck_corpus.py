"""
Serviço de Acervo de Checagens Fatuais e Artigos de Referência (FactCheck Corpus).
Indexa em memória os datasets consolidados de fact-checking do FactChkBR (TSVs),
fornecendo busca instantânea com links diretos para matérias, apurações e estudos,
garantindo que o usuário nunca receba páginas iniciais genéricas de portais.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any
import urllib.parse

from app.schemas.analysis import EvidenceItem, SourceTier

logger = logging.getLogger("factchkbr.factcheck_corpus")

HOMEPAGE_PATHS = {
    "", "/", "/estadao-verifica", "/estadao-verifica/", "/fato-ou-fake", "/fato-ou-fake/",
    "/confere", "/confere/", "/saude/pt-br", "/saude/pt-br/", "/anvisa/pt-br", "/anvisa/pt-br/",
    "/pt", "/pt/", "/pt-br", "/pt-br/", "/saude", "/saude/", "/anvisa", "/anvisa/",
}


def is_homepage_url(url: str) -> bool:
    """Verifica se uma URL aponta para a página inicial de um portal ou seção raiz genérica."""
    if not url or not url.startswith("http"):
        return True
    try:
        parsed = urllib.parse.urlparse(url.rstrip("/"))
        path = parsed.path.rstrip("/")
        if not path or path in HOMEPAGE_PATHS:
            return True
        parts = [p for p in path.split("/") if p]
        if not parts:
            return True
        # Seções raiz de checadores sem slug de matéria
        if parts[0] in ("estadao-verifica", "fato-ou-fake", "confere") and len(parts) <= 1:
            return True
        # Portais governamentais raiz sem matéria específica
        if "gov.br" in parsed.netloc:
            if parts in (["saude", "pt-br"], ["anvisa", "pt-br"], ["saude"], ["anvisa"], ["pt-br"]):
                return True
        return False
    except Exception:
        return True


STOP_WORDS = {
    "existe", "existem", "existir", "haver", "tem", "têm", "ter",
    "diz", "dizem", "afirma", "afirmam", "fala", "falam", "conta",
    "veja", "ver", "mostra", "revela", "promete", "ajuda", "serve",
    "cura", "curar", "trata", "tratar", "seca", "elimina", "perde", "emagrece",
    "vende", "compra", "beber", "bebe", "tomar", "toma", "comer", "come", "usar", "usa",
    "para", "com", "por", "que", "uma", "uns", "umas", "esse", "essa", "isso", "este", "esta", "isto",
    "sobre", "como", "mais", "muito", "todo", "toda", "todos", "todas", "pelo", "pela", "pelos", "pelas",
    "quando", "onde", "qual", "quais", "quem", "cujo", "cuja", "dele", "dela",
}

SYNONYMS = {
    "pressao": {"pressao", "pressão", "hipertensao", "hipertensão", "arterial", "cardiaca", "cardíaca"},
    "pressão": {"pressao", "pressão", "hipertensao", "hipertensão", "arterial", "cardiaca", "cardíaca"},
    "sal": {"sal", "sodio", "sódio", "salgado", "salgada", "cloreto"},
    "agua": {"agua", "água", "liquido", "líquido"},
    "água": {"agua", "água", "liquido", "líquido"},
}


class FactCheckCorpus:
    """Acervo em memória de checagens consolidadas com URLs diretas."""

    def __init__(self, datasets_dir: Path | None = None) -> None:
        if datasets_dir is None:
            datasets_dir = Path(__file__).resolve().parent.parent.parent / "datasets"
        self.datasets_dir = datasets_dir
        self.items: list[dict[str, Any]] = []
        self._loaded = False

    def load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        candidate_files = [
            "factckbr_boatos_saude_consolidado.tsv",
            "dataset_classificacao_temas.tsv",
            "alegacoes_dataset_consolidado.tsv",
        ]

        seen_urls: set[str] = set()

        for fname in candidate_files:
            file_path = self.datasets_dir / fname
            if not file_path.exists():
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        url = (row.get("URL") or row.get("url") or "").strip()
                        if not url or not url.startswith("http") or is_homepage_url(url):
                            continue
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                        title = (row.get("Titulo") or row.get("titulo") or "").strip()
                        claim = (row.get("Claim") or row.get("claim") or "").strip()
                        author = (row.get("Author") or row.get("author") or "Checador Oficial").strip()
                        is_fake_raw = str(row.get("is_fake", "")).lower()
                        is_fake = is_fake_raw in ("true", "1", "verdadeiro", "sim")

                        self.items.append({
                            "url": url,
                            "title": title or claim,
                            "claim": claim,
                            "author": author,
                            "date": (row.get("Data") or row.get("data") or "").strip(),
                            "is_fake": is_fake,
                            "search_text": f"{title} {claim}".lower(),
                        })
            except Exception as exc:
                logger.warning("Falha ao carregar dataset %s: %s", fname, exc)

        logger.info("FactCheckCorpus carregado com %d matérias com link direto.", len(self.items))

    def search(self, query: str, original_text: str | None = None, limit: int = 4) -> list[EvidenceItem]:
        self.load_if_needed()
        if not self.items:
            return []

        clean_q = re.sub(r"[^\w\s]", " ", query.lower())
        raw_words = clean_q.split()
        q_tokens = [w for w in raw_words if len(w) >= 3 and w not in STOP_WORDS]

        if original_text and len(q_tokens) < 3:
            clean_o = re.sub(r"[^\w\s]", " ", original_text.lower())
            orig_tokens = [w for w in clean_o.split() if len(w) >= 3 and w not in STOP_WORDS]
            q_tokens = list(dict.fromkeys(q_tokens + orig_tokens))

        if not q_tokens:
            return []

        # Expansão controlada de sinônimos para palavras-chave
        expanded_tokens: list[set[str]] = []
        for t in q_tokens:
            token_set = {t}
            if t in SYNONYMS:
                token_set.update(SYNONYMS[t])
            expanded_tokens.append(token_set)

        scored: list[tuple[float, dict[str, Any]]] = []

        for item in self.items:
            stext = item["search_text"]
            matches_count = 0
            for tset in expanded_tokens:
                if any(syn in stext for syn in tset):
                    matches_count += 1

            if matches_count < 2:
                # Se for consulta de apenas 1 token forte (ex: "mounjaro" ou "tirzepatida")
                if len(q_tokens) == 1 and matches_count == 1:
                    score = 1.0
                else:
                    continue
            else:
                score = matches_count / len(q_tokens)

            # Bônus para proximidade de título
            title_lower = item["title"].lower()
            title_matches = sum(1 for tset in expanded_tokens if any(syn in title_lower for syn in tset))
            score += title_matches * 0.4

            if score >= 0.60 or matches_count >= 2:
                scored.append((score, item))

        scored.sort(key=lambda x: -x[0])

        evidences: list[EvidenceItem] = []
        for score, it in scored[:limit]:
            stance = "REFUTES" if it["is_fake"] else "SUPPORTS"
            rating = "Falso" if it["is_fake"] else "Verdadeiro"
            pub_date = it.get("date") or None

            evidences.append(
                EvidenceItem(
                    title=it["title"],
                    source_name=f"{it['author']} (Checagem Oficial)",
                    url=it["url"],
                    snippet=it["claim"] or it["title"],
                    rating=rating,
                    is_fact_check=True,
                    published_date=pub_date,
                    claim_reviewed=it["claim"],
                    source_tier=SourceTier.TIER1_OFFICIAL_OR_IFCN.value,
                    stance=stance,
                    is_relevant=True,
                )
            )

        return evidences


_CORPUS_INSTANCE: FactCheckCorpus | None = None


def get_factcheck_corpus() -> FactCheckCorpus:
    global _CORPUS_INSTANCE
    if _CORPUS_INSTANCE is None:
        _CORPUS_INSTANCE = FactCheckCorpus()
        _CORPUS_INSTANCE.load_if_needed()
    return _CORPUS_INSTANCE
