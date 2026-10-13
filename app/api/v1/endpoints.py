from typing import Any
from fastapi import APIRouter, Depends, Request, status
from app.config import Settings, get_settings
from app.core.orchestrator import FactCheckOrchestrator
from app.core.registry import registry
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from app.services.telegram_service import TelegramService, get_telegram_service

router = APIRouter()


def get_orchestrator(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> FactCheckOrchestrator:
    if hasattr(request.app.state, "orchestrator") and request.app.state.orchestrator is not None:
        return request.app.state.orchestrator
    return FactCheckOrchestrator(settings=settings)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Análise de Veracidade",
    description="Endpoint consumido pelo Bot do Telegram para verificar a veracidade de uma mensagem."
)
async def analyze(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator),
    telegram_service: TelegramService = Depends(get_telegram_service),
) -> AnalyzeResponse:
    """Recebe a solicitação, delega ao orquestrador e formata para o Telegram."""
    response = await orchestrator.analyze(text=request.text, urls=request.urls)

    # Gera a mensagem formatada para exibição no Telegram
    formatted_telegram = telegram_service.format_analysis_for_telegram(response, original_text=request.text)
    response.telegram_formatted_text = formatted_telegram

    # Entrega opcional e direta ao Telegram se chat_id foi fornecido
    if request.chat_id is not None:
        should_deliver = request.deliver_to_telegram or (
            settings.TELEGRAM_AUTO_REPLY and telegram_service.is_configured
        )
        if should_deliver and telegram_service.is_configured:
            delivery = await telegram_service.send_message(
                chat_id=request.chat_id,
                text=formatted_telegram,
            )
            response.telegram_delivery = delivery

    return response


@router.post(
    "/analyze/stream",
    summary="Análise de Veracidade com Streaming de Progresso",
    description="Endpoint com Server-Sent Events (SSE) para acompanhamento em tempo real das etapas de verificação."
)
async def analyze_stream(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    import asyncio
    import json
    import logging
    from fastapi.responses import StreamingResponse

    log = logging.getLogger("factchkbr.api.stream")

    async def event_generator():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_step(step_name: str, result: Any, duration: float):
            info = {
                "type": "step",
                "step": step_name,
                "duration": round(duration, 3),
            }
            if step_name == "claim_extractor":
                info["claim"] = getattr(result, "claim", "")
                info["message"] = f"Alegação identificada: \"{info['claim']}\""
            elif step_name == "fact_check_api":
                evs = (getattr(result, "raw_details", {}) or {}).get("total_evidences", 0)
                info["evidences_count"] = evs
                info["message"] = f"Pesquisa concluída: {evs} evidência(s) e estudos localizados"
            elif step_name == "llm_judge":
                info["message"] = "Julgamento e avaliação de riscos consolidados"
            await queue.put(info)

        async def run_pipeline():
            try:
                await queue.put({
                    "type": "start",
                    "message": "Iniciando verificação na esteira FactChkBR..."
                })
                res = await orchestrator.analyze(text=request.text, urls=request.urls, on_step=on_step)
                formatted = telegram_service.format_analysis_for_telegram(res, original_text=request.text)
                res.telegram_formatted_text = formatted

                if request.chat_id is not None:
                    should_deliver = request.deliver_to_telegram or (
                        settings.TELEGRAM_AUTO_REPLY and telegram_service.is_configured
                    )
                    if should_deliver and telegram_service.is_configured:
                        delivery = await telegram_service.send_message(
                            chat_id=request.chat_id,
                            text=formatted,
                        )
                        res.telegram_delivery = delivery

                await queue.put({
                    "type": "complete",
                    "data": res.model_dump(),
                })
            except Exception as exc:
                log.exception("Erro durante pipeline de streaming: %s", exc)
                await queue.put({
                    "type": "error",
                    "error": str(exc),
                })
            finally:
                await queue.put({"type": "done"})

        asyncio.create_task(run_pipeline())

        while True:
            item = await queue.get()
            if item.get("type") == "done":
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Status da Aplicação",
    description="Verifica a integridade da API e lista os analisadores ativos e disponíveis."
)
async def health(
    settings: Settings = Depends(get_settings),
    orchestrator: FactCheckOrchestrator = Depends(get_orchestrator)
) -> HealthResponse:
    active = [a.name for a in orchestrator.get_active_analyzers()]
    available = registry.list_available()

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        active_analyzers=active,
        available_analyzers=available
    )
