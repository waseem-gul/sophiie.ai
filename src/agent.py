import asyncio
import logging
import os
import httpx

from dotenv import load_dotenv
from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    function_tool,
    RunContext,
)
from livekit.plugins import silero
from livekit.plugins.assemblyai import STT
from livekit.plugins.elevenlabs import TTS
from livekit.plugins.openai import LLM
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def getCurrentWeather(self, context: RunContext, location: str):
        url = f"https://wttr.in/{location}?format=j1"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    @function_tool
    async def getForecast(self, context: RunContext, location: str):
        return "cold with a temperature of 30 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    lkapi = api.LiveKitAPI(
        os.environ.get("LIVEKIT_URL"),
        os.environ.get("LIVEKIT_API_KEY"),
        os.environ.get("LIVEKIT_API_SECRET"),
    )

    async def start_track_egress(
        publication: rtc.TrackPublication, participant: rtc.Participant
    ):
        try:
            logger.info(
                f"Starting egress for: {participant.identity} - {publication.sid}"
            )
            track_egress = api.TrackEgressRequest(
                room_name=ctx.room.name,
                track_id=publication.sid,
                file=api.DirectFileOutput(
                    filepath=f"livekit/{ctx.room.name}/{participant.identity}-{publication.sid}.ogg",
                    s3=api.S3Upload(
                        bucket=os.environ.get("AWS_S3_BUCKET"),
                        region=os.environ.get("AWS_REGION"),
                        access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                        force_path_style=True,
                    ),
                ),
            )
            result = await lkapi.egress.start_track_egress(track_egress)
            logger.info(
                f"Track egress started: {result.egress_id} for {participant.identity}"
            )
        except Exception as e:
            logger.error(
                f"Failed to start track egress for {participant.identity}: {e}"
            )

    egress_started_for = set()

    @ctx.room.on("track_published")
    def on_track_published(
        publication: rtc.TrackPublication, participant: rtc.Participant
    ):
        logger.info(
            f"Track published event: {participant.identity} - {publication.sid} (kind: {publication.kind})"
        )

        if (
            publication.kind == rtc.TrackKind.KIND_AUDIO
            and publication.sid not in egress_started_for
        ):
            egress_started_for.add(publication.sid)
            asyncio.create_task(start_track_egress(publication, participant))

    session = AgentSession(
        stt=STT(),
        # llm=LLM.with_ollama(model="phi:latest", base_url="http://localhost:11434/v1"),
        llm=LLM(model="gpt-4.1-mini"),
        tts=TTS(model="eleven_multilingual_v2"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    async def cleanup():
        await log_usage()
        await lkapi.aclose()

    ctx.add_shutdown_callback(cleanup)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(audio_enabled=True),
    )

    logger.info("Checking for already-published tracks...")

    async def check_and_start_egress():
        for (
            track_sid,
            publication,
        ) in ctx.room.local_participant.track_publications.items():
            if (
                publication.kind == rtc.TrackKind.KIND_AUDIO
                and track_sid not in egress_started_for
            ):
                logger.info(f"Found local track: {track_sid}")
                egress_started_for.add(track_sid)
                await start_track_egress(publication, ctx.room.local_participant)

        for participant in ctx.room.remote_participants.values():
            for track_sid, publication in participant.track_publications.items():
                if (
                    publication.kind == rtc.TrackKind.KIND_AUDIO
                    and track_sid not in egress_started_for
                ):
                    logger.info(
                        f"Found remote track: {participant.identity} - {track_sid}"
                    )
                    egress_started_for.add(track_sid)
                    await start_track_egress(publication, participant)

    await check_and_start_egress()

    # Check for up to 2 minutes (20 * 6 seconds)
    async def periodic_check():
        for _ in range(20):
            await asyncio.sleep(6)
            await check_and_start_egress()

    asyncio.create_task(periodic_check())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
