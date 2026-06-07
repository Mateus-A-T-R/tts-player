import io
import os
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import edge_tts
import numpy as np
import soundfile as sf

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KPipeline = None
    KOKORO_AVAILABLE = False

SAMPLE_RATE = 24000

# ── Voice definitions ─────────────────────────────────────────────────────────
# engine: "edge" = Microsoft Neural TTS (online, very natural)
#         "kokoro" = local model (offline, faster, less natural)

VOICES = {
    "pt-BR": {
        "edge": [
            {"id": "pt-BR-AntonioNeural",           "label": "Antonio (Masculino)"},
            {"id": "pt-BR-ThalitaMultilingualNeural","label": "Thalita (Feminino) ✦"},
            {"id": "pt-BR-FranciscaNeural",          "label": "Francisca (Feminino)"},
        ],
        "kokoro": [
            {"id": "pf_dora",  "label": "Dora (Feminino) — offline"},
            {"id": "pm_alex",  "label": "Alex (Masculino) — offline"},
            {"id": "pm_santa", "label": "Santa (Masculino) — offline"},
        ],
    },
    "en-US": {
        "edge": [
            {"id": "en-US-RogerNeural",       "label": "Roger (Male)"},
            {"id": "en-US-AndrewNeural",       "label": "Andrew (Male)"},
            {"id": "en-US-ChristopherNeural",  "label": "Christopher (Male)"},
            {"id": "en-US-EricNeural",         "label": "Eric (Male)"},
            {"id": "en-US-GuyNeural",          "label": "Guy (Male)"},
            {"id": "en-US-BrianNeural",        "label": "Brian (Male)"},
            {"id": "en-US-JennyNeural",        "label": "Jenny (Female) ✦"},
            {"id": "en-US-AriaNeural",         "label": "Aria (Female)"},
        ],
        "kokoro": [
            {"id": "af_bella",   "label": "Bella (Female) — offline"},
            {"id": "af_sarah",   "label": "Sarah (Female) — offline"},
            {"id": "am_michael", "label": "Michael (Male) — offline"},
            {"id": "bm_george",  "label": "George (Male) — offline"},
        ],
    },
}

# Narrator preset — Thalita as narrator (pt-BR), Roger (en-US)
NARRATOR_VOICE   = {"pt-BR": "pt-BR-ThalitaMultilingualNeural", "en-US": "en-US-RogerNeural"}
NARRATOR_PITCH   = "-3Hz"   # slight warmth without changing character
NARRATOR_RATE    = "-15%"   # deliberate storytelling pace
NARRATOR_SILENCE = 420      # ms of silence between sentences

_KOKORO_LANG_CODES = {"pt-BR": "p", "en-US": "a"}
_kokoro_pipelines: dict[str, KPipeline] = {}


def get_kokoro_pipeline(language: str) -> KPipeline:
    if language not in _kokoro_pipelines:
        lang_code = _KOKORO_LANG_CODES.get(language, "a")
        _kokoro_pipelines[language] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _kokoro_pipelines[language]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 200:
            sub = re.split(r"(?<=[,;:])\s+", part)
            result.extend(s.strip() for s in sub if s.strip())
        else:
            result.append(part)
    return result if result else [text.strip()]


# ── SSML helpers for narrator mode ───────────────────────────────────────────

def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_narrator_ssml(sentence: str, voice: str, lang: str = "pt-BR") -> str:
    """Build expressive SSML for narrator mode with emotional prosody."""
    stripped = sentence.strip()
    is_exclamatory = stripped.endswith("!")
    is_question    = stripped.endswith("?")
    has_ellipsis   = bool(re.search(r"\.{3}|…", stripped))

    esc = _xml_escape(stripped)

    # Emphasis on quoted dialogue — must run BEFORE break tags so the " in
    # <break time="Xms"/> attributes don't get falsely matched as quote pairs
    esc = re.sub(r'["“„«]([^"“”„»\n]+)["”„»]',
                 r'<emphasis level="moderate">\1</emphasis>', esc)

    # Natural pauses at internal punctuation
    esc = re.sub(r"(,)( )",        r'\1<break time="120ms"/>\2', esc)
    esc = re.sub(r"(;)( )",        r'\1<break time="180ms"/>\2', esc)
    esc = re.sub(r"(—|–)( ?)",     r'<break time="260ms"/>\1\2', esc)
    esc = re.sub(r"(\.{3}|…)( |$)", r'\1<break time="500ms"/>\2', esc)

    # Sentence-level prosody (overlaid on top of base narrator prosody)
    if is_exclamatory:
        inner = f'<prosody rate="+8%" pitch="+3Hz">{esc}</prosody>'
    elif is_question:
        inner = f'<prosody rate="+3%" pitch="+5Hz">{esc}</prosody>'
    elif has_ellipsis:
        inner = f'<prosody rate="-10%">{esc}</prosody>'
    else:
        inner = esc

    body = f'<prosody rate="{NARRATOR_RATE}" pitch="{NARRATOR_PITCH}">{inner}</prosody>'
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang}">'
        f'<voice name="{voice}">{body}</voice>'
        f'</speak>'
    )


# ── Silence generator (WAV, for narrator pauses) ─────────────────────────────

def generate_silence(duration_ms: int) -> bytes:
    samples = int(SAMPLE_RATE * duration_ms / 1000)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(samples, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ── Edge TTS (async, returns MP3 bytes) ──────────────────────────────────────

async def synthesize_edge(sentence: str, voice: str, speed: float, pitch: str = "+0Hz") -> bytes:
    rate_pct = int((speed - 1.0) * 100)
    rate = f"{rate_pct:+d}%"
    comm = edge_tts.Communicate(sentence, voice, rate=rate, pitch=pitch)
    chunks: list[bytes] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


async def synthesize_edge_narrator(sentence: str, voice: str, lang: str = "pt-BR") -> bytes:
    """Narrator preset with expressive SSML — emotional prosody and natural pauses."""
    ssml = _build_narrator_ssml(sentence, voice, lang)
    comm = edge_tts.Communicate(ssml, voice)
    chunks: list[bytes] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


# ── Kokoro (sync, returns WAV bytes — call via asyncio.to_thread) ─────────────

def synthesize_kokoro(pipeline: KPipeline, sentence: str, voice: str, speed: float) -> bytes:
    chunks: list[np.ndarray] = []
    try:
        for _, _, audio in pipeline(sentence, voice=voice, speed=speed):
            if audio is not None and len(audio) > 0:
                chunks.append(audio)
    except Exception as exc:
        raise RuntimeError(f"Kokoro synthesis failed: {exc}") from exc

    if not chunks:
        return b""

    combined = np.concatenate(chunks).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, combined, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()
