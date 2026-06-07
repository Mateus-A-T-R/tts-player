"""
narrator_rules.py — Regras expressivas para o modo narrador.

Cada regra é aplicada em ordem na função `apply_narrator_rules()`.
Para adicionar novas regras: edite as listas abaixo ou implemente
uma nova função e registre em RULE_PIPELINE.

Todas as regras são puramente locais (sem API externa, sem custo).
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# 1. VERBOS EMOTIVOS — detecta verbo de fala/emoção e ajusta prosódia
# ─────────────────────────────────────────────────────────────────────────────
# Cada entrada: (regex_padrão, rate_delta, pitch_delta, volume_delta)
# Os deltas são somados ao base do narrador em _apply_verb_prosody().
# volume_delta: "loud" | "soft" | None

EMOTION_VERBS_PTBR = [
    # Gritos / intensidade alta
    (r"\b(gritou|berrou|vociferou|urrou|bramiu|exclamou|irrompeu)\b",
     "+14%", "+8Hz", "loud"),

    # Sussurros / voz baixa
    (r"\b(sussurrou|murmurou|cochicho|cochichou|segredou|balbuciou)\b",
     "-18%", "-6Hz", "soft"),

    # Choro / tristeza
    (r"\b(chorou|soluçou|gemeu|lamentou|implorou|suplicou)\b",
     "-12%", "-4Hz", "soft"),

    # Riso / alegria
    (r"\b(riu|gargalhou|cacarejou|respondeu animado|disse rindo)\b",
     "+10%", "+6Hz", None),

    # Raiva / tensão
    (r"\b(rosnou|resmungou|exigiu|ordenou|ameaçou|rugiu)\b",
     "+8%", "+4Hz", "loud"),

    # Medo / hesitação
    (r"\b(tremeu|hesitou|gaguejou|balbuciou com medo|disse tremendo)\b",
     "-10%", "-2Hz", "soft"),

    # Frieza / autoridade
    (r"\b(declarou|afirmou com firmeza|sentenciou|pronunciou)\b",
     "-5%", "-2Hz", None),
]

EMOTION_VERBS_ENUS = [
    (r"\b(shouted|yelled|screamed|roared|cried out|exclaimed)\b",
     "+14%", "+8Hz", "loud"),
    (r"\b(whispered|murmured|breathed|said softly|muttered)\b",
     "-18%", "-6Hz", "soft"),
    (r"\b(sobbed|wept|cried|moaned|pleaded|begged)\b",
     "-12%", "-4Hz", "soft"),
    (r"\b(laughed|chuckled|giggled|said with a smile)\b",
     "+10%", "+6Hz", None),
    (r"\b(growled|snapped|demanded|ordered|threatened)\b",
     "+8%", "+4Hz", "loud"),
    (r"\b(trembled|hesitated|stuttered|stammered)\b",
     "-10%", "-2Hz", "soft"),
    (r"\b(declared|announced|stated firmly|pronounced)\b",
     "-5%", "-2Hz", None),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRASES CURTAS DRAMÁTICAS — palavras ≤ N ganham pausa e ritmo mais lento
# ─────────────────────────────────────────────────────────────────────────────

DRAMATIC_SHORT_MAX_WORDS = 5      # frases com até N palavras
DRAMATIC_PAUSE_BEFORE_MS = 650    # pausa antes da frase (ms)
DRAMATIC_RATE_DELTA       = "-18%" # leitura mais lenta
DRAMATIC_PITCH_DELTA      = "-2Hz" # levemente mais grave

# Exceções: frases curtas que NÃO devem ser dramatizadas (ex: continuações)
DRAMATIC_EXCEPTIONS_RE = re.compile(
    r"^(sim|não|ok|talvez|exato|claro|pois|então|mas|e|ou|porém|"
    r"yes|no|ok|maybe|exactly|sure|but|and|or|however)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENUMERAÇÕES — listas com vírgulas aceleram progressivamente
# ─────────────────────────────────────────────────────────────────────────────
# Ativado quando a frase tem ≥ 3 vírgulas (provável enumeração)

ENUMERATION_MIN_COMMAS = 3
ENUMERATION_RATE_DELTA = "+6%"


# ─────────────────────────────────────────────────────────────────────────────
# 4. PARÊNTESES / APARTE — conteúdo entre parênteses em voz mais baixa e rápida
# ─────────────────────────────────────────────────────────────────────────────

PARENTHESIS_RATE  = "+12%"
PARENTHESIS_PITCH = "-4Hz"
PARENTHESIS_RE    = re.compile(r"\(([^)]+)\)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LETRAS MAIÚSCULAS (ênfase extra) — palavras TOTALMENTE em maiúsculas
#    Ex: "NUNCA faça isso!" → ênfase forte na palavra
# ─────────────────────────────────────────────────────────────────────────────

CAPS_WORD_RE = re.compile(r'\b([A-ZÁÉÍÓÚÃÕÂÊÎÔÛ]{3,})\b')


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAVESSÃO DE DIÁLOGO — linha começa com "— " (fala de personagem)
#    Ganha pausa antes e leitura levemente mais rápida (voz de personagem)
# ─────────────────────────────────────────────────────────────────────────────

DIALOGUE_LINE_RE  = re.compile(r"^[—–-]\s+")
DIALOGUE_PAUSE_MS = 350
DIALOGUE_RATE     = "+5%"


# ─────────────────────────────────────────────────────────────────────────────
# 7. NÚMEROS ESCRITOS — pausas naturais ao redor de números longos
# ─────────────────────────────────────────────────────────────────────────────

NUMBER_RE = re.compile(r"\b(\d{4,})\b")


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR PRINCIPAL — aplica todas as regras e retorna (ssml_inner, meta)
# ─────────────────────────────────────────────────────────────────────────────

def apply_narrator_rules(escaped_text: str, original: str, lang: str = "pt-BR"):
    """
    Aplica as regras do narrador ao texto já escapado para XML.

    Retorna (inner_ssml, meta) onde:
    - inner_ssml: texto pronto para envolver em <prosody> base
    - meta: dict com chaves opcionais:
        "pause_before_ms": int  — pausa extra antes da frase
        "rate_delta": str       — ajuste de rate ("+8%", "-12%", etc.)
        "pitch_delta": str      — ajuste de pitch
        "volume": str           — "loud" | "soft" | None
    """
    meta = {
        "pause_before_ms": 0,
        "rate_delta": None,
        "pitch_delta": None,
        "volume": None,
    }

    text = escaped_text  # trabalha com a cópia escapada

    # ── Regra 6: travessão de diálogo ────────────────────────────────────────
    if DIALOGUE_LINE_RE.search(original):
        meta["pause_before_ms"] = max(meta["pause_before_ms"], DIALOGUE_PAUSE_MS)
        meta["rate_delta"] = DIALOGUE_RATE

    # ── Regra 2: frase curta dramática ───────────────────────────────────────
    word_count = len(original.split())
    if (word_count <= DRAMATIC_SHORT_MAX_WORDS
            and not DRAMATIC_EXCEPTIONS_RE.match(original)
            and len(original) > 3):
        meta["pause_before_ms"] = max(meta["pause_before_ms"], DRAMATIC_PAUSE_BEFORE_MS)
        if not meta["rate_delta"]:
            meta["rate_delta"] = DRAMATIC_RATE_DELTA
        meta["pitch_delta"] = DRAMATIC_PITCH_DELTA

    # ── Regra 1: verbos emotivos ──────────────────────────────────────────────
    verb_list = EMOTION_VERBS_PTBR if lang == "pt-BR" else EMOTION_VERBS_ENUS
    for pattern, rate_d, pitch_d, vol in verb_list:
        if re.search(pattern, original, re.IGNORECASE):
            meta["rate_delta"]  = rate_d
            meta["pitch_delta"] = pitch_d
            meta["volume"]      = vol
            break  # primeira correspondência vence

    # ── Regra 3: enumerações ──────────────────────────────────────────────────
    comma_count = original.count(",")
    if comma_count >= ENUMERATION_MIN_COMMAS and not meta["rate_delta"]:
        meta["rate_delta"] = ENUMERATION_RATE_DELTA

    # ── Regra 5: palavras em CAPS → <emphasis> ────────────────────────────────
    text = CAPS_WORD_RE.sub(
        lambda m: f'<emphasis level="strong">{m.group(1)}</emphasis>',
        text,
    )

    # ── Regra 4: parênteses → aparte em voz diferente ────────────────────────
    text = PARENTHESIS_RE.sub(
        lambda m: (
            f'<prosody rate="{PARENTHESIS_RATE}" pitch="{PARENTHESIS_PITCH}">'
            f'({m.group(1)})</prosody>'
        ),
        text,
    )

    return text, meta
