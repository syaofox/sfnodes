"""Pure helpers for SF Value Dropdown (ported from comfyui-pixaroma
_dropdown_helpers.py).

Everything here is pure: no I/O, no globals, no ComfyUI — so it can be
unit-tested without a ComfyUI import.

THE PARITY RULE: web/sf_dropdown_lib.js mirrors `readable` and the coercion
rules below so the settings panel can mark a row that will not read as the
chosen type. If you change a rule here, change it there in the SAME commit,
or the panel will promise one thing and the run will do another. The rules
are deliberately simple so that mirroring stays trivial.
"""

import json
import math
import re

# THE shared number grammar. Deliberately NOT each language's native parser:
# float() and Number() disagree in both directions, and every disagreement is a
# case where the settings panel promises one thing and the run does another.
# Measured, not assumed - a parity run found JS reading "0x10" as 16 while
# Python refused it, and Python accepting "1_0" while JS refuses.
#   accepts: 5  5.  .5  5.5  +5  -3  1e3  1E3  -1e3
#   refuses: 0x10  0b1  1_0  1,024  1024px  abc  Infinity  NaN  (and "")
#
# [0-9] and NOT \d. Python's \d also matches fullwidth and Arabic-Indic digits
# (what a CJK IME produces in full-width mode) where JavaScript's \d is
# ASCII-only by language definition. Enumerated, not assumed: JS \d matches
# exactly the ten code points U+0030..U+0039.
_NUMBER_RE = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")

# EXACTLY the code points JavaScript's String.prototype.trim removes, enumerated
# by running it over U+0000..U+FFFF rather than read off the spec.
#
# Python's str.strip() is NOT the same set, in BOTH directions: it leaves U+FEFF
# (a BOM, which is what a value pasted out of an Excel CSV or a BOM-marked text
# file carries) and it strips U+001C..U+001F and U+0085, which JS keeps. Either
# way the two languages disagreed about whether a row was a number.
_JS_WHITESPACE = (
    "\t\n\v\f\r "                     # U+0009..U+000D and space
    "\u00a0\u1680"                            # NBSP, OGHAM SPACE MARK
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"  # EN QUAD..HAIR SPACE
    "\u2028\u2029\u202f\u205f\u3000"  # separators, NNBSP, MMSP, IDEOGRAPHIC
    "\ufeff"                                              # BOM / ZWNBSP
)

# The four types a Dropdown node can be set to. The JS uses these exact
# strings in its state blob, so they are a wire format - do not rename.
TYPES = ("text", "int", "float", "bool")

# What an option emits when its value cannot be read as the chosen type.
# A run must never fail because one row of a list is malformed: the node is a
# convenience, and taking down a whole queue over a typo would be absurd.
FALLBACKS = {"text": "", "int": 0, "float": 0.0, "bool": False}

# Accepted spellings for the on/off type, lowercased and stripped.
_TRUE_WORDS = frozenset(("true", "yes", "on", "y", "t"))
_FALSE_WORDS = frozenset(("false", "no", "off", "n", "f"))

# Same clamp Control Panel applies. A hand-edited API file can carry 1e308 or a
# 400-digit integer, and passing that straight into a downstream node is how
# you get an unhelpful crash somewhere far away from the cause.
_LIMIT = 1e12


def normalize_type(kind):
    """Anything -> one of TYPES. Unknown values become 'text'.

    'text' is the safe unknown, not 'int': an unrecognised type is most likely
    a newer version's type we do not know yet, and emitting the raw string
    loses less than emitting 0.
    """
    if not isinstance(kind, str):
        return "text"
    k = kind.strip(_JS_WHITESPACE).lower()
    if k in TYPES:
        return k
    # Tolerate a few obvious aliases so a hand-edited workflow still runs.
    if k in ("string", "str"):
        return "text"
    if k in ("integer", "whole"):
        return "int"
    if k in ("decimal", "number", "double"):
        return "float"
    if k in ("boolean", "toggle", "onoff", "on/off"):
        return "bool"
    return "text"


def _as_number(raw):
    """raw -> finite float, or None if it cannot be read as one."""
    if isinstance(raw, bool):
        # Must precede the int check: bool IS an int in Python, and True would
        # otherwise silently become 1.0 for a float row.
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        try:
            # OverflowError is real here: a bare 400-digit integer parses from
            # JSON as an arbitrary-precision int and float() then raises.
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(raw, str):
        # strip(_JS_WHITESPACE), never bare strip(): the browser trims a
        # different set, so a value pasted with a leading BOM read as a number
        # there and as junk here.
        text = raw.strip(_JS_WHITESPACE)
        if not _NUMBER_RE.match(text):
            # Covers empty/whitespace too, since the pattern needs at least one
            # digit. Everything the grammar refuses is refused IDENTICALLY by
            # the browser, which is the whole point of having a grammar.
            return None
        try:
            value = float(text)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None
    if not math.isfinite(value):
        return None
    return value


def _round_half_away(value):
    """Round to the nearest whole number, halves going AWAY from zero.

    Neither language's default is usable here. Python's round() is banker's
    rounding (2.5 -> 2, 0.5 -> 0), JavaScript's Math.round breaks ties toward
    positive infinity (-3.5 -> -3). They disagree on every exact half, so the
    panel previewed one number and the run emitted another.

    Half-away-from-zero is also what a person expects: someone typing 2.5 into
    a whole-number list means 3, not 2.
    """
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def _number_to_text(value):
    """A number -> the string the BROWSER would show for it.

    Python str() on a whole float keeps the '.0' that JavaScript drops, so the
    same value read as text differed between the panel and the run. Match the
    browser, because the browser is what the user is looking at.
    """
    if isinstance(value, float) and math.isfinite(value):
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        return repr(value)
    return str(value)


# The single category name used when none is written. Mirrors the JS.
_DEFAULT_CATEGORY = "default"


def _normalize_category(value):
    """Anything -> a category name. Empty/non-string -> "default"."""
    if not isinstance(value, str):
        return _DEFAULT_CATEGORY
    s = value.strip(_JS_WHITESPACE)
    return s or _DEFAULT_CATEGORY


def readable(raw, kind):
    """Would `raw` read cleanly as `kind`? Mirrored by the JS for the warning marks.

    Text is always readable (anything can be shown as text), which is why
    switching a list TO text never marks a row.
    """
    kind = normalize_type(kind)
    if kind == "text":
        return True
    if kind == "bool":
        if isinstance(raw, bool):
            return True
        if isinstance(raw, str) and raw.strip(_JS_WHITESPACE).lower() in (_TRUE_WORDS | _FALSE_WORDS):
            return True
        # A number reads as on/off by the usual zero/non-zero rule, and the
        # clamp cannot change that answer, so magnitude is irrelevant here.
        return _as_number(raw) is not None

    number = _as_number(raw)
    if number is None:
        return False
    # A value the clamp would MOVE is not readable, even though it parsed.
    # Without this the panel showed no warning on a 15-digit seed and the run
    # then sent 1000000000000 instead - the panel promising one number and the
    # run sending another, which is the exact thing the warning marks exist to
    # prevent. Flagged rather than clamp-free: letting 1e308 through to a
    # downstream node helps nobody.
    return -_LIMIT <= number <= _LIMIT


def coerce_value(raw, kind):
    """raw + type -> the Python value the node emits. Never raises."""
    kind = normalize_type(kind)

    if kind == "text":
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bool):
            # Emit the spelling the user would have typed, not Python's.
            return "true" if raw else "false"
        if isinstance(raw, (int, float)):
            return _number_to_text(raw)
        try:
            return str(raw)
        except RecursionError:
            # Deeply nested containers (a hand-edited API file can carry
            # thousands of layers) blow up str() with a RecursionError. A run
            # must never fail because one row of a list is malformed.
            return ""

    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            # Must use the SAME set as readable()'s bool branch above, or the
            # two disagree with each other before either disagrees with the JS.
            word = raw.strip(_JS_WHITESPACE).lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
        number = _as_number(raw)
        if number is None:
            return FALLBACKS["bool"]
        return number != 0.0

    number = _as_number(raw)
    if number is None:
        return FALLBACKS[kind]
    number = max(-_LIMIT, min(_LIMIT, number))
    if kind == "int":
        return _round_half_away(number)
    return float(number)


def parse_state(raw):
    """The hidden DropdownState string -> a normalized dict. Never raises.

    Returns {"type": <one of TYPES>, "index": int, "options": [{"name","value","category"}]}.
    "index" is an index into the options of the CURRENT category (state["category"],
    default "default"); rows whose "category" is missing fall to "default", so a
    version-1 state (no category fields at all) filters to everything and behaves
    exactly as before. Every field is coerced into shape, because this string can
    arrive from a hand-edited API file as literally anything.
    """
    state = None
    if isinstance(raw, dict):
        state = raw
    elif isinstance(raw, str):
        try:
            # RecursionError too: deeply nested JSON would otherwise take the
            # whole run down rather than just this node.
            state = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            state = None
    if not isinstance(state, dict):
        state = {}

    kind = normalize_type(state.get("type"))
    category = _normalize_category(state.get("category"))

    raw_options = state.get("options")
    if not isinstance(raw_options, list):
        raw_options = []
    options = []
    for entry in raw_options:
        # A non-dict row (null, a bare string, an array) is dropped rather than
        # crashing the list. Control Panel learned this the hard way: one null
        # row aborted value injection for every OTHER node of its type too.
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        options.append({
            "name": name if isinstance(name, str) else "",
            "value": entry.get("value"),
            "category": _normalize_category(entry.get("category")),
        })

    # Only the current category's rows are in play. Mirrors the JS visibleOptions().
    options = [o for o in options if o["category"] == category]

    index = state.get("index")
    if isinstance(index, bool) or not isinstance(index, (int, float)):
        index = 0
    else:
        try:
            index = int(index)
        except (TypeError, ValueError, OverflowError):
            index = 0

    return {"type": kind, "index": index, "options": options}


def _loads(raw):
    """The hidden state string -> a dict, never raising."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            # RecursionError too: deeply nested JSON would otherwise take the
            # whole run down rather than just this node.
            state = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            return {}
        return state if isinstance(state, dict) else {}
    return {}


def selected_value(raw):
    """The hidden state string -> the single value this node outputs.

    TWO accepted shapes, on purpose:

    1. LEAN, `{"type": ..., "value": ...}` - what the browser actually injects
       at graphToPrompt time. The injected string IS the node's cache key, so it
       carries ONLY what changes the result. The option NAMES, the rest of the
       list, the mode and any UI flag are display-only: renaming a row or
       recolouring the node must not re-run the graph, and editing a row you
       have not selected must not either.

    2. FULL, `{"type": ..., "index": N, "options": [...]}` - the shape stored in
       the workflow. Accepted so a hand-written or hand-edited API file still
       runs, and so the node degrades sanely if injection is ever missed.

    An empty list, or an index past the end, gives the type's fallback rather
    than raising: an unconfigured Dropdown should still run and hand on an empty
    string, not turn the node red.
    """
    state = _loads(raw)
    kind = normalize_type(state.get("type"))

    # Shape 1 wins when present. Checked by KEY, not truthiness: "" and 0 and
    # False are all perfectly ordinary selected values.
    if "value" in state:
        return coerce_value(state.get("value"), kind)

    parsed = parse_state(state)
    options = parsed["options"]
    index = parsed["index"]
    if not options or index < 0 or index >= len(options):
        return FALLBACKS[parsed["type"]]
    return coerce_value(options[index].get("value"), parsed["type"])
