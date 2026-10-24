"""Offline interpretation of a typed search request into a structured search spec (IMG-03).

The interpreter is a deterministic parser over a fixed English vocabulary
(design package, Deliverable 07, "grammar / keyword parser"). It was chosen over
a local language model for this first version because:

- its output is fully predictable and testable against a written request set,
- it needs no model pack, GPU or memory budget,
- it cannot produce anything but the spec below: there is no code path that
  executes text from the request.

Rules:
- The spec is plain data and is checked by validate_spec() before it is stored
  or run. Unknown keys, wrong types and values outside the vocabulary are refused.
- Words the parser does not understand are never dropped silently. They are
  listed in "not_understood", and a search cannot run until the operator edits
  them away or explicitly accepts that they are ignored.
- Things the parser cannot decide on its own are not guessed: "weekend" and
  "weekdays" differ between countries, so the operator must name the days.
- A subtype the detector does not tell apart (pickup, van, ...) is kept as a
  requirement. Matching objects of the parent classes are reported as
  "possible", never as matches.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass

INTERPRETER_VERSION = "grammar-en-1"
SPEC_SCHEMA = 1

CATEGORIES = ("vehicle", "car", "bus", "truck", "motorcycle", "bicycle")
DETECTOR_CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle")
# Subtypes the current detector does not identify, with the detector classes they may appear as.
SUBTYPES = {
    "pickup": ("car", "truck"),
    "van": ("car", "truck"),
    "suv": ("car",),
    "taxi": ("car",),
    "minibus": ("bus", "car"),
    "semi_trailer": ("truck",),
}
COLOURS = ("red", "orange", "yellow", "green", "blue", "white", "grey", "black", "brown")
CONDITIONS = ("day", "dusk", "night")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
UNCERTAIN = ("list_separately", "exclude")

_CATEGORY_WORDS = {
    "vehicle": "vehicle", "vehicles": "vehicle", "car": "car", "cars": "car", "automobile": "car",
    "automobiles": "car", "bus": "bus", "buses": "bus", "busses": "bus", "truck": "truck", "trucks": "truck",
    "lorry": "truck", "lorries": "truck", "motorcycle": "motorcycle", "motorcycles": "motorcycle",
    "motorbike": "motorcycle", "motorbikes": "motorcycle", "bicycle": "bicycle", "bicycles": "bicycle",
    "bike": "bicycle", "bikes": "bicycle",
}
_SUBTYPE_WORDS = {
    "pickup": "pickup", "pickups": "pickup", "pick-up": "pickup", "pick-ups": "pickup",
    "van": "van", "vans": "van", "suv": "suv", "suvs": "suv", "taxi": "taxi", "taxis": "taxi", "cab": "taxi",
    "cabs": "taxi", "minibus": "minibus", "minibuses": "minibus", "semi-trailer": "semi_trailer",
    "semi-trailers": "semi_trailer", "semitrailer": "semi_trailer", "semitrailers": "semi_trailer",
    "semi": "semi_trailer", "semis": "semi_trailer", "articulated": "semi_trailer",
}
# The subtype's natural category when the request names it on its own ("vans").
_SUBTYPE_CATEGORY = {"pickup": "truck", "van": "vehicle", "suv": "car", "taxi": "car", "minibus": "bus",
                     "semi_trailer": "truck"}
_COLOUR_WORDS = {c: c for c in COLOURS} | {"gray": "grey", "silver": "grey", "dark-grey": "grey", "dark-gray": "grey"}
_DAY_WORDS = {}
for _i, _full in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")):
    for _w in (_full, _full + "s", _full[:3], _full[:3] + "."):
        _DAY_WORDS[_w] = WEEKDAYS[_i]
_DAY_WORDS.update({"tues": "tue", "weds": "wed", "thur": "thu", "thurs": "thu"})

_STOP = {
    "find", "show", "list", "search", "look", "looking", "for", "me", "all", "any", "every", "the", "a", "an",
    "please", "with", "that", "which", "who", "are", "is", "be", "of", "images", "image", "pictures", "picture",
    "photos", "photo", "frames", "frame", "where", "there", "coloured", "colored", "colour", "color", "in",
    "on", "to", "i", "want", "need", "and", "or", "ones", "one", "type", "types", "kind", "visible", "seen",
    "appear", "appears", "only", "also", "some", "can", "you", "get", "give", "containing", "contain", "contains",
    "have", "has", "showing", "shows", "at", "during", "time", "times", "hours", "hour", "from", "between", "it",
}
_NEG = {"not", "no", "except", "excluding", "exclude", "without", "non", "but", "other", "than", "never", "excl"}


class SpecError(ValueError):
    pass


@dataclass
class Context:
    """What the interpreter may refer to by name in this mission."""
    region_names: tuple[str, ...] = ()


def empty_spec() -> dict:
    return {
        "schema": SPEC_SCHEMA,
        "targets": [],
        "exclude": {"categories": [], "subtypes": []},
        "regions": {"include": [], "exclude": []},
        "time": {"windows": [], "date_from": None, "date_to": None, "conditions": []},
        "min_count": 1,
        "uncertain": "list_separately",
        "not_understood": [],
        "notes": [],
    }


def target(category: str, subtype: str | None = None, colours=(), exclude_colours=()) -> dict:
    return {"category": category, "subtype": subtype, "colours": sorted(set(colours)),
            "exclude_colours": sorted(set(exclude_colours))}


# ---------------------------------------------------------------- validation
def _check(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def validate_spec(spec: object) -> list[str]:
    """Return the problems with a spec; an empty list means it is valid. Never raises on bad input."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be an object"]
    want = set(empty_spec())
    extra = set(spec) - want
    missing = want - set(spec)
    _check(not extra, f"unknown keys: {sorted(extra)}", errors)
    _check(not missing, f"missing keys: {sorted(missing)}", errors)
    if errors:
        return errors
    _check(spec["schema"] == SPEC_SCHEMA, f"schema must be {SPEC_SCHEMA}", errors)
    ts = spec["targets"]
    if not isinstance(ts, list) or not ts:
        errors.append("at least one target is required")
        ts = []
    for i, t in enumerate(ts):
        if not isinstance(t, dict) or set(t) != {"category", "subtype", "colours", "exclude_colours"}:
            errors.append(f"target {i + 1}: must have category, subtype, colours, exclude_colours")
            continue
        _check(t["category"] in CATEGORIES, f"target {i + 1}: unknown category {t['category']!r}", errors)
        _check(t["subtype"] is None or t["subtype"] in SUBTYPES, f"target {i + 1}: unknown subtype {t['subtype']!r}", errors)
        for k in ("colours", "exclude_colours"):
            v = t[k]
            _check(isinstance(v, list) and all(c in COLOURS for c in v), f"target {i + 1}: {k} must be from {COLOURS}", errors)
        if isinstance(t["colours"], list) and isinstance(t["exclude_colours"], list):
            _check(not set(t["colours"]) & set(t["exclude_colours"]),
                   f"target {i + 1}: a colour cannot be both required and excluded", errors)
    ex = spec["exclude"]
    if not isinstance(ex, dict) or set(ex) != {"categories", "subtypes"}:
        errors.append("exclude must have categories and subtypes")
    else:
        _check(isinstance(ex["categories"], list) and all(c in DETECTOR_CLASSES for c in ex["categories"]),
               "exclude.categories must be detector classes", errors)
        _check(isinstance(ex["subtypes"], list) and all(c in SUBTYPES for c in ex["subtypes"]),
               "exclude.subtypes must be known subtypes", errors)
    rg = spec["regions"]
    if not isinstance(rg, dict) or set(rg) != {"include", "exclude"}:
        errors.append("regions must have include and exclude")
    else:
        for k in ("include", "exclude"):
            _check(isinstance(rg[k], list) and all(isinstance(n, str) and n.strip() for n in rg[k]),
                   f"regions.{k} must be a list of region names", errors)
    tm = spec["time"]
    if not isinstance(tm, dict) or set(tm) != {"windows", "date_from", "date_to", "conditions"}:
        errors.append("time must have windows, date_from, date_to, conditions")
    else:
        if not isinstance(tm["windows"], list):
            errors.append("time.windows must be a list")
        else:
            for i, w in enumerate(tm["windows"]):
                if not isinstance(w, dict) or set(w) != {"days", "start", "end"}:
                    errors.append(f"window {i + 1}: must have days, start, end")
                    continue
                _check(isinstance(w["days"], list) and w["days"] and all(d in WEEKDAYS for d in w["days"]),
                       f"window {i + 1}: days must be from {WEEKDAYS}", errors)
                for k in ("start", "end"):
                    _check(isinstance(w[k], str) and (re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", w[k]) is not None
                                                      or (k == "end" and w[k] == "24:00")),
                           f"window {i + 1}: {k} must be HH:MM (end may be 24:00)", errors)
                _check(w.get("start") != w.get("end"), f"window {i + 1}: start and end must differ", errors)
        for k in ("date_from", "date_to"):
            _check(tm[k] is None or (isinstance(tm[k], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tm[k]) is not None),
                   f"time.{k} must be YYYY-MM-DD or empty", errors)
        if isinstance(tm.get("date_from"), str) and isinstance(tm.get("date_to"), str):
            _check(tm["date_from"] <= tm["date_to"], "time.date_from is after time.date_to", errors)
        _check(isinstance(tm["conditions"], list) and all(c in CONDITIONS for c in tm["conditions"]),
               f"time.conditions must be from {CONDITIONS}", errors)
    _check(isinstance(spec["min_count"], int) and not isinstance(spec["min_count"], bool) and 1 <= spec["min_count"] <= 500,
           "min_count must be a whole number from 1 to 500", errors)
    _check(spec["uncertain"] in UNCERTAIN, f"uncertain must be one of {UNCERTAIN}", errors)
    for k in ("not_understood", "notes"):
        _check(isinstance(spec[k], list) and all(isinstance(x, str) for x in spec[k]), f"{k} must be a list of text", errors)
    return errors


def detector_classes(t: dict) -> tuple[str, ...]:
    """The detector classes an object may have to be considered for a target."""
    if t["subtype"]:
        # The detector may label a pickup as a car or a truck, so the subtype decides, not the category word.
        return SUBTYPES[t["subtype"]]
    return DETECTOR_CLASSES if t["category"] == "vehicle" else (t["category"],)


# ---------------------------------------------------------------- parsing
_TIME = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"
_DATE = r"(\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4})"


def _hhmm(h: str, m: str | None, ap: str | None) -> str | None:
    hh, mm = int(h), int(m or 0)
    if ap:
        if not 1 <= hh <= 12:
            return None
        pm = ap.startswith("p")
        hh = (hh % 12) + (12 if pm else 0)
    if hh == 24 and mm == 0:
        hh = 0
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


def _date(s: str) -> str | None:
    if "-" in s:
        y, mo, d = s.split("-")
    else:
        d, mo, y = re.split(r"[./]", s)
    try:
        from datetime import date
        return date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return None


def interpret(text: str, ctx: Context | None = None) -> dict:
    """Turn a request such as "red pickup trucks between 17:00 and 05:00" into a spec. Never raises on text."""
    ctx = ctx or Context()
    spec = empty_spec()
    notes: list[str] = spec["notes"]
    unknown: list[str] = spec["not_understood"]
    s = " " + text.lower().replace("’", "'").replace("–", "-").replace("—", "-") + " "

    def cut(m: re.Match) -> None:
        nonlocal s
        s = s[:m.start()] + " " * (m.end() - m.start()) + s[m.end():]

    # --- regions by name (longest names first so "parking lot 2" wins over "parking lot")
    for name in sorted(ctx.region_names, key=len, reverse=True):
        if not name.strip():
            continue
        pat = re.compile(r"\b(outside(?: of)?|not in|except in|in|inside|within)\s+(?:the\s+)?(?:region\s+)?['\"]?"
                         + re.escape(name.lower()) + r"['\"]?(?:\s+region)?(?=\W)")
        for m in list(pat.finditer(s))[::-1]:
            key = "exclude" if m.group(1).startswith(("outside", "not", "except")) else "include"
            if name not in spec["regions"][key]:
                spec["regions"][key].insert(0, name)
            cut(m)
    m = re.search(r"\b(?:in|inside|within|outside(?: of)?)\s+(?:the\s+)?region\s+['\"]?([\w\- ]+?)['\"]?(?=[,.;]|\s+(?:and|or|between|from|after|before|on)\b|\s*$)", s)
    if m:
        unknown.append(f"region '{m.group(1).strip()}' (no region of that name in this mission)")
        cut(m)

    # --- uncertainty treatment
    for pat, val in ((r"\b(only|just)\s+(certain|sure|definite|confirmed|clear)(\s+(ones|matches|results))?\b", "exclude"),
                     (r"\b(no|without|exclude|skip)\s+(uncertain|unsure|possible|maybe)(\s+(ones|matches|results))?\b", "exclude"),
                     (r"\b(include|show|list|with)\s+(uncertain|unsure|possible)(\s+(ones|matches|results))?\b", "list_separately")):
        m = re.search(pat, s)
        if m:
            spec["uncertain"] = val
            cut(m)

    # --- count
    m = re.search(r"\b(?:at least|min(?:imum)?(?: of)?|(?:with\s+)?(\d+)\s+or more)\s*(\d+)?\b", s)
    if m and (m.group(1) or m.group(2)):
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= 500:
            spec["min_count"] = n
            notes.append(f"Images need at least {n} matching objects.")
        else:
            unknown.append(m.group(0).strip())
        cut(m)

    # --- dates
    m = re.search(r"\b(?:from|between)\s+" + _DATE + r"\s+(?:to|until|till|and|-)\s+" + _DATE + r"\b", s)
    if m:
        a, b = _date(m.group(1)), _date(m.group(2))
        if a and b and a <= b:
            spec["time"]["date_from"], spec["time"]["date_to"] = a, b
        else:
            unknown.append(m.group(0).strip())
        cut(m)
    for pat, keys in ((r"\b(?:on|for)\s+" + _DATE + r"\b", ("date_from", "date_to")),
                      (r"\b(?:since|after|from)\s+" + _DATE + r"\b", ("date_from",)),
                      (r"\b(?:until|till|before|up to)\s+" + _DATE + r"\b", ("date_to",))):
        m = re.search(pat, s)
        if m:
            d = _date(m.group(1))
            if d:
                for k in keys:
                    spec["time"][k] = d
            else:
                unknown.append(m.group(0).strip())
            cut(m)
    if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{4}", text):
        notes.append("Dates written with dots or slashes are read as day.month.year.")

    # --- time windows
    clock = []
    for pat in (r"\b(?:between|from)\s+" + _TIME + r"\s*(?:and|to|until|till|-)\s*" + _TIME + r"(?=\W)",
                r"(?<![\d:])" + r"(\d{1,2}):(\d{2})()\s*-\s*(\d{1,2}):(\d{2})()" + r"(?=\W)"):
        for m in list(re.finditer(pat, s))[::-1]:
            g = m.groups()
            a, b = _hhmm(*g[0:3]), _hhmm(*g[3:6])
            if a and b and a != b:
                clock.insert(0, (a, b))
            else:
                unknown.append(m.group(0).strip())
            cut(m)
    for m in list(re.finditer(r"\b(after|since|before|until|till)\s+" + _TIME + r"(?=\W)", s))[::-1]:
        t = _hhmm(m.group(2), m.group(3), m.group(4))
        if t is None or (m.group(2) and int(m.group(2)) > 24 and not m.group(3)):
            unknown.append(m.group(0).strip())
        elif m.group(1) in ("after", "since"):
            clock.insert(0, (t, "24:00") if t != "00:00" else None)
            notes.append(f"'{m.group(0).strip()}' is read as {t} until midnight.")
        else:
            clock.insert(0, ("00:00", t) if t != "00:00" else None)
            notes.append(f"'{m.group(0).strip()}' is read as midnight until {t}.")
        cut(m)
    clock = [c for c in clock if c]

    # --- condition
    for pat, cond in ((r"\b(at|by|during the|during)\s+night(time)?\b|\bnight(time)?\b|\bin the dark\b", "night"),
                      (r"\b(during the day|in daylight|by day|daytime|daylight)\b", "day"),
                      (r"\b(at|during)\s+(dusk|dawn|twilight)\b|\b(dusk|dawn|twilight)\b", "dusk")):
        m = re.search(pat, s)
        if m:
            if cond not in spec["time"]["conditions"]:
                spec["time"]["conditions"].append(cond)
            cut(m)
    if spec["time"]["conditions"]:
        notes.append("Day, dusk and night are measured from each image's content, not from the clock.")

    # --- weekdays
    days: list[str] = []
    m = next((mm for mm in re.finditer(r"\b(?:from\s+)?([a-z]+\.?)\s+(?:to|through|thru|until|till|-)\s+([a-z]+\.?)(?=\W)", s)
              if mm.group(1) in _DAY_WORDS and mm.group(2) in _DAY_WORDS), None)
    if m:
        a, b = WEEKDAYS.index(_DAY_WORDS[m.group(1)]), WEEKDAYS.index(_DAY_WORDS[m.group(2)])
        i = a
        while True:
            days.append(WEEKDAYS[i])
            if i == b:
                break
            i = (i + 1) % 7
        cut(m)
    m = re.search(r"\b(weekends?|weekdays?|working days?|business days?|workdays?)\b", s)
    if m:
        unknown.append(f"'{m.group(1)}' (the days differ by country; name the days, e.g. 'friday and saturday')")
        cut(m)

    # --- tokens
    if re.search(r"\bsilver\b", s):
        notes.append("Silver is searched as grey: the colour reader does not tell them apart.")
    s = re.sub(r"\bpick up\b", "pickup", s)
    s = re.sub(r"\bsemi[ -]trailers?\b", lambda mm: "semi-trailers" if mm.group(0).endswith("s") else "semi-trailer", s)
    s = re.sub(r"\bsport utility vehicles?\b", "suv", s)
    s = re.sub(r"\bdark gr[ae]y\b", "grey", s)
    s = re.sub(r"\bnon-", "non ", s)
    s = re.sub(r"\bpickup trucks?\b", lambda mm: "pickup", s)
    s = re.sub(r"\b(\w+) ones\b", r"\1", s)
    tokens = re.findall(r"[a-z][a-z\-\.']*|\d+|[,;]", s)

    # Split into clauses at commas, ';' and 'and'/'or' that follow a noun.
    neg = False
    pending_colours: list[str] = []
    pending_neg_colours: list[str] = []
    pending_subtype: str | None = None
    cur: list[dict] = []           # targets of the current noun phrase
    all_targets: list[dict] = []

    def flush_adjectives_to(ts: list[dict]) -> None:
        nonlocal pending_colours, pending_neg_colours
        for t in ts:
            t["colours"] = sorted(set(t["colours"]) | set(pending_colours))
            t["exclude_colours"] = sorted(set(t["exclude_colours"]) | set(pending_neg_colours))
        pending_colours, pending_neg_colours = [], []

    i = 0
    while i < len(tokens):
        w = tokens[i].strip(".'")
        w_raw = tokens[i]
        if w_raw in (",", ";"):
            neg = False
            i += 1
            continue
        if w in ("and", "or"):
            # "red and blue cars": stay in the adjective list. "cars and buses": new noun phrase.
            nxt = tokens[i + 1].strip(".'") if i + 1 < len(tokens) else ""
            if not (pending_colours or pending_neg_colours) or nxt not in _COLOUR_WORDS:
                if nxt not in ("not", "no", "but", "except", "excluding", "without"):
                    neg = False
            i += 1
            continue
        if w in _NEG:
            neg = True
            i += 1
            continue
        if w in _COLOUR_WORDS:
            c = _COLOUR_WORDS[w]
            (pending_neg_colours if neg else pending_colours).append(c)
            i += 1
            continue
        if w in _SUBTYPE_WORDS:
            st = _SUBTYPE_WORDS[w]
            nxt = tokens[i + 1].strip(".'") if i + 1 < len(tokens) else ""
            if neg and not pending_colours:
                spec["exclude"]["subtypes"].append(st)
                neg = False
                i += 1
                continue
            if nxt in _CATEGORY_WORDS:     # "pickup truck"-style compound
                pending_subtype = st
                i += 1
                continue
            cat = _SUBTYPE_CATEGORY[st]
            t = target(cat, st)
            flush_adjectives_to([t])
            cur = [t]
            all_targets.append(t)
            neg = False
            i += 1
            continue
        if w in _CATEGORY_WORDS:
            cat = _CATEGORY_WORDS[w]
            if neg and not pending_colours and not pending_neg_colours and pending_subtype is None:
                # "except buses", "no trucks"
                if cat == "vehicle":
                    unknown.append(w)
                else:
                    spec["exclude"]["categories"].append(cat)
                neg = False
                i += 1
                continue
            st = pending_subtype
            pending_subtype = None
            t = target(cat, st)
            if neg and pending_colours and not pending_neg_colours:
                # "not red cars" - the colours are negated
                pending_neg_colours, pending_colours = pending_colours, []
            flush_adjectives_to([t])
            cur = [t]
            all_targets.append(t)
            neg = False
            i += 1
            continue
        if w in _STOP:
            i += 1
            continue
        if w in ("ones", "them", "those", "these"):
            i += 1
            continue
        unknown.append(w_raw)
        i += 1

    # Trailing adjectives: "cars that are red", "trucks, not white"
    if (pending_colours or pending_neg_colours) and cur:
        flush_adjectives_to(cur)
    elif pending_colours or pending_neg_colours:
        t = target("vehicle")
        flush_adjectives_to([t])
        all_targets.append(t)
        notes.append("No vehicle type was named, so any vehicle class is searched.")
    if pending_subtype:
        t = target(_SUBTYPE_CATEGORY[pending_subtype], pending_subtype)
        all_targets.append(t)

    if not all_targets:
        all_targets.append(target("vehicle"))
        notes.append("No vehicle type was named, so any vehicle class is searched.")
    # Merge identical targets
    seen = []
    for t in all_targets:
        if t not in seen:
            seen.append(t)
    spec["targets"] = seen
    for t in seen:
        clash = set(t["colours"]) & set(t["exclude_colours"])
        if clash:
            t["exclude_colours"] = [c for c in t["exclude_colours"] if c not in clash]
            unknown.append(f"colour both wanted and excluded: {', '.join(sorted(clash))}")
        if t["subtype"]:
            notes.append(f"'{t['subtype'].replace('_', '-')}' is not identified by the current detector: matching "
                         f"{' and '.join(detector_classes(t))} detections are listed as possible, never as certain.")
    spec["exclude"]["categories"] = sorted(set(spec["exclude"]["categories"]))
    spec["exclude"]["subtypes"] = sorted(set(spec["exclude"]["subtypes"]))
    if any(t["colours"] or t["exclude_colours"] for t in seen):
        notes.append("Colour cannot be seen in infrared night images; there it is reported as not observable.")

    if clock:
        spec["time"]["windows"] = [{"days": list(days or WEEKDAYS), "start": a, "end": b} for a, b in clock]
    elif days:
        spec["time"]["windows"] = [{"days": days, "start": "00:00", "end": "24:00"}]
    # single day names like "on sunday" (not caught by the range rule) were left as tokens
    single = [d for d in spec["not_understood"] if d.strip(".") in _DAY_WORDS]
    if single:
        dd = sorted({_DAY_WORDS[d.strip('.')] if d.strip('.') in _DAY_WORDS else d for d in single}, key=WEEKDAYS.index)
        spec["not_understood"] = [d for d in spec["not_understood"] if d not in single]
        if spec["time"]["windows"] and not days:
            for w in spec["time"]["windows"]:
                w["days"] = dd
        elif not spec["time"]["windows"]:
            spec["time"]["windows"] = [{"days": dd, "start": "00:00", "end": "24:00"}]
    for w in spec["time"]["windows"]:
        w["days"] = sorted(set(w["days"]), key=WEEKDAYS.index)
        if w["end"] <= w["start"]:
            notes.append(f"{w['start']}-{w['end']} crosses midnight: it starts on the selected days and ends the next morning.")
    if spec["time"]["windows"] or spec["time"]["date_from"] or spec["time"]["date_to"]:
        notes.append("Times are local time in the mission's time zone.")

    errs = validate_spec(spec)
    if errs:  # a parser bug must never produce an invalid spec silently
        raise SpecError("; ".join(errs))
    return spec


def describe(spec: dict) -> list[str]:
    """Plain-English lines for the confirmation screen."""
    out = []
    for t in spec["targets"]:
        name = t["category"] + "s" if t["category"] != "bus" else "buses"
        if t["subtype"]:
            name = f"{t['subtype'].replace('_', '-')}s ({', '.join(detector_classes(t))} detections)"
        bits = []
        if t["colours"]:
            bits.append(" or ".join(t["colours"]))
        line = f"Find {' '.join(bits) + ' ' if bits else ''}{name}"
        if t["exclude_colours"]:
            line += f", not {' or '.join(t['exclude_colours'])}"
        out.append(line + ".")
    ex = spec["exclude"]
    if ex["categories"] or ex["subtypes"]:
        out.append("Leave out: " + ", ".join(ex["categories"] + [s.replace("_", "-") for s in ex["subtypes"]]) + ".")
    rg = spec["regions"]
    out.append("Where: " + (("inside " + ", ".join(rg["include"])) if rg["include"] else "the view's analysis regions")
               + ((", outside " + ", ".join(rg["exclude"])) if rg["exclude"] else "") + ".")
    tm = spec["time"]
    when = []
    for w in tm["windows"]:
        when.append(f"{','.join(w['days']) if len(w['days']) < 7 else 'every day'} {w['start']}-{w['end']}")
    if tm["date_from"] or tm["date_to"]:
        when.append(f"dates {tm['date_from'] or '…'} to {tm['date_to'] or '…'}")
    if tm["conditions"]:
        when.append("conditions: " + ", ".join(tm["conditions"]))
    out.append("When: " + ("; ".join(when) if when else "any time") + ".")
    if spec["min_count"] > 1:
        out.append(f"Only images with at least {spec['min_count']} matches.")
    out.append("Uncertain colour or type: " + ("listed separately as possible." if spec["uncertain"] == "list_separately"
                                             else "left out (only certain matches)."))
    return out


def apply_edits(spec: dict, edited: dict) -> tuple[dict, dict]:
    """Return (validated edited spec, diff) for an operator-edited spec. Raises SpecError when invalid."""
    errs = validate_spec(edited)
    if errs:
        raise SpecError("; ".join(errs))
    diff = {k: {"from": spec.get(k), "to": edited[k]} for k in edited if spec.get(k) != edited[k]}
    return copy.deepcopy(edited), diff
