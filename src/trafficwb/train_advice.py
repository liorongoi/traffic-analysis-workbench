"""Advice after a training round, in plain words.

Two parts, both computed when a round ends and kept with it:

* ``curve_advice``: reads the training and validation loss per epoch (and what the dataset and the test held) and
  says what would most likely make the next model better: more epochs, more or more varied images, steadier labels,
  more examples of a rare class, more images of a light condition where vehicles are missed.
* ``swap_advice``: the new model beside the model the mission uses now, on exactly the same test images, and whether
  to replace it or keep it. Recall (vehicles found of those there) decides first, precision (found boxes that are
  vehicles) second. It only advises: the model in use changes when the operator chooses it.

Pure functions, no database or model: the training job gathers the figures and stores the result.
"""
from __future__ import annotations

import math

CONDITION_NAMES = {"day": "daylight", "dusk": "dusk", "night": "night", "unknown": "unknown light"}


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _item(kind: str, level: str, title: str, text: str, **extra) -> dict:
    return {"kind": kind, "level": level, "title": title, "text": text, **extra}


def _suggest_epochs(n: int) -> int:
    """The next of the epoch choices on the page above n (they go 10, 20, 30, 40, 60, 80, 100, 150, 200)."""
    for x in (10, 20, 30, 40, 60, 80, 100, 150, 200):
        if x >= n:
            return x
    return 200


def curve_advice(history: list[dict], epochs: int | None = None, *, val_before: float | None = None,
                 train_images: int | None = None, val_images: int | None = None,
                 class_counts: dict[str, int] | None = None, overall: dict | None = None,
                 by_condition: dict[str, dict] | None = None) -> dict:
    """What the loss curves (and the data behind them) say, and what to do next.

    history: [{epoch, train_loss, val_loss}] as the trainer records it. Returns {"verdict", "summary", "items"},
    items most important first, each {kind, level (warn | tip | ok), title, text}.
    """
    h = [r for r in history or [] if r.get("train_loss") is not None]
    items: list[dict] = []
    verdict = "unknown"
    n = len(h)
    if n == 0:
        return {"verdict": verdict, "summary": "No loss was recorded, so there is nothing to read.", "items": []}
    tl = [float(r["train_loss"]) for r in h]
    vl = [float(r["val_loss"]) for r in h if r.get("val_loss") is not None]
    has_val = len(vl) == n

    if n < 3:
        items.append(_item("few_epochs", "warn", "Too few epochs to read the curves",
                           f"The round ran {n} epoch{'s' if n != 1 else ''}. Train for at least 20 epochs so the curves "
                           "show whether the model is still learning or has started to learn the images by heart."))
        verdict = "too_short"
    if not has_val:
        items.append(_item("no_validation", "warn", "No validation images",
                           "Without validation images the curve cannot show when the model starts to learn the training "
                           "images by heart, and the last epoch is kept whatever happened. Freeze a dataset version with a "
                           "validation share (for example 70/15/15)."))
    train_drop = (tl[0] - min(tl)) / tl[0] if tl[0] > 0 else 0.0

    if has_val and n >= 3:
        best_i = min(range(n), key=lambda i: vl[i])
        best_ep = int(h[best_i].get("epoch", best_i + 1))
        lo = vl[best_i]
        end_rise = (vl[-1] - lo) / lo if lo > 0 else 0.0
        k = max(2, n // 4)
        tail_drop = (vl[-k] - vl[-1]) / vl[-k] if vl[-k] > 0 else 0.0
        diffs = [b - a for a, b in zip(vl, vl[1:])]
        flips = sum(1 for a, b in zip(diffs, diffs[1:]) if a * b < 0)
        mean_v = sum(vl) / n
        jump = sorted(abs(d) for d in diffs)[len(diffs) // 2] / mean_v if mean_v > 0 else 0.0
        noisy = n >= 5 and flips / max(len(diffs) - 1, 1) >= 0.6 and jump >= 0.03

        if val_before is not None and lo >= 0.98 * val_before:
            verdict = "no_gain"
            items.append(_item("no_gain", "warn", "Training did not lower the validation loss",
                               f"Validation loss was {val_before:.3f} before training and its best was {lo:.3f}. The new "
                               "images taught the model nothing it did not know already. Most often the boxes on the "
                               "reviewed images disagree with each other (a vehicle boxed on one image and not on the "
                               "next), or the new images are too few or too alike. Check the reviews, then add images "
                               "the model gets wrong."))
        if best_ep <= max(2, int(0.7 * n)) and end_rise > 0.05 and tl[-1] < tl[best_i]:
            verdict = "overfitting" if verdict in ("unknown", "too_short") else verdict
            items.append(_item("overfitting", "warn", f"From epoch {best_ep} the model learned the training images by heart",
                               f"After epoch {best_ep} the training loss kept falling while the validation loss rose "
                               f"{100 * end_rise:.0f}% again. The model kept is the one from epoch {best_ep}, so nothing is "
                               "lost, but more epochs will not help. What helps: more reviewed images, and more varied "
                               "ones (other cameras, hours, weather, traffic), so there is more to learn than these "
                               f"images. Next time about {_suggest_epochs(best_ep + max(3, best_ep // 4))} epochs is enough.",
                               suggested_epochs=_suggest_epochs(best_ep + max(3, best_ep // 4))))
        elif best_i >= n - 2 and tail_drop > 0.02:
            verdict = "still_learning" if verdict in ("unknown", "too_short") else verdict
            more = _suggest_epochs(max(2 * n, n + 10))
            items.append(_item("still_learning", "tip", "It was still improving when training ended",
                               f"Validation loss fell {100 * tail_drop:.0f}% over the last {k} epochs and was lowest at the "
                               f"end. Train again with more epochs (about {more}), or start the next round from this new "
                               "model so it carries on from here.", suggested_epochs=more))
        elif verdict == "unknown":
            verdict = "converged"
            items.append(_item("converged", "ok", f"The curves levelled off at epoch {best_ep}",
                               "Training and validation loss settled and did not part ways: the training itself went "
                               "well. More epochs will not add much; more reviewed images, above all of what the model "
                               "still misses, will."))
        if noisy:
            items.append(_item("noisy", "tip", "Validation loss jumps up and down",
                               "From one epoch to the next it moves about "
                               f"{100 * jump:.0f}% in alternating directions. "
                               + (f"There are only {val_images} validation images, so a few of them move the figure. "
                                  if val_images is not None and val_images < 30 else "")
                               + "Review more images so the validation share grows, and box every vehicle the same way "
                               "on every reviewed image (partly hidden ones too, or none of them)."))
    if n >= 3 and train_drop < 0.05:
        items.append(_item("not_learning", "warn", "Training loss hardly fell",
                           f"It fell only {100 * train_drop:.0f}% over {n} epochs: the model found little it could learn. "
                           "Check that the reviewed boxes are right and tight, and that the dataset is not mostly empty "
                           "images."))
        if verdict == "unknown":
            verdict = "not_learning"

    if train_images is not None:
        if train_images < 50:
            items.append(_item("few_images", "warn", f"Only {train_images} training images",
                               "A detector usually needs a few hundred reviewed images of a camera view before training "
                               "improves it reliably. Review more images, then freeze a new dataset version."))
        elif train_images < 200:
            items.append(_item("few_images", "tip", f"{train_images} training images",
                               "More reviewed images, above all of hours and conditions not yet in the dataset, are the "
                               "surest way to a better model."))
    if val_images is not None and 0 < val_images < 10 and has_val:
        items.append(_item("few_validation", "tip", f"Only {val_images} validation images",
                           "With so few the validation loss, and so the epoch kept, depends on chance. A larger "
                           "validation share helps (15% of the images or more)."))

    counts = {c: int(v) for c, v in (class_counts or {}).items() if v}
    total = sum(counts.values())
    if total >= 20 and len(counts) > 1:
        rare = sorted((c for c, v in counts.items() if v < 15), key=lambda c: counts[c])
        top = max(counts, key=lambda c: counts[c])
        if rare:
            items.append(_item("rare_class", "tip", "Few examples of " + ", ".join(rare),
                               "The training images hold only " + ", ".join(f"{counts[c]} boxes of {c}" for c in rare)
                               + f", against {counts[top]} of {top}. The model can hardly learn a class from a dozen "
                               "examples: review images that show them."))
        elif counts[top] / total > 0.9:
            items.append(_item("imbalance", "tip", f"Almost every example is a {top}",
                               f"{100 * counts[top] / total:.0f}% of the boxes are {top}. Other vehicle types will be "
                               "found less well; review images that show them."))

    if overall:
        r, p = overall.get("recall"), overall.get("precision")
        if r is not None and r < 0.85 and (p is None or p >= r + 0.05):
            items.append(_item("missed", "tip", f"It misses vehicles more than it invents them (recall {_pct(r)})",
                               f"It found {overall.get('true_positives', '?')} of {overall.get('references', '?')} "
                               "vehicles in the test images. To raise recall, review images where vehicles were missed "
                               "and box them: that is what teaches the model to find them."))
        if p is not None and p < 0.7:
            items.append(_item("false_alarms", "tip", f"Many boxes are not vehicles (precision {_pct(p)})",
                               "Review whole images (approve or correct every box) where it boxes lights, signs or "
                               "shadows: only fully reviewed images teach it what is not a vehicle."))
    if overall and by_condition and overall.get("recall") is not None:
        worst = [(c, v) for c, v in by_condition.items()
                 if v.get("recall") is not None and (v.get("references") or 0) >= 10
                 and v["recall"] < overall["recall"] - 0.1]
        for c, v in sorted(worst, key=lambda x: x[1]["recall"])[:2]:
            name = CONDITION_NAMES.get(c, c)
            items.append(_item("condition", "tip", f"Weakest in {name} (recall {_pct(v['recall'])})",
                               f"Against {_pct(overall['recall'])} overall. Review more {name} images so the next "
                               "dataset version holds more of them.", condition=c))

    order = {"warn": 0, "tip": 1, "ok": 2}
    items.sort(key=lambda x: order.get(x["level"], 3))
    summary = {
        "overfitting": "The model started to learn the training images by heart: more and more varied images will help, "
                       "more epochs will not.",
        "still_learning": "The model was still improving when training stopped: more epochs will help.",
        "converged": "Training went well; the next gain comes from more reviewed images, not more epochs.",
        "no_gain": "This round taught the model nothing new: check the reviews and add images it gets wrong.",
        "not_learning": "The model hardly learned: check the reviewed boxes and the images in the dataset.",
        "too_short": "Too few epochs to judge: train for longer.",
    }.get(verdict, "See below what would most likely help.")
    return {"verdict": verdict, "summary": summary, "items": items}


def swap_advice(new: dict | None, current: dict | None, *, new_by_condition: dict | None = None,
                current_by_condition: dict | None = None, current_label: str = "the model in use",
                same_as_start: bool = False) -> dict:
    """Replace the model in use with the new one, or keep it? Recall first, then precision, on the same test images.

    new/current: the "overall" part of each evaluation report. Returns {"verdict": replace | keep | unclear,
    "headline", "reasons": [...], "cautions": [...], "figures": {...}}. Never acts by itself.
    """
    if not new or not current:
        return {"verdict": "unclear", "headline": "The new model could not be compared with " + current_label + ".",
                "reasons": ["Evaluate both on the same test images by hand below, then decide."], "cautions": [],
                "figures": {}}
    refs = current.get("references") or new.get("references") or 0
    r_new, r_cur = new.get("recall"), current.get("recall")
    p_new, p_cur = new.get("precision"), current.get("precision")
    f_new, f_cur = new.get("true_positives") or 0, current.get("true_positives") or 0
    figures = {"references": refs, "found_new": f_new, "found_current": f_cur,
               "recall_new": r_new, "recall_current": r_cur, "precision_new": p_new, "precision_current": p_cur,
               "f1_new": new.get("f1"), "f1_current": current.get("f1"),
               "ap50_new": new.get("ap50"), "ap50_current": current.get("ap50"),
               "false_alarms_new": new.get("false_positives_full"),
               "false_alarms_current": current.get("false_positives_full")}
    if r_new is None or r_cur is None or not refs:
        return {"verdict": "unclear", "headline": "The test images hold no labelled vehicles to compare on.",
                "reasons": ["Add reviewed images to the test share of a dataset version, then evaluate again."],
                "cautions": [], "figures": figures}
    tol = max(2, math.ceil(0.01 * refs))            # fewer vehicles than this either way: the same recall
    d_found = f_new - f_cur
    dp = None if p_new is None or p_cur is None else p_new - p_cur
    reasons: list[str] = []
    cautions: list[str] = []
    found_txt = (f"The new model finds {f_new} of the {refs} vehicles in the test images, {current_label} finds {f_cur} "
                 f"(recall {_pct(r_new)} against {_pct(r_cur)}).")
    prec_txt = ("" if dp is None else
                f"Of the boxes it draws, {_pct(p_new)} are vehicles, against {_pct(p_cur)}"
                + (f" ({new.get('false_positives_full')} wrong boxes against {current.get('false_positives_full')})"
                   if new.get("false_positives_full") is not None and current.get("false_positives_full") is not None else "")
                + ".")
    if d_found >= tol:
        verdict = "replace"
        headline = f"Replace: the new model finds {d_found} more vehicle{'s' if d_found != 1 else ''}."
        reasons.append(found_txt)
        if prec_txt:
            reasons.append(prec_txt)
        if dp is not None and dp < -0.1:
            cautions.append(f"It also draws more wrong boxes (precision down {100 * -dp:.0f} points). Recall comes "
                            "first, but expect more boxes to reject in review.")
    elif d_found <= -tol:
        verdict = "keep"
        headline = (f"Keep {current_label}: the new model misses {-d_found} more vehicle{'s' if -d_found != 1 else ''}.")
        reasons.append(found_txt)
        if dp is not None and dp > 0.02:
            reasons.append(prec_txt + " Fewer wrong boxes do not make up for missed vehicles.")
        elif prec_txt:
            reasons.append(prec_txt)
    else:
        reasons.append(found_txt + " That is the same recall within "
                       f"{tol} vehicle{'s' if tol != 1 else ''}.")
        if dp is not None and dp > 0.02:
            verdict = "replace"
            headline = "Replace: it finds as many vehicles with fewer wrong boxes."
            reasons.append(prec_txt)
        elif dp is not None and dp < -0.02:
            verdict = "keep"
            headline = f"Keep {current_label}: the new model finds as many vehicles but draws more wrong boxes."
            reasons.append(prec_txt)
        else:
            a_new, a_cur = new.get("ap50"), current.get("ap50")
            if a_new is not None and a_cur is not None and a_new - a_cur > 0.02:
                verdict = "replace"
                headline = "Replace, by a small margin: same recall and precision, better ranked boxes (AP)."
                reasons.append(f"AP at IoU 0.5 is {_pct(a_new)} against {_pct(a_cur)}.")
            else:
                verdict = "keep"
                headline = f"Keep {current_label}: the new model is no better on these test images."
                if prec_txt:
                    reasons.append(prec_txt)
    for c, v in sorted((new_by_condition or {}).items()):
        cur = (current_by_condition or {}).get(c) or {}
        if (cur.get("references") or 0) >= 10 and v.get("recall") is not None and cur.get("recall") is not None:
            d = v["recall"] - cur["recall"]
            name = CONDITION_NAMES.get(c, c)
            if d <= -0.05 and verdict == "replace":
                cautions.append(f"In {name} it finds fewer: recall {_pct(v['recall'])} against {_pct(cur['recall'])}.")
            elif d >= 0.05 and verdict == "keep":
                reasons.append(f"In {name} it does find more ({_pct(v['recall'])} against {_pct(cur['recall'])}); "
                               "a model for those images alone may be worth saving.")
    if refs < 15:
        verdict = "unclear"
        headline = f"Too few vehicles in the test images ({refs}) to tell the two apart."
        cautions.append("Review more images so the test share holds at least a few dozen vehicles, then evaluate again.")
    elif refs < 50:
        cautions.append(f"The test images hold only {refs} vehicles, so a difference of a few may be chance.")
    if same_as_start:
        reasons.append(f"{current_label[0].upper() + current_label[1:]} is also the model this round started from.")
    return {"verdict": verdict, "headline": headline, "reasons": reasons, "cautions": cautions, "figures": figures}
