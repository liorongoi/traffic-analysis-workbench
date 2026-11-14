"""Web API of the indoor/outdoor sorting mission (scenes.py)."""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from ..scenes import MODE, SceneService, available, learn, learner_summary, reset_learning
from ..store import Mission


class DecideBody(BaseModel):
    version_ids: list[int]
    label: str | None = None        # indoor | outdoor | None: back to LanePulse's choice


def sorted_folder(m: Mission) -> Path:
    cfg = m.config()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", cfg.name).strip(" .") or "mission"
    return Path(cfg.output_dir) / f"{name} - indoor and outdoor"


def register(app: FastAPI, data_root: Path, mission: Callable, user: Callable, api_error: type) -> None:

    def sorting(m: Mission) -> SceneService:
        if m.config().mode != MODE:
            raise api_error(400, "not_sorting", "This mission does not sort indoor and outdoor images.")
        return SceneService(m)

    @app.get("/api/scenes/status")
    def scenes_status(_u=Depends(user)):
        return available() | {"learner": learner_summary(data_root)}

    @app.get("/api/missions/{mission_id}/scenes")
    def scenes_list(m: Mission = Depends(mission)):
        svc = sorting(m)
        items = svc.items()
        return {"items": [asdict(i) for i in items], "counts": svc.counts(items), "learner": learner_summary(data_root),
                "folder": str(sorted_folder(m))}

    @app.post("/api/missions/{mission_id}/scenes/decide")
    def scenes_decide(body: DecideBody, m: Mission = Depends(mission), u=Depends(user)):
        svc = sorting(m)
        try:
            n = svc.decide(body.version_ids, body.label, u.username)
        except ValueError as exc:
            raise api_error(400, "bad_label", str(exc)) from None
        except KeyError as exc:
            raise api_error(404, "not_sorted", str(exc.args[0])) from None
        return {"decided": n, "counts": svc.counts()}

    @app.post("/api/missions/{mission_id}/scenes/learn")
    def scenes_learn(m: Mission = Depends(mission), u=Depends(user)):
        sorting(m)
        return learn(data_root, u.username) | {"learner": learner_summary(data_root)}

    @app.post("/api/missions/{mission_id}/scenes/forget")
    def scenes_forget(m: Mission = Depends(mission)):
        sorting(m)
        reset_learning(data_root)
        return {"learner": learner_summary(data_root)}

    @app.post("/api/missions/{mission_id}/scenes/copy")
    def scenes_copy(m: Mission = Depends(mission)):
        svc = sorting(m)
        try:
            return svc.copy_out(sorted_folder(m))
        except OSError as exc:
            raise api_error(500, "copy_failed", f"The images could not be copied: {exc}",
                            "Check that the exports folder exists, is writable and has free space.") from None
