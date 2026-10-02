"""Alert fan out with level filtering and a cooldown.

Channels are deliberately simple and synchronous. Each one is isolated, so a
dead webhook can never stop the incident being written to the local log.
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from abc import ABC, abstractmethod
from email.message import EmailMessage
from pathlib import Path

from pyraguard.config import PROJECT_ROOT, AlertConfig
from pyraguard.logging_utils import get_logger
from pyraguard.schemas import FrameResult, HazardLevel, IncidentEvent

log = get_logger(__name__)


def build_payload(event: IncidentEvent, result: FrameResult) -> dict:
    payload = {"event": event.to_dict(), "assessment": result.assessment.to_dict(), "camera_id": result.camera_id}
    if result.plan is not None:
        payload["plan"] = {
            "summary": result.plan.summary,
            "actions": [a.to_dict() for a in result.plan.actions],
            "prohibitions": [p.to_dict() for p in result.plan.prohibitions],
            "routes": [r.to_dict() for r in result.plan.routes],
            "blocked_zones": result.plan.blocked_zones,
            "groundedness": round(result.plan.groundedness, 3),
        }
    return payload


def render_text(payload: dict) -> str:
    event = payload["event"]
    lines = [f"PyraGuard {event['kind'].upper()}: {event['level']} at camera {event['camera_id']} (zone {event['zone_id']}), score {event['score']:.0f}"]
    plan = payload.get("plan")
    if plan:
        lines.append(plan["summary"])
        lines.extend(f"  {i}. {a['text']} [{', '.join(a['citations'])}]" for i, a in enumerate(plan["actions"], 1))
        lines.extend(f"  DO NOT: {p['text']} [{', '.join(p['citations'])}]" for p in plan["prohibitions"])
        lines.extend(f"  ROUTE: {r['instructions']}" for r in plan["routes"][:4])
    return "\n".join(lines)


class AlertChannel(ABC):
    name = "channel"

    @abstractmethod
    def send(self, payload: dict) -> None: ...


class ConsoleChannel(AlertChannel):
    name = "console"

    def send(self, payload: dict) -> None:
        log.warning("\n%s", render_text(payload))


class JsonlChannel(AlertChannel):
    name = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class WebhookChannel(AlertChannel):
    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url, self.timeout = url, timeout

    def send(self, payload: dict) -> None:
        body = json.dumps({"text": render_text(payload), **payload}).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=self.timeout).close()  # noqa: S310 - operator configured URL


class EmailChannel(AlertChannel):
    name = "email"

    def __init__(self) -> None:
        self.host = os.environ.get("PYRAGUARD_SMTP_HOST", "")
        self.port = int(os.environ.get("PYRAGUARD_SMTP_PORT", "587"))
        self.user = os.environ.get("PYRAGUARD_SMTP_USER", "")
        self.password = os.environ.get("PYRAGUARD_SMTP_PASSWORD", "")
        self.to = os.environ.get("PYRAGUARD_ALERT_EMAIL_TO", "")
        if not (self.host and self.to):
            raise RuntimeError("PYRAGUARD_SMTP_HOST and PYRAGUARD_ALERT_EMAIL_TO must be set for email alerts")

    def send(self, payload: dict) -> None:
        message = EmailMessage()
        event = payload["event"]
        message["Subject"] = f"PyraGuard {event['level']} at {event['camera_id']}"
        message["From"] = self.user or "pyraguard@localhost"
        message["To"] = self.to
        message.set_content(render_text(payload))
        with smtplib.SMTP(self.host, self.port, timeout=10) as smtp:
            smtp.starttls()
            if self.user:
                smtp.login(self.user, self.password)
            smtp.send_message(message)


def build_channels(cfg: AlertConfig) -> list[AlertChannel]:
    channels: list[AlertChannel] = []
    for name in cfg.channels:
        try:
            if name == "console":
                channels.append(ConsoleChannel())
            elif name == "jsonl":
                path = Path(cfg.jsonl_path)
                channels.append(JsonlChannel(path if path.is_absolute() else PROJECT_ROOT / path))
            elif name == "webhook":
                url = os.environ.get("PYRAGUARD_WEBHOOK_URL", "")
                if url:
                    channels.append(WebhookChannel(url))
            elif name == "email":
                channels.append(EmailChannel())
            else:
                log.warning("Unknown alert channel: %s", name)
        except Exception as exc:
            log.warning("Alert channel %s disabled: %s", name, exc)
    return channels


class AlertDispatcher:
    def __init__(self, config: AlertConfig | None = None, channels: list[AlertChannel] | None = None) -> None:
        self.cfg = config or AlertConfig()
        self.channels = channels if channels is not None else build_channels(self.cfg)
        self.min_level = HazardLevel[self.cfg.min_level]
        self._last_sent: dict[str, float] = {}
        self.sent: list[dict] = []

    def dispatch(self, event: IncidentEvent, result: FrameResult) -> bool:
        """Send the event to every channel. Returns True when an alert went out."""
        reportable = event.kind == "closed" or event.level >= self.min_level
        if not reportable:
            return False
        last = self._last_sent.get(event.camera_id)
        # escalations always go out; repeats of the same state respect the cooldown
        if event.kind not in ("opened", "escalated", "closed") and last is not None and event.timestamp - last < self.cfg.cooldown_seconds:
            return False
        payload = build_payload(event, result)
        for channel in self.channels:
            try:
                channel.send(payload)
            except Exception as exc:
                log.error("Alert channel %s failed: %s", channel.name, exc)
        self._last_sent[event.camera_id] = event.timestamp
        self.sent.append(payload)
        return True
