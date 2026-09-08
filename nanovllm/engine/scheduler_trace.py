"""Bounded, opt-in flight recorder. Contains metadata, never prompt/token values."""

import json
import logging
from collections import deque
from pathlib import Path


class SchedulerTrace:
    def __init__(self, path, metadata, history=512, following=128,
                 max_triggers=32, max_bytes=16 * 1024 * 1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history = deque(maxlen=history)
        self.following = following
        self.remaining = 0
        self.max_triggers = max_triggers
        self.triggers = 0
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.event_id = 0
        self.last_written = -1
        self.enabled = True
        # Never silently mix runs or overwrite a previous diagnosis.
        with self.path.open("x", encoding="utf-8") as output:
            header = json.dumps({"event": "header", "schema_version": 1,
                                 "history_events": history, "following_events": following,
                                 "max_triggers": max_triggers, "max_bytes": max_bytes,
                                 **metadata}) + "\n"
            output.write(header)
            self.bytes_written = len(header.encode("utf-8"))

    def record(self, event, trigger=False):
        if not self.enabled:
            return
        row = {**event, "event_id": self.event_id}
        self.event_id += 1
        self.history.append(row)
        if trigger and self.triggers < self.max_triggers:
            self.triggers += 1
            self.remaining = self.following + 1
            rows = [item for item in self.history
                    if item["event_id"] > self.last_written]
        elif self.remaining:
            rows = [row]
        else:
            return
        try:
            payload = "".join(json.dumps(item, allow_nan=False) + "\n" for item in rows)
            size = len(payload.encode("utf-8"))
            if self.bytes_written + size > self.max_bytes:
                self.enabled = False
                logging.getLogger(__name__).warning("Scheduler trace byte limit reached: %s", self.path)
                return
            with self.path.open("a", encoding="utf-8") as output:
                output.write(payload)
            self.bytes_written += size
            self.last_written = rows[-1]["event_id"]
            self.remaining -= 1
            if self.triggers >= self.max_triggers and not self.remaining:
                self.enabled = False
        except (OSError, ValueError):
            self.enabled = False
            logging.getLogger(__name__).exception("Scheduler trace disabled after write failure")
