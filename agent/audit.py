"""
what: the audit trail -- one JSON line per thing the agent intends to do.
why:  rule 4, "audit before action". Stage 3's run restarted a real container
      that had already died 1,537 times, and there is no record anywhere that it
      happened: not in the trace (gone with the process), not in memory (that
      stores incidents, not actions). This file is that gap.

      Writing AFTER the action would only record the actions that finished --
      exactly the ones you least need to investigate. An action that crashes
      halfway, or hangs, or takes down the process, must still leave a line.
how:  JSONL: one JSON object per line, opened in append mode, flushed AND
      fsynced. Append mode and one-line records mean two processes writing at
      once interleave whole lines rather than corrupting each other, and a
      half-written file still parses up to its last complete line.

      This module is imported directly by graph.py, exactly like the dispatcher
      and for the same reason: an audit that run.py could swap for a quieter one
      is not an audit. The path is configurable; whether to write is not.
"""

import json
import os
from datetime import datetime, timezone

# Overridable so a container or the VPS can put it on a mounted volume. It is
# gitignored: the trail is run history, and on the VPS it is evidence.
DEFAULT_PATH = os.environ.get("LAB_AGENT_AUDIT", "audit.jsonl")


def write(record: dict, path: str | None = None) -> dict:
    """Append one record, stamped with the time it was written."""
    # A caller's own "ts" is dropped before stamping. The first version of this
    # function spread the record AFTER the stamp, so a supplied ts silently
    # replaced the real one -- a record that lies about its own time, which is
    # worse than no record. A test caught it; this is what the test protects.
    supplied = {key: value for key, value in record.items() if key != "ts"}
    # ts first, so every line begins the same way and the file sorts
    # chronologically as plain text.
    stamped = {"ts": datetime.now(timezone.utc).isoformat(), **supplied}
    with open(path or DEFAULT_PATH, "a", encoding="utf-8") as handle:
        # default=str so an AgentDecision, a Path or a datetime becomes text
        # instead of raising. An audit line that cannot be written because a
        # value was awkward is the worst possible failure for this file.
        handle.write(json.dumps(stamped, default=str) + "\n")
        handle.flush()  # out of Python's buffer
        os.fsync(handle.fileno())  # and out of the OS cache, onto the disk
    return stamped


def read_all(path: str | None = None) -> list[dict]:
    """Every record in the trail, oldest first. Missing file means no history."""
    target = path or DEFAULT_PATH
    if not os.path.exists(target):
        return []
    records = []
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated last line is what a crash mid-write looks like.
                # Skipping it keeps every COMPLETE record readable, which is the
                # reason for one-object-per-line in the first place.
                continue
    return records
