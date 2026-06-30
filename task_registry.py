"""
Task registry — loads task definitions from a YAML file and provides lookups.

A "task" here is one (lang_instruction, target_object, voice_keywords) triple.
Swap tasks.yaml to retarget the system to a new robot scene without touching code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import yaml


@dataclass(frozen=True)
class Task:
    name: str
    lang: str
    object: str
    keywords: List[str]
    # Optional per-task safety cap — overrides PolicyRouter's default
    # max_task_runtime_s. Use a longer value for multi-step tasks (e.g. pour).
    max_runtime_s: Optional[float] = None
    # Optional per-task visual completion criterion (a success-condition phrase,
    # e.g. "an orange monster can is resting on the tray"). Used by the dedicated
    # completion verifier to ask the right yes/no question for THIS task. If
    # unset, the verifier falls back to its default ball-in-bowl question, so
    # existing tasks keep working unchanged.
    completion_check: Optional[str] = None


class TaskRegistry:
    def __init__(self, tasks: List[Task]):
        if not tasks:
            raise ValueError("TaskRegistry needs at least one task")
        self._tasks = list(tasks)
        self._by_name = {t.name: t for t in self._tasks}
        if len(self._by_name) != len(self._tasks):
            raise ValueError("Duplicate task name in registry")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskRegistry":
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict) or "tasks" not in data:
            raise ValueError(f"{path}: expected top-level 'tasks:' list")
        tasks = []
        for i, entry in enumerate(data["tasks"]):
            for k in ("name", "lang", "object", "keywords"):
                if k not in entry:
                    raise ValueError(f"{path}: tasks[{i}] missing '{k}'")
            mr = entry.get("max_runtime_s")
            cc = entry.get("completion_check")
            tasks.append(Task(
                name=str(entry["name"]),
                lang=str(entry["lang"]),
                object=str(entry["object"]),
                keywords=[str(kw).lower() for kw in entry["keywords"]],
                max_runtime_s=float(mr) if mr is not None else None,
                completion_check=str(cc) if cc is not None else None,
            ))
        return cls(tasks)

    # ── lookups ───────────────────────────────────────────────

    def get(self, name: str) -> Task:
        return self._by_name[name]

    def names(self) -> List[str]:
        return [t.name for t in self._tasks]

    def objects(self) -> List[str]:
        return [t.object for t in self._tasks]

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._tasks)

    def resolve(self, text: str) -> Optional[str]:
        """Map a free-text voice command (or object description) to a task name.
        Returns the task whose longest keyword substring appears in text — prevents
        short keywords (e.g. 'ball') from shadowing more-specific ones ('pink ball')."""
        text_lower = text.lower().strip()
        best_task: Optional[str] = None
        best_len = -1
        for task in self._tasks:
            for kw in task.keywords:
                if kw in text_lower and len(kw) > best_len:
                    best_task = task.name
                    best_len = len(kw)
        return best_task
