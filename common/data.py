"""CSV/XLSX loading, transcript parsing, and reproducible sample-level splits."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

POSITIVE_LABEL = "信访风险件"
NEGATIVE_LABEL = "正常"
LABEL_MAP = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
ID2LABEL = {0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}

_TURN_LINE_RE = re.compile(
    r"^(?:\[(?P<time>[^\]]+)\]\s*)?"
    r"(?P<speaker>[^：:]{1,16})[：:]\s*(?P<text>.*)$"
)


@dataclass
class Turn:
    """One speaker turn from a transcript."""

    text: str
    speaker: str
    timestamp: str = ""
    line: str = ""

    @property
    def full_text(self) -> str:
        return self.line or self.text


@dataclass
class Record:
    """One complete sample (one CSV row or all segments of one XLSX c)."""

    id: str
    raw_text: str
    label: int
    turns: list[Turn]

    @property
    def lines(self) -> list[str]:
        return [turn.full_text for turn in self.turns]


def _clean_line(line: str) -> str:
    return line.replace("\ufeff", "").strip()


def parse_turns(raw_text: str) -> list[Turn]:
    """Parse ``[time] speaker: text`` lines into speaker turns."""

    turns: list[Turn] = []
    for raw_line in raw_text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        match = _TURN_LINE_RE.match(line)
        if match:
            turns.append(
                Turn(
                    text=match.group("text").strip(),
                    speaker=match.group("speaker").strip(),
                    timestamp=(match.group("time") or "").strip(),
                    line=line,
                )
            )
            continue

        # Fallback: treat a bare line as an unlabeled utterance.
        turns.append(Turn(text=line, speaker="", line=line))

    if not turns:
        turns.append(Turn(text=_clean_line(raw_text), speaker="", line=_clean_line(raw_text)))
    return turns


def load_records(csv_path: str | Path) -> list[Record]:
    """Load complete samples; turn_label is never a feature or training target."""

    csv_path = Path(csv_path)
    if csv_path.suffix.lower() == '.xlsx':
        from .xlsx_data import load_segmented_rows
        records = []
        for ident, raw_text, label in load_segmented_rows(csv_path):
            if label not in LABEL_MAP:
                raise ValueError(f'Unexpected label {label!r} for {ident} in {csv_path}')
            records.append(Record(ident, raw_text, LABEL_MAP[label], parse_turns(raw_text)))
        return records
    if csv_path.suffix.lower() != '.csv':
        raise ValueError('Data source must be .csv or .xlsx')
    records: list[Record] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("label") or "").strip()
            if label not in LABEL_MAP:
                raise ValueError(
                    f"Unexpected label {label!r} in {csv_path}; "
                    f"expected one of {sorted(LABEL_MAP)}"
                )
            raw_text = row.get("merged_text") or ""
            records.append(
                Record(
                    id=row.get("id") or "",
                    raw_text=raw_text,
                    label=LABEL_MAP[label],
                    turns=parse_turns(raw_text),
                )
            )
    return records


def sample_unit(path):
    return 'one_c_all_s_segments' if Path(path).suffix.lower() == '.xlsx' else 'one_csv_row_merged_text'


def make_splits(
    records: Iterable[Record],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Record]]:
    """Create stratified train/validation/test splits."""

    from sklearn.model_selection import train_test_split

    records = list(records)
    if not records:
        raise ValueError("No records found")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    labels = [record.label for record in records]
    train, rest = train_test_split(
        records,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=labels,
    )

    rest_labels = [record.label for record in rest]
    relative_val = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        rest,
        test_size=1 - relative_val,
        random_state=seed,
        stratify=rest_labels,
    )
    return {"train": train, "val": val, "test": test}


def split_ids(
    records: Iterable[Record],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Record]]:
    """Alias kept for callers that prefer ``split_ids`` naming."""

    return make_splits(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
