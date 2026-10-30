"""Group-aware, deterministic train/val/test splitting for OpenDarkSTS datasets.

Why this module exists
----------------------
The Roboflow-derived Madhulika dataset and the older "balanced" datasets built on
top of it leak: pre-augmented copies of a single source image are spread across
train/valid/test, and adjacent tiles cut from the same satellite scene on the same
day end up on both sides of the split. Any metric measured that way is optimistic
and effectively meaningless.

The fix is to never split *images*; split *groups* of images that share an
underlying observation, and put a whole group in exactly one split.

Group rule (authoritative)
--------------------------
The group key is derived from the most original filename available for an image:

    group key = "<region>|<date>"            (default, coarse)
    group key = "<region>|<date>|<scene>"    (when ``use_scene=True``)

where

* ``region`` is a named geographic region (``texas_gulf``, ``gibraltar``,
  ``suez``, ``singapore``) resolved from the source raster bounds when they are
  known, or a coarse ``lon<i>_lat<j>`` one-degree cell otherwise, or
  ``unknown``. Region -- not the *collection* the file arrived in -- is used on
  purpose: the Madhulika Roboflow scenes, the ``sentinel2`` review tiles and the
  ``ship_tiles_new_apr29_2025`` review tiles are all imagery of the same Gulf of
  Mexico footprint and several share acquisition dates, so keying on collection
  would silently allow the same scene into two splits.
* ``date`` is the first ``YYYY-MM-DD`` found in the name. One Sentinel-2 pass
  over one region on one day is one observation; every tile, crop, zoom level,
  render profile and Roboflow augmentation of it belongs to the same group.
* ``scene`` is the ``sceneN`` token when present, else ``scene0``. It is parsed
  and recorded always, but by default it is **not** part of the key: different
  scene numbers on the same date over the same region can overlap, and the
  Madhulika full-scene renders carry no scene token at all, so separating on it
  could re-open a leak. ``use_scene=True`` is available for datasets where scene
  numbering is known to be disjoint.

Before the date is extracted the name is normalised by stripping, in order:
the directory and extension, a Roboflow ``_<ext>.rf.<hash>`` suffix, trailing
``-N-`` / ``-Copy`` / ``(N)`` duplicate markers, a ``True_color`` / ``False_color``
render-profile token, ``zoomed*``/``cropped``/``highres``/``midres`` qualifiers, a
``_tile_<row>_<col>`` tile suffix, and a ``<collection>__`` prefix. Those are all
markers of *renders of the same observation*, never of a different observation.

If no date can be parsed, the normalised base name becomes the group key on its
own, so an unparsable file can never be merged with an unrelated one.

Assignment is deterministic given a seed: groups are shuffled with
``random.Random(seed)``, stable-sorted by descending weight, then greedily given
to whichever split is furthest below its target share (measured over image count
and over every tracked class), with optional hard minimums for a split.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "REGION_BOXES",
    "DEFAULT_RATIOS",
    "ParsedSourceName",
    "GroupStats",
    "region_for_lonlat",
    "region_for_bounds",
    "parse_source_name",
    "group_key",
    "group_key_for_name",
    "assign_groups_to_splits",
    "check_no_group_overlap",
]


# (min_lon, max_lon, min_lat, max_lat)
REGION_BOXES: dict[str, tuple[float, float, float, float]] = {
    "texas_gulf": (-96.5, -93.0, 27.0, 30.0),
    "gibraltar": (-7.5, -3.5, 35.0, 37.5),
    "suez": (31.5, 34.5, 27.0, 31.5),
    "singapore": (102.5, 105.5, 0.0, 2.5),
}

DEFAULT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SCENE_RE = re.compile(r"(?:^|[_\-])scene[_\-]?(\d+)(?=$|[_\-])", re.IGNORECASE)
_TILE_RE = re.compile(r"[_\-]tile[_\-](\d+)[_\-](\d+)(?=$|[_\-])", re.IGNORECASE)
_ROBOFLOW_RE = re.compile(r"_(?:jpg|jpeg|png|tif|tiff)\.rf\.[0-9a-f]+$", re.IGNORECASE)
_COPY_RE = re.compile(r"[\s_\-]*(?:copy|duplicate)(?:[\s_\-]*\d+)?$", re.IGNORECASE)
_PAREN_COPY_RE = re.compile(r"[\s_\-]*\(\d+\)$")
_DASH_INDEX_RE = re.compile(r"-\d+-$")
_RENDER_RE = re.compile(r"(true[_\- ]?color|false[_\- ]?color)", re.IGNORECASE)
_QUALIFIER_RE = re.compile(
    r"[\s_\-]*(zoomed[\s_\-]?(?:in|out)?|highres|hires|midres|lowres|cropped|small|large)",
    re.IGNORECASE,
)
_TRIM_RE = re.compile(r"^[\s_\-]+|[\s_\-]+$")


def region_for_lonlat(lon: float, lat: float) -> str:
    """Return the named region containing ``(lon, lat)``, else a 1-degree cell id."""
    for name, (min_lon, max_lon, min_lat, max_lat) in REGION_BOXES.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return name
    return f"lon{int(math.floor(lon))}_lat{int(math.floor(lat))}"


def region_for_bounds(bounds: Mapping[str, float] | Sequence[float] | None) -> str:
    """Return the region for a raster bounds mapping/sequence, or ``"unknown"``.

    Accepts either ``{"left","right","top","bottom"}`` or ``(left, bottom, right, top)``.
    """
    if bounds is None:
        return "unknown"
    if isinstance(bounds, Mapping):
        try:
            left = float(bounds["left"])
            right = float(bounds["right"])
            top = float(bounds["top"])
            bottom = float(bounds["bottom"])
        except (KeyError, TypeError, ValueError):
            return "unknown"
    else:
        values = list(bounds)
        if len(values) != 4:
            return "unknown"
        try:
            left, bottom, right, top = (float(value) for value in values)
        except (TypeError, ValueError):
            return "unknown"
    return region_for_lonlat((left + right) / 2.0, (top + bottom) / 2.0)


@dataclass(frozen=True)
class ParsedSourceName:
    """Normalised pieces of an original image filename."""

    original: str
    base: str
    date: str | None
    scene: str
    collection: str | None = None
    tile_row: int | None = None
    tile_col: int | None = None
    render_profile: str | None = None


def _strip(value: str) -> str:
    return _TRIM_RE.sub("", value)


def parse_source_name(name: str | Path) -> ParsedSourceName:
    """Parse the most-original filename of an image into grouping components."""
    original = str(name)
    stem = Path(original.replace("\\", "/")).name
    stem = re.sub(r"\.(jpe?g|png|tiff?|webp|bmp)$", "", stem, flags=re.IGNORECASE)

    collection: str | None = None
    if "__" in stem:
        head, _, tail = stem.partition("__")
        if head and tail:
            collection = head
            stem = tail

    stem = _ROBOFLOW_RE.sub("", stem)
    # Duplicate markers may stack, e.g. "...True_color-1-" or "...-Copy (2)".
    for _ in range(4):
        before = stem
        stem = _PAREN_COPY_RE.sub("", stem)
        stem = _COPY_RE.sub("", stem)
        stem = _DASH_INDEX_RE.sub("", stem)
        stem = _strip(stem)
        if stem == before:
            break

    render_match = _RENDER_RE.search(stem)
    render_profile = None
    if render_match:
        render_profile = render_match.group(1).lower().replace(" ", "_").replace("-", "_")
        stem = _RENDER_RE.sub("", stem)

    scene_match = _SCENE_RE.search(stem)
    scene = f"scene{int(scene_match.group(1))}" if scene_match else "scene0"
    if scene_match:
        stem = _SCENE_RE.sub("", stem, count=1)

    tile_match = _TILE_RE.search(stem)
    tile_row = tile_col = None
    if tile_match:
        tile_row = int(tile_match.group(1))
        tile_col = int(tile_match.group(2))
        stem = _TILE_RE.sub("", stem, count=1)

    stem = _QUALIFIER_RE.sub("", stem)
    stem = re.sub(r"[\s_\-]{2,}", "-", _strip(stem))

    date_match = _DATE_RE.search(original)
    date = "-".join(date_match.groups()) if date_match else None

    return ParsedSourceName(
        original=original,
        base=stem,
        date=date,
        scene=scene,
        collection=collection,
        tile_row=tile_row,
        tile_col=tile_col,
        render_profile=render_profile,
    )


def group_key(parsed: ParsedSourceName, region: str = "unknown", *, use_scene: bool = False) -> str:
    """Build the split group key for a parsed name and a resolved region."""
    region = region or "unknown"
    if not parsed.date:
        return f"{region}|nodate|{parsed.base or parsed.original}"
    if use_scene:
        return f"{region}|{parsed.date}|{parsed.scene}"
    return f"{region}|{parsed.date}"


def group_key_for_name(name: str | Path, region: str = "unknown", *, use_scene: bool = False) -> str:
    """Convenience wrapper: parse ``name`` and return its group key."""
    return group_key(parse_source_name(name), region, use_scene=use_scene)


@dataclass
class GroupStats:
    """Aggregate counts for one group of images that must stay together."""

    key: str
    images: int = 0
    boxes: Counter = field(default_factory=Counter)

    def add(self, class_counts: Mapping[str, int] | Counter | None = None, images: int = 1) -> None:
        self.images += images
        if class_counts:
            self.boxes.update(class_counts)

    @property
    def weight(self) -> int:
        return self.images + sum(self.boxes.values())


def _metric_value(stats: GroupStats, metric: str) -> int:
    if metric == "images":
        return stats.images
    return int(stats.boxes.get(metric, 0))


def assign_groups_to_splits(
    groups: Iterable[GroupStats],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
    metrics: Sequence[str] | None = None,
    min_requirements: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, str]:
    """Assign whole groups to splits deterministically.

    ``groups``            groups to assign (order-independent; sorted internally).
    ``ratios``            target share per split, e.g. ``{"train":.7,"val":.15,"test":.15}``.
    ``seed``              seed for the deterministic shuffle.
    ``metrics``           metrics balanced across splits; defaults to ``["images"]``
                          plus every class seen in ``groups``.
    ``min_requirements``  hard floors, e.g. ``{"test": {"images": 100, "ships": 60}}``.
                          After the greedy pass, groups are moved out of the
                          largest split until the floors are met (or no group
                          that helps is left).

    Returns ``{group_key: split_name}``.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    if not ratios:
        raise ValueError("ratios must not be empty")
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("ratios must sum to a positive number")
    ratios = {name: value / total_ratio for name, value in ratios.items()}

    group_list = sorted(groups, key=lambda item: item.key)
    if not group_list:
        return {}

    seen_classes: list[str] = []
    for stats in group_list:
        for class_name in stats.boxes:
            if class_name not in seen_classes:
                seen_classes.append(class_name)
    metric_names = list(metrics) if metrics is not None else ["images", *sorted(seen_classes)]

    totals = {metric: sum(_metric_value(stats, metric) for stats in group_list) for metric in metric_names}
    targets = {
        split: {metric: ratios[split] * totals[metric] for metric in metric_names} for split in ratios
    }

    shuffled = list(group_list)
    random.Random(seed).shuffle(shuffled)
    order = sorted(shuffled, key=lambda item: -item.weight)

    split_names = sorted(ratios, key=lambda name: (-ratios[name], name))
    current = {split: {metric: 0 for metric in metric_names} for split in split_names}
    assignment: dict[str, str] = {}

    def deficit(split: str) -> float:
        worst = -math.inf
        for metric in metric_names:
            target = targets[split][metric]
            if target <= 0:
                continue
            worst = max(worst, (target - current[split][metric]) / target)
        return worst if worst != -math.inf else 0.0

    for stats in order:
        split = max(split_names, key=lambda name: (deficit(name), ratios[name], name))
        assignment[stats.key] = split
        for metric in metric_names:
            current[split][metric] += _metric_value(stats, metric)

    if min_requirements:
        by_key = {stats.key: stats for stats in group_list}
        for split, requirements in min_requirements.items():
            if split not in current:
                continue
            for metric, minimum in requirements.items():
                if metric not in metric_names:
                    continue
                while current[split][metric] < minimum:
                    donor_split = max(
                        (name for name in split_names if name != split),
                        key=lambda name: (current[name][metric], name),
                    )
                    candidates = [
                        by_key[key]
                        for key, value in assignment.items()
                        if value == donor_split and _metric_value(by_key[key], metric) > 0
                    ]
                    if not candidates:
                        break
                    needed = minimum - current[split][metric]
                    candidates.sort(
                        key=lambda stats: (
                            abs(_metric_value(stats, metric) - needed),
                            stats.weight,
                            stats.key,
                        )
                    )
                    chosen = candidates[0]
                    assignment[chosen.key] = split
                    for name in metric_names:
                        current[donor_split][name] -= _metric_value(chosen, name)
                        current[split][name] += _metric_value(chosen, name)

    return assignment


def check_no_group_overlap(item_groups: Mapping[str, tuple[str, str]]) -> dict[str, object]:
    """Verify that no group appears in more than one split.

    ``item_groups`` maps an item id to ``(split, group_key)``.
    Returns a report dict with ``ok``, ``groups_by_split`` and ``overlapping_groups``.
    """
    groups_by_split: dict[str, set[str]] = {}
    splits_by_group: dict[str, set[str]] = {}
    for _item, (split, group) in item_groups.items():
        groups_by_split.setdefault(split, set()).add(group)
        splits_by_group.setdefault(group, set()).add(split)
    overlapping = {
        group: sorted(splits) for group, splits in splits_by_group.items() if len(splits) > 1
    }
    return {
        "ok": not overlapping,
        "group_count": len(splits_by_group),
        "groups_by_split": {split: sorted(values) for split, values in sorted(groups_by_split.items())},
        "group_counts_by_split": {split: len(values) for split, values in sorted(groups_by_split.items())},
        "overlapping_groups": overlapping,
    }
