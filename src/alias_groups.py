from __future__ import annotations

from dataclasses import dataclass
import re


ALIAS_CONFIG_VERSION = "2"


@dataclass(frozen=True)
class AliasMetadata:
    alias_group: str
    alias_variants: tuple[str, ...]
    alias_source: str


_ALIAS_GROUP_CONFIG: dict[str, dict[str, tuple[str, ...] | str]] = {
    "bedside_table_family": {
        "canonical_labels": ("night stand", "bedside table"),
        "risk_level": "low_risk",
        "variants": (
            "night stand",
            "nightstand",
            "bedside table",
            "bedside cabinet",
            "bedside stand",
        ),
    },
    "trash_can_family": {
        "canonical_labels": ("trash can",),
        "risk_level": "low_risk",
        "variants": (
            "trash can",
            "trash bin",
            "garbage bin",
            "waste bin",
            "wastebasket",
            "compost bin",
            "recycling bin",
        ),
    },
    "wardrobe_family": {
        "canonical_labels": ("wardrobe",),
        "risk_level": "low_risk",
        "variants": (
            "wardrobe",
            "wardrobes",
            "wardrobe closet",
            "wardrobe cabinet",
            "closet wardrobe",
        ),
    },
    "ladder_family": {
        "canonical_labels": ("ladder",),
        "risk_level": "low_risk",
        "variants": (
            "ladder",
            "folded ladder",
            "stepladder",
            "step ladder",
        ),
    },
    "piano_family": {
        "canonical_labels": ("piano",),
        "risk_level": "low_risk",
        "variants": (
            "piano",
            "keyboard piano",
        ),
    },
    "washing_machine_family": {
        "canonical_labels": ("washing machine",),
        "risk_level": "low_risk",
        "variants": (
            "washing machine",
            "washing machines",
            "washer",
        ),
    },
    "clothes_dryer_family": {
        "canonical_labels": ("clothes dryer",),
        "risk_level": "low_risk",
        "variants": (
            "clothes dryer",
            "clothes dryers",
            "laundry dryer",
        ),
    },
    "refrigerator_family": {
        "canonical_labels": ("refrigerator",),
        "risk_level": "low_risk",
        "variants": (
            "refrigerator",
            "mini fridge",
            "fridge",
        ),
    },
    "plant_family": {
        "canonical_labels": ("plant",),
        "risk_level": "low_risk",
        "variants": (
            "plant",
            "potted plant",
            "house plant",
            "houseplant",
        ),
    },
    "sofa_family": {
        "canonical_labels": ("sofa",),
        "risk_level": "low_risk",
        "variants": (
            "sofa",
            "couch",
        ),
    },
    "bag_family": {
        "canonical_labels": ("bag",),
        "risk_level": "review_needed",
        "variants": (
            "bag",
            "backpack",
        ),
    },
    "storage_container_family": {
        "canonical_labels": ("storage container",),
        "risk_level": "review_needed",
        "variants": (
            "storage container",
            "storage box",
            "storage bin",
        ),
    },
    "cabinet_family": {
        "canonical_labels": ("cabinet",),
        "risk_level": "review_needed",
        "variants": (
            "cabinet",
            "cabinets",
            "kitchen cabinet",
            "kitchen cabinets",
            "bathroom cabinet",
            "file cabinet",
        ),
    },
    "counter_family": {
        "canonical_labels": ("counter",),
        "risk_level": "low_risk",
        "variants": (
            "counter",
            "kitchen counter",
        ),
    },
    "chair_family": {
        "canonical_labels": ("chair",),
        "risk_level": "review_needed",
        "variants": (
            "chair",
            "armchair",
            "office chair",
            "dining chair",
            "folding chair",
            "sofa chair",
        ),
    },
    "table_family": {
        "canonical_labels": ("table",),
        "risk_level": "review_needed",
        "variants": (
            "table",
            "coffee table",
            "dining table",
        ),
    },
    "fan_family": {
        "canonical_labels": ("fan",),
        "risk_level": "low_risk",
        "variants": (
            "fan",
            "ceiling fan",
        ),
    },
    "bookshelf_family": {
        "canonical_labels": ("bookshelf",),
        "risk_level": "low_risk",
        "variants": (
            "bookshelf",
            "bookshelves",
            "book shelf",
        ),
    },
    "suitcase_family": {
        "canonical_labels": ("suitcase",),
        "risk_level": "review_needed",
        "variants": (
            "suitcase",
            "suitcases",
            "luggage",
        ),
    },
}

_CANONICAL_TO_GROUP: dict[str, str] = {}
for _group_name, _config in _ALIAS_GROUP_CONFIG.items():
    for _label in _config.get("canonical_labels", ()):
        _CANONICAL_TO_GROUP[str(_label).strip().lower()] = str(_group_name)


def _normalize_variant_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _singleton_alias_group_name(canonical_label: str) -> str:
    text = _normalize_variant_text(canonical_label)
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not slug:
        slug = "unknown"
    return f"{slug}_family"


def get_alias_group_risk_level(alias_group: str) -> str:
    config = _ALIAS_GROUP_CONFIG.get(str(alias_group or "").strip().lower())
    if not isinstance(config, dict):
        return "singleton"
    risk_level = str(config.get("risk_level", "")).strip().lower()
    return risk_level or "singleton"


def get_explicit_alias_group_config() -> dict[str, dict[str, object]]:
    return {
        str(group_name): {
            "canonical_labels": tuple(config.get("canonical_labels", ())),
            "variants": tuple(config.get("variants", ())),
            "risk_level": str(config.get("risk_level", "singleton")),
        }
        for group_name, config in sorted(_ALIAS_GROUP_CONFIG.items())
    }


def resolve_alias_metadata(*, raw_label: str, canonical_label: str) -> AliasMetadata:
    canonical = _normalize_variant_text(canonical_label)
    raw = _normalize_variant_text(raw_label)

    explicit_group = _CANONICAL_TO_GROUP.get(canonical)
    if explicit_group:
        config = _ALIAS_GROUP_CONFIG[explicit_group]
        variants = tuple(
            dict.fromkeys(
                _normalize_variant_text(value)
                for value in (
                    *config.get("variants", ()),
                    canonical,
                    raw,
                )
                if _normalize_variant_text(str(value))
            )
        )
        return AliasMetadata(
            alias_group=explicit_group,
            alias_variants=variants or (canonical,),
            alias_source="explicit",
        )

    variants = tuple(
        dict.fromkeys(
            value
            for value in (canonical, raw)
            if value
        )
    )
    return AliasMetadata(
        alias_group=_singleton_alias_group_name(canonical),
        alias_variants=variants or (canonical or "unknown",),
        alias_source="singleton_fallback",
    )


def validate_alias_coverage(canonical_labels: set[str]) -> tuple[bool, list[str]]:
    missing = sorted(
        {
            _normalize_variant_text(label)
            for label in canonical_labels
            if _normalize_variant_text(label) and _normalize_variant_text(label) not in _CANONICAL_TO_GROUP
        }
    )
    return (len(missing) == 0, missing)
