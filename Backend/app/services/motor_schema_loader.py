"""
Motor Schema Loader -- CSV-driven in-memory schema for Motor listing categories.

Parses motor_categories.csv, motor_category_fields.csv, and motor_field_constraints.csv
once at startup and caches the result as a singleton. All downstream stages query this
loader instead of touching the filesystem.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MotorCategory:
    category_id: str
    category_name: str


@dataclass
class MotorField:
    field_name: str
    required: bool
    depends_on: Optional[str]  # parent field name, or None


@dataclass
class MotorFieldConstraint:
    category_name: str  # category name or "All" for global constraints
    field_name: str
    allowed_values: Optional[List[str]]  # None means free-text
    source: str  # "fixed" | "free_text"


@dataclass
class MotorFormSchema:
    """Complete schema for a single motor category."""
    category: MotorCategory
    fields: List[MotorField]
    constraints: Dict[str, MotorFieldConstraint]
    dependency_graph: Dict[str, str] = field(default_factory=dict)  # field -> depends_on


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class MotorSchemaLoader:
    """
    Loads and caches the three Motor CSV files in memory.
    Thread-safe singleton access via ``get_motor_schema_loader()``.
    """

    def __init__(
        self,
        categories_csv: str,
        fields_csv: str,
        constraints_csv: str,
    ):
        self._categories: Dict[str, MotorCategory] = {}
        self._fields_by_category: Dict[str, List[MotorField]] = {}
        # Keyed by (category_name, field_name). "All" category_name = global.
        self._constraints: Dict[tuple, MotorFieldConstraint] = {}

        self._load_categories(categories_csv)
        self._load_fields(fields_csv)
        self._load_constraints(constraints_csv)
        self._validate()

        logger.info(
            f"MotorSchemaLoader initialised -- "
            f"{len(self._categories)} categories, "
            f"{sum(len(v) for v in self._fields_by_category.values())} field entries, "
            f"{len(self._constraints)} constraint entries"
        )

    # ---- CSV parsing -------------------------------------------------------

    def _load_categories(self, path: str) -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Motor categories CSV not found: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = MotorCategory(
                    category_id=row["category_id"].strip(),
                    category_name=row["category_name"].strip(),
                )
                self._categories[cat.category_name] = cat
        logger.debug(f"Loaded {len(self._categories)} motor categories from {csv_path}")

    def _load_fields(self, path: str) -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Motor category fields CSV not found: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat_name = row["category_name"].strip()
                depends_on_raw = row.get("depends_on", "").strip()
                mf = MotorField(
                    field_name=row["field_name"].strip(),
                    required=row["required"].strip().lower() == "true",
                    depends_on=depends_on_raw if depends_on_raw else None,
                )
                self._fields_by_category.setdefault(cat_name, []).append(mf)
        total = sum(len(v) for v in self._fields_by_category.values())
        logger.debug(f"Loaded {total} motor field entries from {csv_path}")

    def _load_constraints(self, path: str) -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Motor field constraints CSV not found: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat_name = row["category_name"].strip()
                field_name = row["field_name"].strip()
                raw_values = row.get("allowed_values", "").strip()
                allowed = raw_values.split("|") if raw_values else None
                source = row.get("source", "free_text").strip()
                key = (cat_name, field_name)
                self._constraints[key] = MotorFieldConstraint(
                    category_name=cat_name,
                    field_name=field_name,
                    allowed_values=allowed,
                    source=source,
                )
        logger.debug(f"Loaded {len(self._constraints)} motor field constraints from {csv_path}")

    # ---- Validation --------------------------------------------------------

    def _validate(self) -> None:
        """Run integrity checks across the three CSVs."""
        errors: List[str] = []

        # Every category referenced in fields must exist
        for cat_name in self._fields_by_category:
            if cat_name not in self._categories:
                errors.append(
                    f"Category '{cat_name}' in fields CSV not found in categories CSV"
                )

        # Every depends_on must reference a field present in the same category
        for cat_name, fields in self._fields_by_category.items():
            field_names = {f.field_name for f in fields}
            for f in fields:
                if f.depends_on and f.depends_on not in field_names:
                    errors.append(
                        f"Field '{f.field_name}' in category '{cat_name}' depends on "
                        f"'{f.depends_on}' which does not exist in that category"
                    )

        # Detect circular dependencies per category
        for cat_name, fields in self._fields_by_category.items():
            dep_map = {f.field_name: f.depends_on for f in fields if f.depends_on}
            for start in dep_map:
                visited = set()
                current = start
                while current in dep_map:
                    if current in visited:
                        errors.append(
                            f"Circular dependency detected in category '{cat_name}' "
                            f"involving field '{current}'"
                        )
                        break
                    visited.add(current)
                    current = dep_map[current]

        if errors:
            for e in errors:
                logger.error(f"Motor CSV validation error: {e}")
            raise ValueError(
                f"Motor CSV validation failed with {len(errors)} error(s): "
                + "; ".join(errors)
            )

        logger.info("Motor CSV validation passed")

    # ---- Public API --------------------------------------------------------

    @property
    def category_names(self) -> List[str]:
        """Return the list of all motor category names."""
        return list(self._categories.keys())

    def get_category(self, name: str) -> Optional[MotorCategory]:
        return self._categories.get(name)

    def get_schema_for_category(self, category_name: str) -> MotorFormSchema:
        """
        Build the full form schema for *category_name*.
        Raises ``KeyError`` if the category does not exist.
        """
        if category_name not in self._categories:
            raise KeyError(
                f"Unknown motor category '{category_name}'. "
                f"Valid: {self.category_names}"
            )

        cat = self._categories[category_name]
        fields = self._fields_by_category.get(category_name, [])

        constraints: Dict[str, MotorFieldConstraint] = {}
        dependency_graph: Dict[str, str] = {}

        for f in fields:
            # Category-specific constraint takes priority over global "All"
            constraint = self._constraints.get(
                (category_name, f.field_name)
            ) or self._constraints.get(
                ("All", f.field_name)
            )
            if constraint:
                constraints[f.field_name] = constraint
            if f.depends_on:
                dependency_graph[f.field_name] = f.depends_on

        return MotorFormSchema(
            category=cat,
            fields=fields,
            constraints=constraints,
            dependency_graph=dependency_graph,
        )

    def get_constraint(self, field_name: str, category_name: Optional[str] = None) -> Optional[MotorFieldConstraint]:
        """Look up constraint: category-specific first, then global 'All'."""
        if category_name:
            result = self._constraints.get((category_name, field_name))
            if result:
                return result
        return self._constraints.get(("All", field_name))

    def topological_sort_fields(self, fields: List[MotorField]) -> List[MotorField]:
        """
        Return *fields* in dependency order (parents before children).
        Fields with no dependency come first.
        """
        dep_map = {f.field_name: f.depends_on for f in fields}
        name_to_field = {f.field_name: f for f in fields}
        visited: set = set()
        order: List[str] = []

        def _visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            parent = dep_map.get(name)
            if parent and parent in name_to_field:
                _visit(parent)
            order.append(name)

        for f in fields:
            _visit(f.field_name)

        return [name_to_field[n] for n in order if n in name_to_field]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_motor_schema_loader: Optional[MotorSchemaLoader] = None


def get_motor_schema_loader(
    categories_csv: Optional[str] = None,
    fields_csv: Optional[str] = None,
    constraints_csv: Optional[str] = None,
) -> MotorSchemaLoader:
    """Get or create the singleton ``MotorSchemaLoader``."""
    global _motor_schema_loader
    if _motor_schema_loader is None:
        from app.core.config import settings
        _motor_schema_loader = MotorSchemaLoader(
            categories_csv=categories_csv or settings.MOTOR_CATEGORIES_CSV_PATH,
            fields_csv=fields_csv or settings.MOTOR_CATEGORY_FIELDS_CSV_PATH,
            constraints_csv=constraints_csv or settings.MOTOR_FIELD_CONSTRAINTS_CSV_PATH,
        )
    return _motor_schema_loader
