"""
Motor Image-to-Form Agent -- 6-stage pipeline orchestrator.

Stages:
    1. Motor Category Detection   (VLM)
    2. Visual Neutral Facts        (VLM)
    3. CSV Schema Loader           (Python)
    4. Field Eligibility Resolver  (Python)
    5. Field Value Generator       (VLM + rules)
    6. Final Form Output Generator (Python)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.motor_schema_loader import (
    MotorFormSchema,
    MotorField,
    MotorFieldConstraint,
    get_motor_schema_loader,
)
from app.services.openrouter import OpenRouterService

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage output types
# ---------------------------------------------------------------------------

@dataclass
class MotorCategoryResult:
    """Output of Stage 1."""
    category: str
    confidence: float
    reasoning: str


@dataclass
class VisualFacts:
    """Output of Stage 2."""
    facts: List[str]
    raw_description: str
    image_quality_scores: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EligibleField:
    """A single field that the agent should attempt to fill."""
    field_name: str
    required: bool
    depends_on: Optional[str]
    allowed_values: Optional[List[str]]
    source: str  # "fixed" | "free_text"


@dataclass
class FieldValueResult:
    """Value generated for a single field (output of Stage 5 + post-processing)."""
    value: Any
    confidence: float
    source: str  # "image" | "constraint" | "default"
    needs_user_input: bool
    depends_on: Optional[str] = None


@dataclass
class MotorFormOutput:
    """Final output of Stage 6."""
    status: str  # "success" | "partial"
    category: str
    category_confidence: float
    fields: Dict[str, Dict[str, Any]]
    completed_stages: List[str]
    image_quality_scores: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MotorAgentService:
    """
    Orchestrates the 6-stage Motor Image-to-Form pipeline.

    Usage::

        agent = MotorAgentService()
        result = await agent.run(image_urls=["https://..."], combined_vision=False)
    """

    def __init__(self) -> None:
        self.openrouter = OpenRouterService()
        self.schema_loader = get_motor_schema_loader()
        self.confidence_threshold: float = settings.MOTOR_CONFIDENCE_THRESHOLD

    # ---- Public entry point ------------------------------------------------

    async def run(
        self,
        image_urls: List[str],
        combined_vision: bool = False,
    ) -> MotorFormOutput:
        """
        Execute the full 6-stage pipeline and return a ``MotorFormOutput``.

        If an intermediate stage fails the agent returns a *partial* result
        containing whatever was completed so far.
        """
        completed_stages: List[str] = []
        errors: List[str] = []

        # -- Stage 1 & 2: Vision -------------------------------------------
        category_result: Optional[MotorCategoryResult] = None
        visual_facts: Optional[VisualFacts] = None

        try:
            if combined_vision:
                category_result, visual_facts = await self._stage_1_2_combined(image_urls)
                completed_stages.append("stage_1_2_combined")
            else:
                category_result = await self._stage_1_category_detection(image_urls)
                completed_stages.append("stage_1_category_detection")
                visual_facts = await self._stage_2_visual_facts(image_urls)
                completed_stages.append("stage_2_visual_facts")
        except Exception as exc:
            errors.append(f"Vision stages failed: {exc}")
            logger.error(f"Motor agent vision stages failed: {exc}", exc_info=True)
            return MotorFormOutput(
                status="partial",
                category="unknown",
                category_confidence=0.0,
                fields={},
                completed_stages=completed_stages,
                errors=errors,
            )

        # -- Stage 3: Schema loading ----------------------------------------
        try:
            schema = self._stage_3_load_schema(category_result.category)
            completed_stages.append("stage_3_schema_loaded")
        except Exception as exc:
            errors.append(f"Schema loading failed: {exc}")
            logger.error(f"Motor agent schema loading failed: {exc}", exc_info=True)
            return MotorFormOutput(
                status="partial",
                category=category_result.category,
                category_confidence=category_result.confidence,
                fields={},
                completed_stages=completed_stages,
                errors=errors,
            )

        # -- Stage 4: Field eligibility -------------------------------------
        try:
            eligible_fields = self._stage_4_resolve_eligibility(schema)
            completed_stages.append("stage_4_field_eligibility")
        except Exception as exc:
            errors.append(f"Field eligibility resolution failed: {exc}")
            logger.error(f"Motor agent eligibility failed: {exc}", exc_info=True)
            return MotorFormOutput(
                status="partial",
                category=category_result.category,
                category_confidence=category_result.confidence,
                fields={},
                completed_stages=completed_stages,
                errors=errors,
            )

        # -- Stage 5: Field value generation --------------------------------
        raw_field_values: Dict[str, Any] = {}
        try:
            raw_field_values = await self._stage_5_generate_field_values(
                image_urls=image_urls,
                visual_facts=visual_facts,
                eligible_fields=eligible_fields,
                category=category_result.category,
            )
            completed_stages.append("stage_5_field_values")
        except Exception as exc:
            errors.append(f"Field value generation failed: {exc}")
            logger.error(f"Motor agent field value generation failed: {exc}", exc_info=True)
            # Continue to Stage 6 with empty values -- all fields become needs_user_input

        # -- Stage 6: Final form output -------------------------------------
        form_output = self._stage_6_build_output(
            category_result=category_result,
            schema=schema,
            eligible_fields=eligible_fields,
            raw_field_values=raw_field_values,
            completed_stages=completed_stages,
            errors=errors,
            image_quality_scores=visual_facts.image_quality_scores if visual_facts else [],
        )

        return form_output

    # ---- Stage implementations --------------------------------------------

    async def _stage_1_category_detection(
        self, image_urls: List[str]
    ) -> MotorCategoryResult:
        data = await self.openrouter.detect_motor_category(image_urls)
        return MotorCategoryResult(
            category=data["category"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
        )

    async def _stage_2_visual_facts(
        self, image_urls: List[str]
    ) -> VisualFacts:
        data = await self.openrouter.extract_motor_visual_facts(image_urls)
        return VisualFacts(
            facts=data["facts"],
            raw_description=data["raw_description"],
            image_quality_scores=data.get("image_quality_scores", []),
        )

    async def _stage_1_2_combined(
        self, image_urls: List[str]
    ) -> tuple:
        """Returns (MotorCategoryResult, VisualFacts)."""
        data = await self.openrouter.detect_motor_category_and_visual_facts(image_urls)
        cat_result = MotorCategoryResult(
            category=data["category"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
        )
        vis_facts = VisualFacts(
            facts=data["facts"],
            raw_description=data["raw_description"],
            image_quality_scores=data.get("image_quality_scores", []),
        )
        return cat_result, vis_facts

    def _stage_3_load_schema(self, category_name: str) -> MotorFormSchema:
        return self.schema_loader.get_schema_for_category(category_name)

    def _stage_4_resolve_eligibility(
        self, schema: MotorFormSchema
    ) -> List[EligibleField]:
        """
        Build the ordered list of eligible fields (parents before children).
        """
        sorted_fields = self.schema_loader.topological_sort_fields(schema.fields)

        eligible: List[EligibleField] = []
        for mf in sorted_fields:
            constraint = schema.constraints.get(mf.field_name)
            eligible.append(EligibleField(
                field_name=mf.field_name,
                required=mf.required,
                depends_on=mf.depends_on,
                allowed_values=constraint.allowed_values if constraint else None,
                source=constraint.source if constraint else "free_text",
            ))

        logger.debug(
            f"Stage 4: {len(eligible)} eligible fields for "
            f"category '{schema.category.category_name}'"
        )
        return eligible

    async def _stage_5_generate_field_values(
        self,
        image_urls: List[str],
        visual_facts: VisualFacts,
        eligible_fields: List[EligibleField],
        category: str = "",
    ) -> Dict[str, Any]:
        """
        Call VLM for field values and apply post-processing rules.
        Description and Title are regular form fields in the response.
        """
        # Prepare eligible_fields as dicts for the prompt builder
        ef_dicts = [
            {
                "field_name": ef.field_name,
                "required": ef.required,
                "depends_on": ef.depends_on,
                "allowed_values": ef.allowed_values,
                "source": ef.source,
            }
            for ef in eligible_fields
        ]

        raw = await self.openrouter.generate_motor_field_values(
            image_urls=image_urls,
            visual_facts=visual_facts.facts,
            raw_description=visual_facts.raw_description,
            eligible_fields=ef_dicts,
            category=category,
        )

        # Post-process: validate constraints, apply threshold, enforce dependencies
        return self._post_process_field_values(raw, eligible_fields)

    def _post_process_field_values(
        self,
        raw: Dict[str, Any],
        eligible_fields: List[EligibleField],
    ) -> Dict[str, Any]:
        """
        Apply constraint validation, confidence thresholding, and dependency
        enforcement to raw VLM output.
        """
        ef_map = {ef.field_name: ef for ef in eligible_fields}
        processed: Dict[str, Dict[str, Any]] = {}

        for ef in eligible_fields:
            field_data = raw.get(ef.field_name, {})
            if not isinstance(field_data, dict):
                # VLM might return a bare value instead of {value, confidence}
                field_data = {"value": field_data, "confidence": 0.3}

            value = field_data.get("value")
            confidence = float(field_data.get("confidence", 0.0))

            # 1. Fixed-vocabulary validation
            if ef.allowed_values and value is not None:
                if str(value) not in ef.allowed_values:
                    logger.warning(
                        f"Field '{ef.field_name}': value '{value}' not in allowed_values, "
                        f"resetting to null"
                    )
                    value = None
                    confidence = 0.0

            # 2. Confidence threshold
            needs_user_input = confidence < self.confidence_threshold or value is None

            processed[ef.field_name] = {
                "value": value,
                "confidence": round(confidence, 2),
                "needs_user_input": needs_user_input,
            }

        # 3. Dependency cascade: only null children when parent value is truly None.
        #    If the parent has a value (even with low confidence), the child should
        #    still keep its own value -- the user can review both.
        for ef in eligible_fields:
            if ef.depends_on and ef.depends_on in processed:
                parent = processed[ef.depends_on]
                if parent["value"] is None:
                    child = processed[ef.field_name]
                    child["value"] = None
                    child["confidence"] = 0.0
                    child["needs_user_input"] = True

        return processed

    def _stage_6_build_output(
        self,
        category_result: MotorCategoryResult,
        schema: MotorFormSchema,
        eligible_fields: List[EligibleField],
        raw_field_values: Dict[str, Any],
        completed_stages: List[str],
        errors: List[str],
        image_quality_scores: List[Dict[str, Any]] = None,
    ) -> MotorFormOutput:
        """
        Assemble the final JSON-serialisable output.
        """
        completed_stages.append("stage_6_output")

        fields_output: Dict[str, Dict[str, Any]] = {}

        for ef in eligible_fields:
            fv = raw_field_values.get(ef.field_name)
            if fv and isinstance(fv, dict):
                entry: Dict[str, Any] = {
                    "value": fv.get("value"),
                    "confidence": fv.get("confidence", 0.0),
                    "source": "image",
                    "needs_user_input": fv.get("needs_user_input", True),
                }
            else:
                # Field was not filled (e.g. Stage 5 failed)
                entry = {
                    "value": None,
                    "confidence": 0.0,
                    "source": "image",
                    "needs_user_input": True,
                }

            if ef.depends_on:
                entry["depends_on"] = ef.depends_on
            if ef.required:
                entry["required"] = True

            fields_output[ef.field_name] = entry

        status = "success" if not errors else "partial"

        return MotorFormOutput(
            status=status,
            category=category_result.category,
            image_quality_scores=image_quality_scores or [],
            category_confidence=category_result.confidence,
            fields=fields_output,
            completed_stages=completed_stages,
            errors=errors,
        )
