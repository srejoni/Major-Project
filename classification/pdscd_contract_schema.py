"""
schema.py
=========
Locked input/output contract for the RAG + LLM stage of the
RSNA Lumbar Spine Degenerative Classification Pipeline.

Build plan reference: Phase 1.1 (write this file) / Phase 1.2 (agree it).

INPUT side
----------
Mirrors, field-for-field, the ensemble team's authoritative contract:
    pdscd_output_schema.json  (schema_version "1.0.0")
    pdscd_output_format.md
Do NOT rename, drop, or "simplify" these field names. Phase 7.2 re-validates
real ensemble exports against this file the moment they arrive -- if this
file drifts from their JSON Schema, that safety net is worthless.

The ensemble export is PER STUDY (one JSON object, `predictions` = exactly
25 entries = 5 conditions x 5 levels). It is NOT per-finding. Downstream
phases that want to loop "per finding" (Phase 3 retrieval, Phase 5 LLM
calls) should iterate `EnsembleStudyOutput.predictions`, where each item
is a `Prediction` -- that's your "per finding" unit.

OUTPUT side
-----------
What THIS stage produces, per finding: `FindingRecommendation`.
25 of these (one per prediction) roll up into `StudyRecommendations`,
which is what Phase 6.3 / Phase 8 validate pipeline output against.

Requires: pydantic>=2.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Shared enums -- must match pdscd_output_schema.json section 4 EXACTLY,
# character for character. These strings are used as dict/filter keys
# throughout Phase 2 and Phase 3 -- a typo here breaks retrieval silently.
# ---------------------------------------------------------------------------

class Condition(str, Enum):
    SPINAL_CANAL_STENOSIS = "spinal_canal_stenosis"
    LEFT_NEURAL_FORAMINAL_NARROWING = "left_neural_foraminal_narrowing"
    RIGHT_NEURAL_FORAMINAL_NARROWING = "right_neural_foraminal_narrowing"
    LEFT_SUBARTICULAR_STENOSIS = "left_subarticular_stenosis"
    RIGHT_SUBARTICULAR_STENOSIS = "right_subarticular_stenosis"


class Level(str, Enum):
    L1_L2 = "L1/L2"
    L2_L3 = "L2/L3"
    L3_L4 = "L3/L4"
    L4_L5 = "L4/L5"
    L5_S1 = "L5/S1"


class SeverityLabel(str, Enum):
    NORMAL_MILD = "Normal/Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"


class PredictionStatus(str, Enum):
    OK = "ok"
    MISSING_DETECTION = "missing_detection"
    MISSING_LABEL = "missing_label"          # training-time only; won't appear at inference
    DUPLICATE_RESOLVED = "duplicate_resolved"


class ModelAgreement(str, Enum):
    UNANIMOUS = "unanimous_3_of_3"
    MAJORITY = "majority_2_of_3"
    SPLIT = "split_no_majority"


class SeriesDescription(str, Enum):
    SAGITTAL_T1 = "Sagittal T1"
    SAGITTAL_T2_STIR = "Sagittal T2/STIR"
    AXIAL_T2 = "Axial T2"


# ---------------------------------------------------------------------------
# INPUT -- sub-objects (per finding)
# ---------------------------------------------------------------------------

class Probabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normal_mild: float = Field(ge=0, le=1)
    moderate: float = Field(ge=0, le=1)
    severe: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> "Probabilities":
        total = self.normal_mild + self.moderate + self.severe
        if not (0.98 <= total <= 1.02):  # float tolerance
            raise ValueError(f"probabilities must sum to ~1.0, got {total}")
        return self


class ModelVote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity_code: Literal[0, 1, 2]
    probabilities: Probabilities


class ModelVotes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    efficientnet_b3: ModelVote
    convnext_tiny: ModelVote
    resnet34: ModelVote


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_description: Optional[SeriesDescription]
    n_slices_used: Optional[Literal[1, 3]]
    yolo_detection_confidence: Optional[float] = Field(default=None, ge=0, le=1)


class Prediction(BaseModel):
    """One finding: one condition x one level. This is the 'per finding'
    unit the original build-plan bullet was describing."""

    model_config = ConfigDict(extra="forbid")

    condition: Condition
    level: Level
    status: PredictionStatus

    severity_code: Optional[Literal[0, 1, 2]]
    severity_label: Optional[SeverityLabel]
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    probabilities: Optional[Probabilities]
    competition_weight: Literal[1, 2, 4]
    model_votes: Optional[ModelVotes]
    model_agreement: Optional[ModelAgreement]
    source: Source

    @model_validator(mode="after")
    def _status_field_consistency(self) -> "Prediction":
        """Mirrors the allOf/if-then block in pdscd_output_schema.json.
        status in {ok, duplicate_resolved} -> prediction fields required.
        status == missing_detection        -> prediction fields must be null.
        """
        prediction_fields = [
            self.severity_code, self.severity_label, self.confidence,
            self.probabilities, self.model_votes, self.model_agreement,
        ]
        must_be_present = self.status in (PredictionStatus.OK, PredictionStatus.DUPLICATE_RESOLVED)

        if must_be_present and any(f is None for f in prediction_fields):
            raise ValueError(
                f"status='{self.status.value}' requires all prediction fields to be "
                f"non-null (condition={self.condition.value}, level={self.level.value})"
            )
        if self.status == PredictionStatus.MISSING_DETECTION and any(f is not None for f in prediction_fields):
            raise ValueError(
                f"status='missing_detection' requires all prediction fields to be null "
                f"(condition={self.condition.value}, level={self.level.value})"
            )
        return self

    @property
    def needs_review_from_status(self) -> bool:
        """True if THIS finding's status alone should force a review flag in Phase 5.4,
        independent of any confidence/agreement threshold. Catches the 'no prediction
        exists at all' case, which a numeric confidence check can't see (confidence
        is None here, not low) -- Phase 5.4 must OR this in, not rely on thresholds alone.
        """
        return self.status == PredictionStatus.MISSING_DETECTION


# ---------------------------------------------------------------------------
# INPUT -- study-level sub-objects
# ---------------------------------------------------------------------------

class ModelVersions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    efficientnet_b3: str
    convnext_tiny: str
    resnet34: str


class EnsembleWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    efficientnet_b3: float = Field(ge=0, le=1)
    convnext_tiny: float = Field(ge=0, le=1)
    resnet34: float = Field(ge=0, le=1)


class EnsembleMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["soft_probability_averaging", "weighted_average", "majority_vote"]
    weights: EnsembleWeights
    note: Optional[str] = None


class SeverityCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    normal_mild: int = Field(ge=0, alias="Normal/Mild")
    moderate: int = Field(ge=0, alias="Moderate")
    severe: int = Field(ge=0, alias="Severe")


class StudySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    highest_severity_code: Optional[Literal[0, 1, 2]]
    highest_severity_label: Optional[SeverityLabel]
    severity_counts: SeverityCounts
    missing_prediction_count: int = Field(ge=0, le=25)
    low_confidence_count: int = Field(ge=0, le=25)


# ---------------------------------------------------------------------------
# INPUT -- top level. This is what Phase 1.4's validation script, and later
# Phase 7.2, validate every real/mock export against.
# ---------------------------------------------------------------------------

class EnsembleStudyOutput(BaseModel):
    """Full per-study payload exported by the ensemble stage.
    Field-for-field mirror of pdscd_output_schema.json (schema_version 1.0.0).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    study_id: int
    generated_at: datetime
    pipeline_version: str
    model_versions: ModelVersions
    ensemble_method: EnsembleMethod
    study_summary: StudySummary
    predictions: list[Prediction] = Field(min_length=25, max_length=25)

    @model_validator(mode="after")
    def _exactly_25_unique_condition_level_pairs(self) -> "EnsembleStudyOutput":
        pairs = [(p.condition, p.level) for p in self.predictions]
        if len(set(pairs)) != 25:
            raise ValueError("predictions must contain exactly one entry per condition x level (25 unique pairs)")
        expected = {(c, l) for c in Condition for l in Level}
        if set(pairs) != expected:
            raise ValueError(f"predictions is missing entries for: {expected - set(pairs)}")
        return self


# ---------------------------------------------------------------------------
# OUTPUT -- what THIS stage (RAG + LLM) produces, per finding.
# Populated in Phase 5, validated in Phase 6.3 / Phase 8.
# ---------------------------------------------------------------------------

class FindingRecommendation(BaseModel):
    """One output record per finding (per condition x level)."""

    model_config = ConfigDict(extra="forbid")

    study_id: int
    condition: Condition
    level: Level

    flagged_for_review: bool
    review_reason: Optional[str] = None
    # e.g. "low_confidence", "split_no_majority", "missing_detection".
    # Free text for now (Phase 1.1) -- tighten to an Enum once Phase 5.4's
    # actual threshold logic is written and the real reason set is known.
    # Required whenever flagged_for_review is True.

    summary_text: Optional[str] = None
    # 2-3 sentence grounded clinical summary. None when flagged_for_review is True.

    grounding_sources: list[str] = Field(default_factory=list)
    # excerpt_id values from excerpts.json (Phase 2.4). Empty when flagged_for_review.

    @model_validator(mode="after")
    def _review_flag_consistency(self) -> "FindingRecommendation":
        if self.flagged_for_review:
            if self.summary_text is not None:
                raise ValueError("summary_text must be None when flagged_for_review is True")
            if self.grounding_sources:
                raise ValueError("grounding_sources must be empty when flagged_for_review is True")
            if self.review_reason is None:
                raise ValueError("review_reason is required when flagged_for_review is True")
        else:
            if self.summary_text is None:
                raise ValueError("summary_text is required when flagged_for_review is False")
            if not self.grounding_sources:
                raise ValueError("grounding_sources must be non-empty when flagged_for_review is False")
        return self


class StudyRecommendations(BaseModel):
    """25 FindingRecommendations for one study. What Phase 6.1's end-to-end
    run should produce, and what Phase 6.3 validates."""

    model_config = ConfigDict(extra="forbid")

    study_id: int
    recommendations: list[FindingRecommendation] = Field(min_length=25, max_length=25)

    @model_validator(mode="after")
    def _covers_all_findings_for_this_study(self) -> "StudyRecommendations":
        pairs = set()
        for r in self.recommendations:
            if r.study_id != self.study_id:
                raise ValueError(f"recommendation study_id {r.study_id} does not match study {self.study_id}")
            pairs.add((r.condition, r.level))
        expected = {(c, l) for c in Condition for l in Level}
        if pairs != expected:
            raise ValueError("recommendations must cover all 25 condition x level combinations")
        return self


# ---------------------------------------------------------------------------
# ADDED for the LLM provider-layer replacement task -- structured-output
# validation. These do NOT replace or modify FindingRecommendation /
# StudyRecommendations above, and no existing field was touched. They
# validate the RAW JSON a provider returns (per the requirement that every
# response from every provider be schema-validated before use), BEFORE
# that text is dropped into FindingRecommendation.summary_text by
# generate_response.py. Flagging this as a pure addition, per instructions
# not to silently modify the schema.
# ---------------------------------------------------------------------------

class LLMFindingSummary(BaseModel):
    """Expected shape of a single-finding structured LLM response
    (used by call_llm.py)."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


class LLMBatchResultItem(BaseModel):
    """One entry inside a batched per-study structured LLM response."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    level: str
    summary: str = Field(min_length=1)


class LLMBatchOutput(BaseModel):
    """Expected shape of a per-study batched structured LLM response
    (used by batch_llm.py). Note: this validates SHAPE only (every item
    has condition/level/summary, no extra keys). Checking that the set of
    (condition, level) pairs matches THIS study's expected findings --
    no duplicates, none missing -- is still done in batch_llm.py, because
    this schema has no way to know what set was expected for a given call."""

    model_config = ConfigDict(extra="forbid")

    results: list[LLMBatchResultItem]
