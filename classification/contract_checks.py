"""
classification/contract_checks.py

T3-5: import-time guard for the ensemble-export contract.

assemble_study_output.py builds exactly 25 predictions (5 conditions x 5 levels)
and the output schema pins condition/level to enums. Nothing used to check that
the code's CONDITIONS / LEVELS agree with each other or with the schema, so a
dict-literal typo, a duplicated key (which silently collapses to 4 conditions),
or a renamed schema enum would surface only as a malformed export downstream.

Design notes
  * Failures are explicit `raise ContractError`, NOT `assert`: plain asserts are
    stripped under `python -O`, and this is a safety gate.
  * The schema is searched by walking it for properties named "condition" /
    "level" (following local $ref and allOf/anyOf/oneOf wrappers) rather than
    assuming where the enum lives. EVERY enum found must match the code, so a
    drifted copy anywhere in the schema is caught and its location is reported.
  * Covers the JSON Schema only. It does not compare against the teammate's
    Pydantic mirror (schema.py); JSON-vs-schema.py drift remains a separate,
    cross-repo item.
"""
import json
from pathlib import Path

EXPECTED_N_CONDITIONS = 5
EXPECTED_N_PREDICTIONS = 25


class ContractError(RuntimeError):
    """The export's structural constants disagree with the output schema."""


def _resolve_ref(root, node):
    hops = 0
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ContractError(f"unsupported non-local $ref in schema: {ref!r}")
        target = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            try:
                target = target[part]
            except (KeyError, TypeError, IndexError):
                raise ContractError(f"unresolvable $ref in schema: {ref!r}")
        node = target
        hops += 1
        if hops > 25:
            raise ContractError(f"$ref cycle while resolving {ref!r}")
    return node


def _enum_of(root, node):
    """Enum list a property subschema pins values to, or None."""
    node = _resolve_ref(root, node)
    if not isinstance(node, dict):
        return None
    if isinstance(node.get("enum"), list):
        return node["enum"]
    for key in ("allOf", "anyOf", "oneOf"):
        for branch in node.get(key, []) or []:
            found = _enum_of(root, branch)
            if found is not None:
                return found
    return None


def schema_enums_for_property(schema, prop):
    """[(location, enum_list), ...] for every property literally named `prop`."""
    out = []

    def walk(node, where):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and prop in props:
                e = _enum_of(schema, props[prop])
                if e is not None:
                    out.append((f"{where}/properties/{prop}", list(e)))
            for k, v in node.items():
                walk(v, f"{where}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}/{i}")

    walk(schema, "#")
    return out


def check_output_contract(conditions, levels, schema_path):
    """Raise ContractError listing every disagreement; return a summary dict if clean."""
    conditions, levels = list(conditions), list(levels)
    problems = []

    if len(conditions) != EXPECTED_N_CONDITIONS:
        problems.append(f"len(CONDITIONS) is {len(conditions)}, expected {EXPECTED_N_CONDITIONS}: {conditions}")
    if len(set(conditions)) != len(conditions):
        problems.append(f"CONDITIONS has duplicates: {conditions}")
    if len(set(levels)) != len(levels):
        problems.append(f"LEVELS has duplicates: {levels}")
    n_pred = len(conditions) * len(levels)
    if n_pred != EXPECTED_N_PREDICTIONS:
        problems.append(f"len(CONDITIONS) * len(LEVELS) is {len(conditions)} * {len(levels)} = {n_pred}, "
                        f"expected {EXPECTED_N_PREDICTIONS}")

    schema_path = Path(schema_path)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"schema file not found: {schema_path}"
                            + ("\n  - also: " + "\n  - also: ".join(problems) if problems else ""))
    except json.JSONDecodeError as e:
        raise ContractError(f"schema file is not valid JSON: {schema_path} ({e})")

    locations = {}
    for prop, code_vals in (("condition", conditions), ("level", levels)):
        enums = schema_enums_for_property(schema, prop)
        if not enums:
            problems.append(f"no enum found for a property named {prop!r} anywhere in {schema_path.name}")
            continue
        locations[prop] = [w for w, _ in enums]
        for where, enum in enums:
            only_code = sorted(set(code_vals) - set(enum))
            only_schema = sorted(set(enum) - set(code_vals))
            if only_code or only_schema:
                problems.append(f"{prop}: code and schema disagree at {where} "
                                f"(only in code: {only_code}; only in schema: {only_schema})")

    if problems:
        raise ContractError(f"export contract check FAILED against {schema_path}:\n  - "
                            + "\n  - ".join(problems))
    return {"schema_path": str(schema_path),
            "n_conditions": len(conditions),
            "n_levels": len(levels),
            "n_predictions": n_pred,
            "schema_enum_locations": locations}
