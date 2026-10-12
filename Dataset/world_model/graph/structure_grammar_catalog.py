"""Project reviewed record grammar paths into source-field catalog definitions.

The grammar describes complete records.  This projection assigns only source
field paths and their possible raw leaf types; it never creates observations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import AbstractSet, Iterable, Mapping

from jsonschema import Draft202012Validator

from Dataset.world_model.graph.field_contract import EXACT_MISSING_RULES
from Dataset.world_model.graph.field_paths import (
    DYNAMIC_KEYED_OBJECTS, ENTITY_KEYED_OBJECTS, MIXED_KEYED_OBJECTS,
    SNAPSHOT_DICTIONARIES, SNAPSHOT_ENTITY_PATHS,
    source_field_id,
)
from Dataset.world_model.graph.field_catalog import SourceDeclaration
from Dataset.world_model.graph.structure_closure import (
    GRAMMAR_SOURCE_FAMILIES, declaration_not_applicable,
    declared_effective_families,
    reviewed_source_grammar_declaration, source_grammar_for_declaration,
)
from Dataset.world_model.graph.structure_schema import (
    ArrayItem, ClosedRecordGrammar, DynamicMapKey, GrammarField, VariantSelector,
    dynamic_map_key_matches,
    is_publication_excluded_path,
)


_BASIS = "Dataset/world_model/graph/structure_closure.py:source_grammar_for_declaration"
_MISSING_STATUSES = frozenset({
    "unknown", "missing_source_record", "inactive", "not_applicable",
})
_TEMPLATE_COMPONENT = re.compile(r"\{([^{}]*)\}")


@dataclass(frozen=True)
class GrammarSourceRule:
    source_family: str
    field_path: str
    source_branch: str | None
    source_subtype: str | None
    source_field_id: str
    source_value_types: tuple[str, ...]
    value_statuses: tuple[str, ...]
    schema_refs: tuple[str, ...]
    # (dimension, admission rules) for every open key this path templates, in
    # path order.  A key is a member of the map only when one of its rules --
    # the DynamicMapKey patterns the grammar walk used to template the path --
    # admits it, so a historical literal key is derivable from the record
    # grammar itself and never from a wildcard.
    dynamic_keys: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def catalog_rule(self) -> dict[str, object]:
        return {
            "normalized_id": "structural:source:" + self.source_field_id,
            "binding_slot": None,
            "selector": "record_grammar_source_leaf",
            "source_value_types": self.source_value_types,
            "value_types": self.source_value_types,
            "value_statuses": self.value_statuses,
            "structure_statuses": ("source_only",),
            "declared_shape": None,
            "declaration_basis": _BASIS,
            "record_family": self.source_family,
            "record_schema": self.schema_refs[0] if len(self.schema_refs) == 1 else None,
        }

    def bind_dynamic_key(self, field_path: str) -> dict[str, str] | None:
        """Return the key assignment that turns this template into ``field_path``.

        A runtime stores a dynamic map under its concrete keys, so it can spell
        ``window_status.event_family_counts.road_closure`` where the grammar
        declares ``window_status.event_family_counts.{event_family}``.  That
        literal is an instance of this template, and this decides it exactly:
        every non-key component must match verbatim, the two paths must have
        the same number of components, every open key of the template must be
        bound once, and each bound key must be admitted by its own
        ``DynamicMapKey`` rule.  A path that fails any of those returns
        ``None`` rather than a partial match, so an unrelated literal key can
        never reach a declaration through this route.
        """
        template = self.field_path.split(".")
        parts = field_path.split(".")
        if len(template) != len(parts) or not self.dynamic_keys:
            return None
        rules_by_dimension = dict(self.dynamic_keys)
        if len(rules_by_dimension) != len(self.dynamic_keys):
            return None
        assignment: dict[str, str] = {}
        for expected, actual in zip(template, parts):
            match = _TEMPLATE_COMPONENT.fullmatch(expected)
            if match is None:
                if expected != actual:
                    return None
                continue
            dimension = match.group(1)
            rules = rules_by_dimension.get(dimension)
            if not rules or dimension in assignment:
                return None
            if not any(dynamic_map_key_matches(rule, actual) for rule in rules):
                return None
            assignment[dimension] = actual
        return assignment if set(assignment) == set(rules_by_dimension) else None

    def bound_key_rule(self, source_field_id: str) -> dict[str, object]:
        """Declare one historical literal key as an instance of this template.

        The declaration is this template's own rule, so the admitted types and
        statuses are the record grammar's.  Only the normalized ID follows the
        observed row, which keeps the source and normalized indexes the
        published catalog already assigned to that literal.
        """
        if not self.dynamic_keys:
            raise ValueError(f"grammar source field has no open key: {self.source_field_id}")
        rule = dict(self.catalog_rule)
        rule["normalized_id"] = "structural:source:" + source_field_id
        return rule


@dataclass(frozen=True)
class GrammarCatalogPlan:
    rules: tuple[GrammarSourceRule, ...]
    # Aligned with the declaration sequence passed to compile_catalog_grammars.
    declaration_statuses: tuple[str, ...]
    wholly_closed_families: frozenset[str]


def _path_dimensions(source_family: str, path: str) -> tuple[str | None, bool]:
    """Return this map's open-key dimension and whether it is a mixed map."""
    dynamic_objects = DYNAMIC_KEYED_OBJECTS.get(source_family, {})
    mixed_objects = MIXED_KEYED_OBJECTS.get(source_family, {})
    entity_objects = ENTITY_KEYED_OBJECTS.get(source_family, ())
    mixed_entry = mixed_objects.get(path)
    dimensions = []
    if path in dynamic_objects:
        dimensions.append(dynamic_objects[path])
    if path in entity_objects or (source_family == "formal_event_realization"
                                  and path in SNAPSHOT_ENTITY_PATHS):
        dimensions.append("entity")
    if source_family == "formal_event_realization" and path in SNAPSHOT_DICTIONARIES:
        dimensions.append("tick")
    if mixed_entry is not None:
        dimensions.append(mixed_entry[0])
    if len(dimensions) > 1:
        raise ValueError(f"source path has conflicting dynamic templates: {source_family}:{path}")
    return (dimensions[0] if dimensions else None), mixed_entry is not None


def _field_path(source_family: str, tokens: tuple[object, ...]) -> str:
    path = ""
    for token in tokens:
        if type(token) is int or isinstance(token, ArrayItem):
            path += "[]"
            continue
        dimension, is_mixed = _path_dimensions(source_family, path)
        if isinstance(token, DynamicMapKey):
            if dimension is None:
                raise ValueError(
                    f"grammar dynamic key has no source path template: "
                    f"{source_family}:{path}:{token.rule}")
            component = "{" + dimension + "}"
        elif type(token) is str:
            # Runtime normalizes every key of a dynamic-keyed object to that
            # object's dimension (all-key normalization), so a str key under a
            # dynamic map binds the same dimension its DMK keys would.  Two
            # exceptions keep named business fields literal: arm_manifest
            # source_refs (L5/L6 declare fixed properties keys, a different
            # producer shape than X's path-keyed source_refs), and every mixed
            # map, whose properties keys are business fields while only the
            # pattern keys bind the dimension.
            is_arm_source_refs = (source_family == "arm_manifest"
                                  and path == "source_refs")
            if is_arm_source_refs or is_mixed or dimension is None:
                component = token
            else:
                component = "{" + dimension + "}"
        else:
            raise TypeError(f"unsupported grammar path token: {source_family}:{token!r}")
        path += ("." if path else "") + component
    return path


def _map_key_rules(source_family: str, fields: Iterable[GrammarField]
                   ) -> dict[str, set[str]]:
    """Collect the open-key patterns every dynamic map of one grammar carries.

    A key can also be named as a property of its map, in which case its own
    token carries no pattern and the map's pattern is the only admission rule
    the grammar states for it.  Read from the same walk that emits the path, so
    a pattern only ever authorizes keys of the map it belongs to.
    """
    rules: dict[str, set[str]] = defaultdict(set)
    for field in fields:
        for index, token in enumerate(field.path):
            if isinstance(token, DynamicMapKey):
                rules[_field_path(source_family, field.path[:index])].add(token.rule)
    return rules


def _dynamic_keys(source_family: str, tokens: tuple[object, ...],
                  map_rules: Mapping[str, AbstractSet[str]],
                  ) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the admission rules for every open key one path templates."""
    keys: list[tuple[str, tuple[str, ...]]] = []
    for index, token in enumerate(tokens):
        if type(token) is int or isinstance(token, ArrayItem):
            continue
        parent = _field_path(source_family, tokens[:index])
        dimension, is_mixed = _path_dimensions(source_family, parent)
        if isinstance(token, DynamicMapKey):
            if dimension is None:
                raise ValueError(
                    f"grammar dynamic key has no source path template: "
                    f"{source_family}:{parent}:{token.rule}")
            keys.append((dimension, (token.rule,)))
        elif type(token) is str:
            is_arm_source_refs = (source_family == "arm_manifest"
                                  and parent == "source_refs")
            if is_arm_source_refs or is_mixed or dimension is None:
                continue
            keys.append((dimension, tuple(sorted(map_rules.get(parent, ())))))
        else:
            raise TypeError(f"unsupported grammar path token: {source_family}:{token!r}")
    return tuple(keys)


def _record_dynamic_keys(entry: dict[str, set[str]], source_family: str,
                         tokens: tuple[object, ...], path: str,
                         map_rules: Mapping[str, AbstractSet[str]]) -> None:
    """Merge one path's open-key rules, checking they name its templates."""
    keys = _dynamic_keys(source_family, tokens, map_rules)
    if [dimension for dimension, _ in keys] != _TEMPLATE_COMPONENT.findall(path):
        raise ValueError(
            f"source path has unmatched dynamic templates: {source_family}:{path}")
    for dimension, rules in keys:
        entry.setdefault(dimension, set()).update(rules)


# structure_sources._ENTITY_FILES walks each entities[] item as its own record
# root instead of walking the complete record, so only these families observe
# descendants at a path relative to their item.  Every other family --
# formal_source_scene_setup among them -- keeps the complete root walk.
_ENTITY_ROOT_FAMILIES = frozenset({"arm_branch_roster", "arm_scene_setup", "formal_roster"})
_ENTITY_ROOT_KEY = "entities"


def _runtime_source_path(source_family: str, tokens: tuple[object, ...],
                         path: str) -> str:
    """Return the runtime source field path for one root-relative grammar path.

    An entity-root family projects ``entities[]`` by walking each item with
    ``source_prefix_tokens=("entities", index)`` and no field prefix, so a
    descendant is observed relative to its item: grammar ``entities[].entity_id``
    is observed as ``entity_id``.  Exactly the item prefix is dropped, which
    leaves the root-relative lookups inside ``_field_path`` untouched and so
    keeps dynamic-map authority where it is.  The item itself is a walk root, so
    an empty item is observed at the empty relative path.  The record's own
    ``entities`` field is projected from the record root as the empty-container
    leaf and keeps its root path, as does every path outside ``entities[]``.  A
    family that is not entity-rooted already observes root-relative paths and
    returns them whole.
    """
    if source_family not in _ENTITY_ROOT_FAMILIES:
        return path
    if (len(tokens) < 2 or tokens[0] != _ENTITY_ROOT_KEY
            or not (type(tokens[1]) is int or isinstance(tokens[1], ArrayItem))):
        return path
    prefix = _field_path(source_family, tokens[:2])
    if path == prefix:
        return ""
    if not path.startswith(prefix + "."):
        raise ValueError(
            f"entity-root source path is not under its item: {source_family}:{path}")
    return path[len(prefix) + 1:]


def _raw_types(kind: str, declared: tuple[str, ...] | None) -> tuple[str, ...]:
    if kind == "primitive":
        if not declared:
            raise ValueError("grammar primitive has no source value types")
        return declared
    if kind == "object":
        return ("empty_object",)
    if kind == "array":
        return ("empty_list",)
    raise ValueError(f"unsupported grammar field kind: {kind}")


def _schema_at_pointer(root: dict, pointer: str) -> dict:
    if pointer == "#":
        return root
    if not pointer.startswith("#/"):
        raise ValueError(f"grammar field has an invalid schema pointer: {pointer}")
    current: object = root
    for escaped in pointer[2:].split("/"):
        part = escaped.replace("~1", "/").replace("~0", "~")
        if type(current) is dict:
            current = current[part]
        elif type(current) is list and part.isdecimal():
            current = current[int(part)]
        else:
            raise ValueError(f"grammar field schema pointer has no target: {pointer}")
    if type(current) is not dict:
        raise ValueError(f"grammar field schema pointer is not a schema: {pointer}")
    return current


def _statuses(source_family: str, path: str,
              raw_types: set[str]) -> tuple[str, ...]:
    statuses = set()
    if raw_types - {"null", "empty_list", "empty_object"}:
        statuses.add("present")
    if "null" in raw_types:
        statuses.add("null")
    if "empty_list" in raw_types:
        statuses.add("empty_list")
    if "empty_object" in raw_types:
        statuses.add("empty_object")
    if "string" in raw_types:
        missing = EXACT_MISSING_RULES.get((source_family, path), {})
        statuses.update(_MISSING_STATUSES.intersection(missing))
    return tuple(sorted(statuses))


def _explicit_members(schema: dict, pointer: str) -> tuple[str, ...]:
    """Read an explicit const/enum member list off a branch discriminator rule."""
    if type(schema) is not dict:
        raise ValueError(f"domain-state branch member rule is not an object: {pointer}")
    if "const" in schema:
        members = (schema["const"],)
    elif "enum" in schema:
        members = tuple(schema["enum"])
    else:
        raise ValueError(f"domain-state branch has no explicit member rule: {pointer}")
    if (not members or any(not isinstance(member, str) or not member for member in members)
            or len(set(members)) != len(members)):
        raise ValueError(f"domain-state branch member rule is not explicit: {pointer}")
    return members


def _branch_property(variant: dict, name: str, pointer: str) -> dict:
    properties = variant.get("properties")
    if type(properties) is not dict or type(properties.get(name)) is not dict:
        raise ValueError(f"domain-state branch has no {name} property: {pointer}")
    return properties[name]


def _domain_state_branches(
    schema: dict, schema_ref: str, authority: str,
) -> tuple[tuple[str, str | None, ClosedRecordGrammar], ...]:
    """Split the formal domain-state union into one grammar per real branch.

    Branches come only from the schema's own discriminators: each top-level oneOf
    alternative fixes ``observation_family``, and pad_facility further splits on
    the ``values`` union's ``facility_subtype`` const/enum members.  Every branch
    root keeps its own copies of shared record fields (episode_id, ...), so those
    fields are attributed to each branch just like the runtime walks them.  A
    non-pad branch never carries a facility subtype.
    """
    variants = schema.get("oneOf")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"domain-state grammar has no branch union: {schema_ref}")
    branches: list[tuple[str, str | None, ClosedRecordGrammar]] = []
    for index, variant in enumerate(variants):
        pointer = f"{schema_ref}#/oneOf/{index}"
        if type(variant) is not dict:
            raise ValueError(f"domain-state branch is not an object: {pointer}")
        for branch in _explicit_members(
                _branch_property(variant, "observation_family", pointer),
                f"{pointer}/observation_family"):
            if branch != "pad_facility":
                branches.append((branch, None, ClosedRecordGrammar(
                    schema_ref, authority, variant)))
                continue
            values = _branch_property(variant, "values", pointer)
            subvariants = values.get("oneOf")
            if not isinstance(subvariants, list) or not subvariants:
                raise ValueError(f"pad-facility branch has no subtype union: {pointer}")
            for subindex, subvariant in enumerate(subvariants):
                subpointer = f"{pointer}/values/oneOf/{subindex}"
                if type(subvariant) is not dict:
                    raise ValueError(f"pad-facility subtype is not an object: {subpointer}")
                for subtype in _explicit_members(
                        _branch_property(subvariant, "facility_subtype", subpointer),
                        f"{subpointer}/facility_subtype"):
                    derived = dict(variant)
                    derived["properties"] = dict(variant["properties"])
                    derived["properties"]["values"] = subvariant
                    branches.append((branch, subtype, ClosedRecordGrammar(
                        schema_ref, authority, derived)))
    return tuple(branches)


# Derived per-branch grammars are immutable and shared across declarations that
# resolve to one schema_ref, so each branch compiles once per process.
_DOMAIN_BRANCHES_BY_REF: dict[
    str, tuple[tuple[str, str | None, int, ClosedRecordGrammar], ...]] = {}


def _branch_grammars(source_family: str, effective_family: str,
                     grammar: ClosedRecordGrammar
                     ) -> tuple[tuple[str | None, str | None, int, ClosedRecordGrammar], ...]:
    """Return (branch, subtype, alternative, grammar) rows for one record grammar.

    ``alternative`` only distinguishes union members that share a branch and
    subtype (the two security_command subject_category rows). Those collapse into
    one source field ID exactly as the runtime qualifies them, so their field
    types union under the same key while each alternative is still projected.
    """
    if source_family != "domain_state":
        return ((None, None, 0, grammar),)
    if effective_family != "domain_state":
        raise ValueError(
            f"domain-state grammar has an unregistered effective type: {effective_family!r}")
    expansions = _DOMAIN_BRANCHES_BY_REF.get(grammar.schema_ref)
    if expansions is None:
        expansions = tuple(
            (branch, subtype, ordinal, branched)
            for ordinal, (branch, subtype, branched) in enumerate(
                _domain_state_branches(grammar.schema, grammar.schema_ref,
                                       grammar.authority)))
        _DOMAIN_BRANCHES_BY_REF[grammar.schema_ref] = expansions
    return expansions


def _mixed_fixed_expansions(source_family: str, tokens: tuple[object, ...],
                            literal_children: Mapping[tuple[str, tuple[VariantSelector, ...]],
                                                       AbstractSet[str]],
                            selectors: tuple[VariantSelector, ...],
                            ) -> tuple[tuple[object, ...], ...]:
    """Expand pattern tokens under a mixed map over that map's named keys.

    A mixed map reached only through a pattern token templates the open keys
    and drops the named business keys the runtime keeps literal
    (traffic_profile.seed_semantics.seed00/seed01/seed02, which L1-L5 express
    as ^seed[0-9]{2}$ with required keys and L6/X as fixed properties).  Those
    fixed keys are emitted too so the grammar enumerates every field the
    runtime can produce; only the remaining pattern keys bind the dimension.

    Two guards keep that expansion exact.  A key is a member of the pattern
    only when the pattern's own rule admits it -- ``_visual_state`` names
    ``mode``/``lights_on``/the runtime families as properties and excludes them
    from its pattern, so re-deriving them here would hand fixed fields the
    builder leaf value rule and invent paths the grammar never declared.  And a
    key the walk already emitted as a named property of this same map in this
    same branch keeps its own value rule, so a typed additionalProperties key
    rule must not also count it dynamic.  ``literal_children`` is keyed by
    (map path, variant selectors) so a property named in one oneOf branch never
    masks a pattern key that is legal in another.
    """
    mixed_objects = MIXED_KEYED_OBJECTS.get(source_family, {})
    if not mixed_objects:
        return ()
    pending = [tokens]
    expanded: list[tuple[object, ...]] = []
    while pending:
        current = pending.pop()
        split = False
        for index, token in enumerate(current):
            if not isinstance(token, DynamicMapKey):
                continue
            map_path = _field_path(source_family, current[:index])
            entry = mixed_objects.get(map_path)
            if entry is None:
                continue
            declared = literal_children.get((map_path, selectors), frozenset())
            members = tuple(fixed_key for fixed_key in sorted(entry[1])
                            if fixed_key not in declared
                            and dynamic_map_key_matches(token.rule, fixed_key))
            if not members:
                continue
            split = True
            for fixed_key in members:
                pending.append(current[:index] + (fixed_key,) + current[index + 1:])
            break
        if not split:
            expanded.append(current)
    return tuple(expanded)


def _project_grammar(source_family: str, grammar: ClosedRecordGrammar,
                     types_by_path: dict[tuple[str, str, str | None, str | None], set[str]],
                     refs_by_path: dict[tuple[str, str, str | None, str | None], set[str]],
                     keys_by_path: dict[tuple[str, str, str | None, str | None],
                                        dict[str, set[str]]],
                     source_branch: str | None, source_subtype: str | None) -> None:
    schema = grammar.schema
    validator = Draft202012Validator(schema)
    empty_allowed: dict[tuple[str, str], bool] = {}
    fields = tuple(field for field in grammar.iter_fields() if field.kind != "variant")
    # Named properties of a mixed map keep their own value rules, so a pattern
    # token under that map must not re-derive them as dynamic.  Read from the
    # same walk that emits those literal fields, and keyed by the variant
    # selectors of that walk so a property named in one oneOf branch never
    # masks a pattern key that is legal in another.
    literal_children: dict[tuple[str, tuple[VariantSelector, ...]], set[str]] = defaultdict(set)
    for field in fields:
        for index, token in enumerate(field.path):
            if type(token) is str:
                literal_children[(_field_path(source_family, field.path[:index]),
                                  field.selectors)].add(token)
    keep: list[bool] = []
    for field in fields:
        include = True
        if is_publication_excluded_path(field.path):
            include = False
        elif field.kind in {"object", "array"}:
            key = (field.schema_path, field.kind)
            allowed = empty_allowed.get(key)
            if allowed is None:
                node = _schema_at_pointer(schema, field.schema_path)
                allowed = validator.evolve(schema=node).is_valid(
                    {} if field.kind == "object" else [])
                empty_allowed[key] = allowed
            if not allowed:
                include = False
        keep.append(include)
    # A container that declares children yet has every child integrity-excluded
    # is a pseudo-empty (the L2 frozen files map reaches this), not an original
    # real-empty state, so it must not be recorded as empty_object.  A container
    # that declares no children is a real-empty state and stays empty_object.
    # Key by (path, kind): the L2 files field is an object integrity map in one
    # branch and a string array of real file paths in another, both at ("files",).
    has_child: set[tuple[tuple[object, ...], str]] = set()
    has_included_leaf: set[tuple[tuple[object, ...], str]] = set()

    def _container_kind(token: object) -> str:
        return ("array" if isinstance(token, ArrayItem) or type(token) is int
                else "object")

    for field in fields:
        if field.path:
            has_child.add((field.path[:-1], _container_kind(field.path[-1])))
        # A non-integrity descendant proves the parent is not integrity-only
        # pseudo-empty.  Count every non-integrity field -- a kept leaf, a real
        # empty_object/empty_list item, or a non-empty container -- not just
        # primitive leaves.  Integrity-excluded leaves (the L2 frozen digest map)
        # fail to mark, so only a container whose children are all integrity
        # metadata is dropped; a legal container holding just an
        # empty-object/empty-list child stays.
        if not is_publication_excluded_path(field.path):
            for index, token in enumerate(field.path):
                has_included_leaf.add((field.path[:index], _container_kind(token)))
    map_rules = _map_key_rules(source_family, fields)
    for field, include in zip(fields, keep):
        if not include:
            continue
        slot = (field.path, field.kind)
        if (field.kind in {"object", "array"} and
                slot in has_child and slot not in has_included_leaf):
            continue
        path = _runtime_source_path(source_family, field.path,
                                    _field_path(source_family, field.path))
        key = (source_family, path, source_branch, source_subtype)
        types_by_path[key].update(_raw_types(field.kind, field.source_value_types))
        refs_by_path[key].add(grammar.schema_ref)
        _record_dynamic_keys(keys_by_path[key], source_family, field.path, path,
                             map_rules)
        for extra in _mixed_fixed_expansions(source_family, field.path,
                                             literal_children, field.selectors):
            extra_path = _runtime_source_path(
                source_family, extra, _field_path(source_family, extra))
            extra_key = (source_family, extra_path, source_branch, source_subtype)
            types_by_path[extra_key].update(_raw_types(field.kind, field.source_value_types))
            refs_by_path[extra_key].add(grammar.schema_ref)
            _record_dynamic_keys(keys_by_path[extra_key], source_family, extra,
                                 extra_path, map_rules)


def grammar_declaration_status(declaration: SourceDeclaration) -> str:
    """Classify a declared role without opening a record or compiling a grammar."""
    if not isinstance(declaration, SourceDeclaration):
        raise TypeError(f"expected SourceDeclaration, got {type(declaration).__name__}")
    if declaration_not_applicable(declaration):
        return "not_applicable"
    if reviewed_source_grammar_declaration(declaration):
        return "closed_record_grammar"
    if declaration.source_family in GRAMMAR_SOURCE_FAMILIES:
        return "selected_subtrees_only"
    if declaration.structure_status == "source_schema_pending":
        return "source_schema_pending"
    if declaration.structure_status == "field_scan_defined":
        return "source_projection_only"
    raise ValueError(
        f"unhandled source structure status: {declaration.source_family}:"
        f"{declaration.structure_status}")


def compile_catalog_grammars(
    declarations: Iterable[SourceDeclaration],
) -> GrammarCatalogPlan:
    """Aggregate one definition per family/path across declarations and branches.

    A family can contain reviewed and unreviewed producer profiles.  Only the
    reviewed declarations contribute rules; their complete record grammar is
    still validated separately before any source value is projected.
    """
    types_by_path: dict[tuple[str, str, str | None, str | None], set[str]] = defaultdict(set)
    refs_by_path: dict[tuple[str, str, str | None, str | None], set[str]] = defaultdict(set)
    keys_by_path: dict[tuple[str, str, str | None, str | None],
                       dict[str, set[str]]] = defaultdict(dict)
    statuses = []
    family_statuses: dict[str, set[str]] = defaultdict(set)
    seen_grammars: dict[tuple[str, str, str, str | None, str | None], ClosedRecordGrammar] = {}
    role_grammars: dict[tuple[str, str], tuple[tuple[str, ClosedRecordGrammar], ...]] = {}
    for declaration in declarations:
        status = grammar_declaration_status(declaration)
        if status == "closed_record_grammar":
            role_key = (declaration.source_family, str(declaration.path))
            grammars = role_grammars.get(role_key)
            if grammars is None:
                grammars = tuple(
                    (effective_family, source_grammar_for_declaration(
                        declaration, effective_family=effective_family))
                    for effective_family in declared_effective_families(declaration))
                if not grammars:
                    raise ValueError(f"reviewed source has no effective record family: {role_key}")
                role_grammars[role_key] = grammars
            for effective_family, grammar in grammars:
                for source_branch, source_subtype, ordinal, branched in _branch_grammars(
                        declaration.source_family, effective_family, grammar):
                    grammar_key = (
                        declaration.source_family, effective_family, branched.schema_ref,
                        source_branch, source_subtype, ordinal)
                    previous = seen_grammars.get(grammar_key)
                    if previous is None:
                        _project_grammar(effective_family, branched, types_by_path,
                                         refs_by_path, keys_by_path,
                                         source_branch, source_subtype)
                        seen_grammars[grammar_key] = branched
                    elif (previous is not branched and
                          (previous.authority != branched.authority or
                           previous.schema != branched.schema)):
                        raise ValueError(f"one grammar reference has conflicting definitions: {grammar_key}")
        statuses.append(status)
        family_statuses[declaration.source_family].add(status)

    rules = []
    for key in sorted(types_by_path, key=lambda item: (item[0], item[1], item[2] or "",
                                                       item[3] or "")):
        family, path, source_branch, source_subtype = key
        raw_types = types_by_path[key]
        if not raw_types:
            raise ValueError(f"grammar source field has no raw type: {family}:{path}")
        rules.append(GrammarSourceRule(
            family, path, source_branch, source_subtype,
            source_field_id(family, path, source_branch, source_subtype),
            tuple(sorted(raw_types)),
            _statuses(family, path, raw_types),
            tuple(sorted(refs_by_path[key])),
            tuple((dimension, tuple(sorted(rules)))
                  for dimension, rules in keys_by_path[key].items()),
        ))
    wholly_closed = frozenset(
        family for family, members in family_statuses.items()
        if "closed_record_grammar" in members and
        members <= {"closed_record_grammar", "not_applicable"})
    return GrammarCatalogPlan(tuple(rules), tuple(statuses), wholly_closed)
