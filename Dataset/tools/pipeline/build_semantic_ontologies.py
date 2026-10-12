#!/usr/bin/env python3
"""Build the dataset-independent AeroWorld domain and event TBoxes.

The canonical sources are:

* Dataset/knowledge_graph/domain/catalog.yaml
* Dataset/knowledge_graph/events/catalog.yaml

This builder deliberately emits no predicate-evaluation policy, dataset binding,
scenario expectation, threshold, field path, or executable event rule.  Those
belong to separate specifications.

Ontology module boundaries are stable domain boundaries. Dataset level names
such as L1-L6 belong only to external profile/selection manifests and are not
used as ontology module identifiers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from rdflib import Graph
from rdflib.namespace import OWL, RDF


REPO_ROOT = Path(__file__).resolve().parents[2]
KG_ROOT = REPO_ROOT / "Dataset" / "knowledge_graph"
DOMAIN_DIR = KG_ROOT / "domain"
EVENT_DIR = KG_ROOT / "events"
DOMAIN_CATALOG = DOMAIN_DIR / "catalog.yaml"
EVENT_CATALOG = EVENT_DIR / "catalog.yaml"
MANIFEST_PATH = DOMAIN_DIR / "release-manifest.yaml"
WORLD_ONTOLOGY_IRI = "https://w3id.org/aeroworld/ontology/world"
WORLD_NAMESPACE_IRI = f"{WORLD_ONTOLOGY_IRI}#"
CORE_ONTOLOGY_PATH = KG_ROOT / "core.ttl"
CORE_NAMESPACE_IRI = "https://w3id.org/aeroworld/ontology/core#"
WORLD_ONTOLOGY_PATH = KG_ROOT / "world_core.ttl"
VERSION = "0.3.0"
VERSION_IRI_SEGMENT = VERSION
CREATED_DATE = "2026-07-30"
STABLE_MODULE_IDS = (
    "airspace_operations",
    "ground_mobility",
    "mobility_regulation",
    "agent_interaction_safety",
    "facility_energy_service",
    "communication_process",
    "computation_process",
    "positioning_navigation",
    "environment_hazard",
    "digital_security",
    "operational_constraint_emergency",
    "utm",
)
PREFIXES = """\
@prefix aw: <https://w3id.org/aeroworld/ontology/core#> .
@prefix world: <https://w3id.org/aeroworld/ontology/world#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix vann: <http://purl.org/vocab/vann/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip() + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalize_text(text), encoding="utf-8", newline="\n")


def _ttl_literal(value: str, language: str | None = None) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    suffix = f"@{language}" if language else ""
    return f'"{escaped}"{suffix}'


def _term(value: str, local_prefix: str, local_class_ids: set[str]) -> str:
    if value.startswith("aw:") or value.startswith("world:"):
        return value
    if value in local_class_ids:
        return f"{local_prefix}:{value}"
    return f"world:{value}"


def _statement(subject: str, predicate_objects: list[tuple[str, str]]) -> str:
    lines = [subject]
    for index, (predicate, objects) in enumerate(predicate_objects):
        terminator = " ." if index == len(predicate_objects) - 1 else " ;"
        lines.append(f"    {predicate} {objects}{terminator}")
    return "\n".join(lines)


def _language_objects(label_en: str, label_zh: str) -> str:
    return f"{_ttl_literal(label_en, 'en')} ,\n        {_ttl_literal(label_zh, 'zh')}"


def _domain_iri(slug: str) -> str:
    return f"https://w3id.org/aeroworld/ontology/domain/{slug}"


def _event_iri(slug: str) -> str:
    return f"https://w3id.org/aeroworld/ontology/events/{slug}"


def _catalog_modules(catalog: dict[str, Any], catalog_name: str) -> dict[str, Any]:
    modules = catalog.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError(
            f"{catalog_name} catalog must define a non-empty modules mapping"
        )
    return modules


def _declared_shared_class_terms() -> set[str]:
    """Return the class terms that a generated module may reference."""
    terms: set[str] = set()
    for path, prefix, namespace in (
        (CORE_ONTOLOGY_PATH, "aw", CORE_NAMESPACE_IRI),
        (WORLD_ONTOLOGY_PATH, "world", WORLD_NAMESPACE_IRI),
    ):
        graph = Graph().parse(path, format="turtle")
        for subject in graph.subjects(RDF.type, OWL.Class):
            iri = str(subject)
            if not iri.startswith(namespace):
                continue
            local_name = iri[len(namespace) :]
            terms.add(local_name)
            terms.add(f"{prefix}:{local_name}")
    return terms


def _validate_source_catalogs(
    domain_catalog: dict[str, Any], event_catalog: dict[str, Any]
) -> None:
    if domain_catalog.get("schema") != "aeroworld.domain-catalog.v1":
        raise ValueError("Unsupported domain catalog schema")
    if event_catalog.get("schema") != "aeroworld.event-catalog.v1":
        raise ValueError("Unsupported event catalog schema")
    domain_modules = _catalog_modules(domain_catalog, "Domain")
    event_modules = _catalog_modules(event_catalog, "Event")
    if tuple(domain_modules) != STABLE_MODULE_IDS:
        raise ValueError(
            "Domain catalog module keys must exactly match the ordered stable "
            f"12-module contract: {STABLE_MODULE_IDS}"
        )
    if tuple(event_modules) != STABLE_MODULE_IDS:
        raise ValueError(
            "Event catalog module keys must exactly match the ordered stable "
            f"12-module contract: {STABLE_MODULE_IDS}"
        )
    if tuple(domain_modules) != tuple(event_modules):
        raise ValueError("Domain and event catalogs must declare the same module keys")

    global_predicate_identifiers: set[str] = set()
    global_event_identifiers: set[str] = set()
    shared_class_terms = _declared_shared_class_terms()
    for module_id, domain in domain_modules.items():
        events = event_modules[module_id]
        if domain["slug"] != events["domain_slug"]:
            raise ValueError(f"{module_id}: event catalog domain_slug mismatch")
        local_terms = {
            item["id"]
            for section in (
                "concepts",
                "object_relations",
                "state_properties",
                "predicates",
            )
            for item in domain.get(section, [])
        }
        if len(local_terms) != sum(
            len(domain.get(section, []))
            for section in (
                "concepts",
                "object_relations",
                "state_properties",
                "predicates",
            )
        ):
            raise ValueError(f"{module_id}: duplicate local domain identifier")
        local_class_terms = {item["id"] for item in domain.get("concepts", [])}

        def require_declared_class(value: str, context: str) -> None:
            if value not in local_class_terms and value not in shared_class_terms:
                raise ValueError(
                    f"{module_id}:{context} references undeclared class {value}"
                )

        for concept in domain.get("concepts", []):
            require_declared_class(str(concept["parent"]), f"{concept['id']}.parent")
        for relation in domain.get("object_relations", []):
            require_declared_class(str(relation["domain"]), f"{relation['id']}.domain")
            require_declared_class(str(relation["range"]), f"{relation['id']}.range")
        for state_property in domain.get("state_properties", []):
            require_declared_class(
                str(state_property["domain"]),
                f"{state_property['id']}.domain",
            )
            data_range = str(state_property["range"])
            if not data_range.startswith("xsd:"):
                raise ValueError(
                    f"{module_id}:{state_property['id']} must use an xsd:* range"
                )
            if not str(state_property.get("unit", "")).strip():
                raise ValueError(
                    f"{module_id}:{state_property['id']} must declare an explicit unit"
                )
            if not str(state_property.get("evidence_field_category", "")).strip():
                raise ValueError(
                    f"{module_id}:{state_property['id']} must declare "
                    "evidence_field_category"
                )

        for predicate in domain.get("predicates", []):
            identifier = predicate["identifier"]
            if identifier in global_predicate_identifiers:
                raise ValueError(f"Duplicate predicate identifier: {identifier}")
            global_predicate_identifiers.add(identifier)
            role_keys = [role["key"] for role in predicate["roles"]]
            if not role_keys or len(role_keys) != len(set(role_keys)):
                raise ValueError(
                    f"{module_id}:{predicate['id']} must declare unique argument roles"
                )
            for role in predicate["roles"]:
                require_declared_class(
                    str(role["class"]),
                    f"{predicate['id']}.role.{role['key']}",
                )

        event_ids: set[str] = set()
        for event in events.get("events", []):
            identifier = event["identifier"]
            if identifier in global_event_identifiers:
                raise ValueError(f"Duplicate event identifier: {identifier}")
            global_event_identifiers.add(identifier)
            if event["id"] in event_ids:
                raise ValueError(f"{module_id}: duplicate event term {event['id']}")
            event_ids.add(event["id"])
            role_keys = [role["key"] for role in event["roles"]]
            if not role_keys or len(role_keys) != len(set(role_keys)):
                raise ValueError(
                    f"{module_id}:{event['id']} must declare unique participant roles"
                )
            for role in event["roles"]:
                require_declared_class(
                    str(role["class"]),
                    f"{event['id']}.role.{role['key']}",
                )


def _render_domain_ttl(module_id: str, spec: dict[str, Any]) -> str:
    slug = spec["slug"]
    ontology_iri = _domain_iri(slug)
    namespace = f"{ontology_iri}#"
    local_class_ids = {concept["id"] for concept in spec.get("concepts", [])}
    sections: list[str] = [
        PREFIXES,
        f"@prefix dom: <{namespace}> .",
        "",
        _statement(
            f"<{ontology_iri}>",
            [
                ("a", "owl:Ontology"),
                (
                    "owl:versionIRI",
                    f"<{ontology_iri}/{VERSION_IRI_SEGMENT}>",
                ),
                ("owl:versionInfo", _ttl_literal(VERSION)),
                ("owl:imports", f"<{WORLD_ONTOLOGY_IRI}>"),
                (
                    "dct:title",
                    _language_objects(spec["title_en"], spec["title_zh"]),
                ),
                (
                    "dct:description",
                    _language_objects(spec["scope_en"], spec["scope_zh"]),
                ),
                ("dct:creator", _ttl_literal("AeroWorld Project")),
                (
                    "dct:created",
                    f"{_ttl_literal(CREATED_DATE)}^^xsd:date",
                ),
                ("vann:preferredNamespacePrefix", _ttl_literal("dom")),
                ("vann:preferredNamespaceUri", _ttl_literal(namespace)),
            ],
        ),
    ]

    if spec.get("concepts"):
        sections.extend(
            [
                "",
                "#################################################################",
                "# Stable domain concepts",
                "#################################################################",
            ]
        )
    for concept in spec.get("concepts", []):
        sections.extend(
            [
                "",
                _statement(
                    f"dom:{concept['id']}",
                    [
                        ("a", "owl:Class"),
                        (
                            "rdfs:subClassOf",
                            _term(concept["parent"], "dom", local_class_ids),
                        ),
                        (
                            "rdfs:label",
                            _language_objects(concept["label_en"], concept["label_zh"]),
                        ),
                        (
                            "skos:definition",
                            _language_objects(
                                concept["definition_en"],
                                concept["definition_zh"],
                            ),
                        ),
                    ],
                ),
            ]
        )

    if spec.get("object_relations"):
        sections.extend(
            [
                "",
                "#################################################################",
                "# Stable domain relations",
                "#################################################################",
            ]
        )
    for relation in spec.get("object_relations", []):
        sections.extend(
            [
                "",
                _statement(
                    f"dom:{relation['id']}",
                    [
                        ("a", "owl:ObjectProperty"),
                        (
                            "rdfs:domain",
                            _term(relation["domain"], "dom", local_class_ids),
                        ),
                        (
                            "rdfs:range",
                            _term(relation["range"], "dom", local_class_ids),
                        ),
                        (
                            "rdfs:label",
                            _language_objects(
                                relation["label_en"], relation["label_zh"]
                            ),
                        ),
                        (
                            "skos:definition",
                            _language_objects(
                                relation["definition_en"],
                                relation["definition_zh"],
                            ),
                        ),
                    ],
                ),
            ]
        )

    if spec.get("state_properties"):
        sections.extend(
            [
                "",
                "#################################################################",
                "# Stable state datatype properties",
                "#################################################################",
            ]
        )
    for state_property in spec.get("state_properties", []):
        predicate_objects = [
            ("a", "owl:DatatypeProperty"),
            (
                "rdfs:domain",
                _term(state_property["domain"], "dom", local_class_ids),
            ),
            ("rdfs:range", state_property["range"]),
            (
                "rdfs:label",
                _language_objects(
                    state_property["label_en"], state_property["label_zh"]
                ),
            ),
            (
                "skos:definition",
                _language_objects(
                    state_property["definition_en"],
                    state_property["definition_zh"],
                ),
            ),
        ]
        if state_property.get("unit"):
            predicate_objects.append(
                ("world:unitSymbol", _ttl_literal(state_property["unit"]))
            )
        if state_property.get("evidence_field_category"):
            predicate_objects.append(
                (
                    "world:stateFieldCategory",
                    _ttl_literal(state_property["evidence_field_category"]),
                )
            )
        sections.extend(
            ["", _statement(f"dom:{state_property['id']}", predicate_objects)]
        )

    sections.extend(
        [
            "",
            "#################################################################",
            "# Stable objective-predicate signatures",
            "#################################################################",
        ]
    )
    for predicate in spec.get("predicates", []):
        role_terms: list[str] = []
        for role in predicate["roles"]:
            role_term = f"dom:{predicate['id']}Argument_{role['key']}"
            role_terms.append(role_term)
            sections.extend(
                [
                    "",
                    _statement(
                        role_term,
                        [
                            ("a", "aw:ArgumentRole"),
                            (
                                "aw:argumentRoleKey",
                                _ttl_literal(role["key"]),
                            ),
                            (
                                "world:argumentClass",
                                _term(role["class"], "dom", local_class_ids),
                            ),
                            (
                                "rdfs:label",
                                _ttl_literal(role["key"].replace("_", " "), "en"),
                            ),
                        ],
                    ),
                ]
            )
        sections.extend(
            [
                "",
                _statement(
                    f"dom:{predicate['id']}",
                    [
                        ("a", "aw:PredicateType"),
                        (
                            "aw:predicateIdentifier",
                            _ttl_literal(predicate["identifier"]),
                        ),
                        (
                            "aw:declaresArgumentRole",
                            " ,\n        ".join(role_terms),
                        ),
                        (
                            "rdfs:label",
                            _language_objects(
                                predicate["label_en"], predicate["label_zh"]
                            ),
                        ),
                        (
                            "skos:definition",
                            _language_objects(
                                predicate["definition_en"],
                                predicate["definition_zh"],
                            ),
                        ),
                    ],
                ),
            ]
        )
    return "\n".join(sections)


def _render_domain_shacl(module_id: str, spec: dict[str, Any]) -> str:
    slug = spec["slug"]
    ontology_iri = _domain_iri(slug)
    namespace = f"{ontology_iri}#"
    base = f"""\
{PREFIXES}
@prefix dom: <{namespace}> .
@prefix shape: <{ontology_iri}/shapes#> .

<{ontology_iri}/shapes>
    a owl:Ontology ;
    owl:imports <{ontology_iri}> ;
    owl:versionInfo {_ttl_literal(VERSION)} .

shape:PredicateSignatureShape
    a sh:NodeShape ;
    sh:targetClass aw:PredicateType ;
    sh:closed true ;
    sh:ignoredProperties ( rdf:type ) ;
    sh:property [
        sh:path aw:predicateIdentifier ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path aw:declaresArgumentRole ;
        sh:class aw:ArgumentRole ;
        sh:minCount 1
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:property [
        sh:path skos:definition ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:sparql [
        a sh:SPARQLConstraint ;
        sh:message "One predicate cannot declare duplicate argument-role keys."@en ;
        sh:select \"\"\"
            SELECT $this
            WHERE {{
                $this aw:declaresArgumentRole ?left, ?right .
                ?left aw:argumentRoleKey ?key .
                ?right aw:argumentRoleKey ?key .
                FILTER (?left != ?right)
            }}
        \"\"\"
    ] .

shape:ArgumentRoleShape
    a sh:NodeShape ;
    sh:targetObjectsOf aw:declaresArgumentRole ;
    sh:closed true ;
    sh:ignoredProperties ( rdf:type ) ;
    sh:property [
        sh:path aw:argumentRoleKey ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path world:argumentClass ;
        sh:nodeKind sh:IRI ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 1
    ] .

shape:DomainClassShape
    a sh:NodeShape ;
    sh:targetSubjectsOf rdfs:subClassOf ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:property [
        sh:path skos:definition ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] .

shape:DomainRelationShape
    a sh:NodeShape ;
    sh:targetClass owl:ObjectProperty ;
    sh:property [
        sh:path rdfs:domain ;
        sh:nodeKind sh:IRI ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path rdfs:range ;
        sh:nodeKind sh:IRI ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path skos:definition ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] .

shape:StateDatatypePropertyShape
    a sh:NodeShape ;
    sh:targetSubjectsOf world:stateFieldCategory ;
    sh:property [
        sh:path rdf:type ;
        sh:hasValue owl:DatatypeProperty
    ] ;
    sh:property [
        sh:path rdfs:domain ;
        sh:nodeKind sh:IRI ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path rdfs:range ;
        sh:nodeKind sh:IRI ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:property [
        sh:path skos:definition ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:property [
        sh:path world:unitSymbol ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path world:stateFieldCategory ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] .
"""
    local_class_ids = {concept["id"] for concept in spec.get("concepts", [])}
    exact_domain_range_shapes: list[str] = []
    for relation in spec.get("object_relations", []):
        domain = _term(relation["domain"], "dom", local_class_ids)
        range_ = _term(relation["range"], "dom", local_class_ids)
        exact_domain_range_shapes.append(
            f"""
shape:{relation['id']}ExactDomainRangeShape
    a sh:NodeShape ;
    sh:targetNode dom:{relation['id']} ;
    sh:property [
        sh:path rdfs:domain ;
        sh:hasValue {domain}
    ] ;
    sh:property [
        sh:path rdfs:range ;
        sh:hasValue {range_}
    ] .
"""
        )
    exact_role_shapes: list[str] = []
    for predicate in spec.get("predicates", []):
        roles = [
            f"dom:{predicate['id']}Argument_{role['key']}"
            for role in predicate["roles"]
        ]
        exact_role_shapes.append(
            f"""
shape:{predicate['id']}ExactArgumentRolesShape
    a sh:NodeShape ;
    sh:targetNode dom:{predicate['id']} ;
    sh:property [
        sh:path aw:declaresArgumentRole ;
        sh:minCount {len(roles)} ;
        sh:maxCount {len(roles)} ;
        sh:in ( {' '.join(roles)} )
    ] .
"""
        )
    return base + "".join(exact_domain_range_shapes + exact_role_shapes)


def _render_event_ttl(
    module_id: str, domain_spec: dict[str, Any], event_spec: dict[str, Any]
) -> str:
    slug = domain_spec["slug"]
    ontology_iri = _event_iri(slug)
    namespace = f"{ontology_iri}#"
    domain_iri = _domain_iri(slug)
    local_class_ids = {concept["id"] for concept in domain_spec.get("concepts", [])}
    sections: list[str] = [
        PREFIXES,
        f"@prefix evt: <{namespace}> .",
        f"@prefix dom: <{domain_iri}#> .",
        "",
        _statement(
            f"<{ontology_iri}>",
            [
                ("a", "owl:Ontology"),
                (
                    "owl:versionIRI",
                    f"<{ontology_iri}/{VERSION_IRI_SEGMENT}>",
                ),
                ("owl:versionInfo", _ttl_literal(VERSION)),
                (
                    "owl:imports",
                    f"<{WORLD_ONTOLOGY_IRI}> ,\n        <{domain_iri}>",
                ),
                (
                    "dct:title",
                    _language_objects(event_spec["title_en"], event_spec["title_zh"]),
                ),
                (
                    "dct:description",
                    _language_objects(event_spec["scope_en"], event_spec["scope_zh"]),
                ),
                ("dct:creator", _ttl_literal("AeroWorld Project")),
                (
                    "dct:created",
                    f"{_ttl_literal(CREATED_DATE)}^^xsd:date",
                ),
                ("vann:preferredNamespacePrefix", _ttl_literal("evt")),
                ("vann:preferredNamespaceUri", _ttl_literal(namespace)),
            ],
        ),
        "",
        "#################################################################",
        "# Stable objective event types and complete participant roles",
        "#################################################################",
    ]
    for event in event_spec.get("events", []):
        role_terms: list[str] = []
        for role in event["roles"]:
            role_term = f"evt:{event['id']}Role_{role['key']}"
            role_terms.append(role_term)
            sections.extend(
                [
                    "",
                    _statement(
                        role_term,
                        [
                            ("a", "aw:ParticipantRole"),
                            (
                                "aw:participantRoleKey",
                                _ttl_literal(role["key"]),
                            ),
                            (
                                "world:participantClass",
                                _term(role["class"], "dom", local_class_ids),
                            ),
                            (
                                "rdfs:label",
                                _ttl_literal(role["key"].replace("_", " "), "en"),
                            ),
                        ],
                    ),
                ]
            )
        sections.extend(
            [
                "",
                _statement(
                    f"evt:{event['id']}",
                    [
                        ("a", "aw:EventType"),
                        (
                            "aw:eventTypeIdentifier",
                            _ttl_literal(event["identifier"]),
                        ),
                        (
                            "aw:declaresParticipantRole",
                            " ,\n        ".join(role_terms),
                        ),
                        (
                            "rdfs:label",
                            _language_objects(event["label_en"], event["label_zh"]),
                        ),
                        (
                            "skos:definition",
                            _language_objects(
                                event["definition_en"],
                                event["definition_zh"],
                            ),
                        ),
                    ],
                ),
            ]
        )
    return "\n".join(sections)


def _render_event_shacl(
    module_id: str,
    domain_spec: dict[str, Any],
    event_spec: dict[str, Any],
) -> str:
    slug = domain_spec["slug"]
    ontology_iri = _event_iri(slug)
    namespace = f"{ontology_iri}#"
    base = f"""\
{PREFIXES}
@prefix evt: <{namespace}> .
@prefix shape: <{ontology_iri}/shapes#> .

<{ontology_iri}/shapes>
    a owl:Ontology ;
    owl:imports <{ontology_iri}> ;
    owl:versionInfo {_ttl_literal(VERSION)} .

shape:StableEventTypeShape
    a sh:NodeShape ;
    sh:targetClass aw:EventType ;
    sh:closed true ;
    sh:ignoredProperties ( rdf:type ) ;
    sh:property [
        sh:path aw:eventTypeIdentifier ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path aw:declaresParticipantRole ;
        sh:class aw:ParticipantRole ;
        sh:minCount 1
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:property [
        sh:path skos:definition ;
        sh:minCount 2 ;
        sh:uniqueLang true
    ] ;
    sh:sparql [
        a sh:SPARQLConstraint ;
        sh:message "One event type cannot declare duplicate participant-role keys."@en ;
        sh:select \"\"\"
            SELECT $this
            WHERE {{
                $this aw:declaresParticipantRole ?left, ?right .
                ?left aw:participantRoleKey ?key .
                ?right aw:participantRoleKey ?key .
                FILTER (?left != ?right)
            }}
        \"\"\"
    ] .

shape:CompleteParticipantRoleShape
    a sh:NodeShape ;
    sh:targetObjectsOf aw:declaresParticipantRole ;
    sh:closed true ;
    sh:ignoredProperties ( rdf:type ) ;
    sh:property [
        sh:path aw:participantRoleKey ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path world:participantClass ;
        sh:nodeKind sh:IRI ;
        sh:minCount 1 ;
        sh:maxCount 1
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 1
    ] .
"""
    exact_role_shapes: list[str] = []
    for event in event_spec.get("events", []):
        roles = [f"evt:{event['id']}Role_{role['key']}" for role in event["roles"]]
        exact_role_shapes.append(
            f"""
shape:{event['id']}ExactParticipantRolesShape
    a sh:NodeShape ;
    sh:targetNode evt:{event['id']} ;
    sh:property [
        sh:path aw:declaresParticipantRole ;
        sh:minCount {len(roles)} ;
        sh:maxCount {len(roles)} ;
        sh:in ( {' '.join(roles)} )
    ] .
"""
        )
    return base + "".join(exact_role_shapes)


def _render_manifest(
    domain_catalog: dict[str, Any],
    event_catalog: dict[str, Any],
    generated_artifacts: dict[Path, str],
) -> str:
    domain_modules = []
    event_modules = []
    for module_id, domain_spec in _catalog_modules(domain_catalog, "Domain").items():
        event_spec = _catalog_modules(event_catalog, "Event")[module_id]
        slug = domain_spec["slug"]
        domain_modules.append(
            {
                "module": module_id,
                "ontology_iri": _domain_iri(slug),
                "source": str(DOMAIN_CATALOG.relative_to(REPO_ROOT)).replace("\\", "/"),
                "ttl": str((DOMAIN_DIR / f"{slug}.ttl").relative_to(REPO_ROOT)).replace(
                    "\\", "/"
                ),
                "shacl": str(
                    (DOMAIN_DIR / f"{slug}.shacl.ttl").relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "concept_count": len(domain_spec.get("concepts", [])),
                "relation_count": len(domain_spec.get("object_relations", [])),
                "state_property_count": len(domain_spec.get("state_properties", [])),
                "predicate_count": len(domain_spec.get("predicates", [])),
            }
        )
        event_modules.append(
            {
                "module": module_id,
                "ontology_iri": _event_iri(slug),
                "source": str(EVENT_CATALOG.relative_to(REPO_ROOT)).replace("\\", "/"),
                "ttl": str(
                    (EVENT_DIR / f"{slug}_events.ttl").relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "shacl": str(
                    (EVENT_DIR / f"{slug}_events.shacl.ttl").relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "event_type_count": len(event_spec.get("events", [])),
            }
        )

    artifacts = [
        {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        for path in sorted(generated_artifacts)
    ]
    manifest = {
        "schema": "aeroworld.semantic-tbox-release.v1",
        "version": VERSION,
        "status": "current",
        "scope": "objective_semantic_ontology",
        "world_core_import": WORLD_ONTOLOGY_IRI,
        "authority": {
            "domain_tbox": (
                "Stable concepts, relations, and objective-predicate signatures."
            ),
            "event_tbox": (
                "Stable objective event types and complete participant roles."
            ),
            "predicate_evaluation": "Grounded predicate registry and evaluator.",
            "event_compilation": "Event-family and stage contracts.",
            "dataset_binding": "Ontology selection manifest.",
        },
        "validation": {
            "dataset_independent_domain_terms": True,
            "dataset_independent_event_terms": True,
            "catalog_declared_event_roles_structurally_validated": True,
            "executable_rule_contract": "build_event_contracts.py",
            "episode_selection": "ontology_selection_manifest.json",
        },
        "domain_modules": domain_modules,
        "event_modules": event_modules,
        "artifacts": artifacts,
    }
    return yaml.safe_dump(
        manifest,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def build(*, check: bool = False) -> list[Path]:
    domain_catalog = _load_yaml(DOMAIN_CATALOG)
    event_catalog = _load_yaml(EVENT_CATALOG)
    _validate_source_catalogs(domain_catalog, event_catalog)

    generated_artifacts: dict[Path, str] = {}
    for module_id, domain_spec in _catalog_modules(domain_catalog, "Domain").items():
        event_spec = _catalog_modules(event_catalog, "Event")[module_id]
        slug = domain_spec["slug"]
        domain_ttl = DOMAIN_DIR / f"{slug}.ttl"
        domain_shacl = DOMAIN_DIR / f"{slug}.shacl.ttl"
        event_ttl = EVENT_DIR / f"{slug}_events.ttl"
        event_shacl = EVENT_DIR / f"{slug}_events.shacl.ttl"

        generated_artifacts[domain_ttl] = _normalize_text(
            _render_domain_ttl(module_id, domain_spec)
        )
        generated_artifacts[domain_shacl] = _normalize_text(
            _render_domain_shacl(module_id, domain_spec)
        )
        generated_artifacts[event_ttl] = _normalize_text(
            _render_event_ttl(module_id, domain_spec, event_spec)
        )
        generated_artifacts[event_shacl] = _normalize_text(
            _render_event_shacl(module_id, domain_spec, event_spec)
        )

    for path, content in generated_artifacts.items():
        if path.suffix == ".ttl":
            Graph().parse(data=content, format="turtle")

    manifest_text = _normalize_text(
        _render_manifest(domain_catalog, event_catalog, generated_artifacts)
    )
    expected_artifacts = {
        **generated_artifacts,
        MANIFEST_PATH: manifest_text,
    }
    expected_tbox_paths = set(generated_artifacts)
    unexpected_tbox_paths = sorted(
        {
            *DOMAIN_DIR.glob("*.ttl"),
            *EVENT_DIR.glob("*.ttl"),
        }
        - expected_tbox_paths
    )
    if unexpected_tbox_paths:
        raise RuntimeError(
            "Unexpected ontology module artifacts violate the exact "
            "12-module contract:\n"
            + "\n".join(
                str(path.relative_to(REPO_ROOT)) for path in unexpected_tbox_paths
            )
        )
    if check:
        stale: list[str] = []
        for path, expected in expected_artifacts.items():
            if not path.is_file():
                stale.append(f"missing: {path.relative_to(REPO_ROOT)}")
                continue
            actual = _normalize_text(path.read_text(encoding="utf-8"))
            if actual != expected:
                stale.append(f"stale: {path.relative_to(REPO_ROOT)}")
        if stale:
            raise RuntimeError(
                "Generated semantic ontology artifacts are not current:\n"
                + "\n".join(stale)
            )
    else:
        for path, content in generated_artifacts.items():
            _write_text(path, content)
        _write_text(MANIFEST_PATH, manifest_text)
    return list(expected_artifacts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated artifacts without writing any file.",
    )
    args = parser.parse_args()
    paths = build(check=args.check)
    for path in paths:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
