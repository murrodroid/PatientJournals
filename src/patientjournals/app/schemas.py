from __future__ import annotations

import hashlib
import json
import re
import secrets
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from patientjournals.app.job_store import JobStore, utc_now_iso
from patientjournals.config.schemas import list_output_schemas
from patientjournals.data.bucket import build_storage_bucket, normalize_prefix


_ACTIVE_SCHEMA_KEY = "active_schema_version_id"
_COLUMN_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\[\])?(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[\])?)*$"
)
_SUPPORTED_TYPES = {
    "string",
    "date",
    "integer",
    "number",
    "boolean",
    "list[string]",
    "list[integer]",
    "list[number]",
    "list[boolean]",
}


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _version_id(*, family_id: str, name: str, schema_json: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {"family_id": family_id, "name": name, "schema_json": schema_json}
        ).encode("utf-8")
    ).hexdigest()
    return f"sv_{digest[:20]}"


def _safe_family_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return value or "schema"


def _definitions(schema: dict[str, Any]) -> dict[str, Any]:
    value = schema.get("$defs")
    if not isinstance(value, dict):
        value = schema.get("definitions")
    return value if isinstance(value, dict) else {}


def _unwrap_node(
    root: dict[str, Any],
    node: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    nullable = False
    ref = node.get("$ref")
    if isinstance(ref, str):
        for marker in ("#/$defs/", "#/definitions/"):
            if ref.startswith(marker):
                target = _definitions(root).get(ref[len(marker) :])
                if isinstance(target, dict):
                    return _unwrap_node(root, target)
    variants = node.get("anyOf") or node.get("oneOf")
    if isinstance(variants, list):
        concrete: list[dict[str, Any]] = []
        for item in variants:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "null":
                nullable = True
            else:
                concrete.append(item)
        if len(concrete) == 1:
            unwrapped, nested_nullable = _unwrap_node(root, concrete[0])
            return unwrapped, nullable or nested_nullable
    all_of = node.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        return _unwrap_node(root, all_of[0])
    return node, nullable


def _field_type(root: dict[str, Any], raw_node: dict[str, Any]) -> str:
    node, _nullable = _unwrap_node(root, raw_node)
    node_type = node.get("type")
    if node_type == "array":
        items = node.get("items")
        if isinstance(items, dict):
            item_node, _ = _unwrap_node(root, items)
            item_type = str(item_node.get("type") or "string")
            if item_type in {"string", "integer", "number", "boolean"}:
                return f"list[{item_type}]"
        return "list[string]"
    if node_type == "string" and node.get("format") == "date":
        return "date"
    return str(node_type or "string") if node_type in {
        "string",
        "integer",
        "number",
        "boolean",
    } else "string"


def flatten_schema_fields(schema: dict[str, Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []

    def walk(raw_node: dict[str, Any], prefix: str = "") -> None:
        node, nullable = _unwrap_node(schema, raw_node)
        if node.get("type") == "array":
            items = node.get("items")
            if isinstance(items, dict):
                item_node, _ = _unwrap_node(schema, items)
                if item_node.get("type") == "object" or isinstance(
                    item_node.get("properties"), dict
                ):
                    walk(item_node, f"{prefix}[]")
                    return
        properties = node.get("properties")
        if isinstance(properties, dict):
            required = {
                str(item) for item in (node.get("required") or []) if isinstance(item, str)
            }
            for name, child in properties.items():
                if not isinstance(child, dict):
                    continue
                path = f"{prefix}.{name}" if prefix else str(name)
                child_node, _ = _unwrap_node(schema, child)
                child_properties = child_node.get("properties")
                child_items = child_node.get("items")
                item_node = None
                if isinstance(child_items, dict):
                    item_node, _ = _unwrap_node(schema, child_items)
                if isinstance(child_properties, dict):
                    walk(child, path)
                elif isinstance(item_node, dict) and (
                    item_node.get("type") == "object"
                    or isinstance(item_node.get("properties"), dict)
                ):
                    walk(child, path)
                else:
                    fields.append(
                        {
                            "path": path,
                            "type": _field_type(schema, child),
                            "required": str(name) in required,
                            "description": str(child.get("description") or "").strip(),
                        }
                    )
            return
        if prefix:
            fields.append(
                {
                    "path": prefix,
                    "type": _field_type(schema, raw_node),
                    "required": not nullable,
                    "description": str(raw_node.get("description") or "").strip(),
                }
            )

    walk(schema)
    return sorted(fields, key=lambda item: str(item["path"]))


def dataset_schema_field_paths(record: dict[str, Any]) -> set[str]:
    """Return leaf paths as they appear in persisted dataset rows."""
    schema_json = record.get("schema_json")
    if not isinstance(schema_json, dict):
        return set()
    paths = {str(item["path"]) for item in flatten_schema_fields(schema_json)}
    if str(record.get("name") or "").strip().lower() == "textpage":
        prefix = "page_lines[]."
        return {
            path[len(prefix) :] if path.startswith(prefix) else path
            for path in paths
        }
    return paths


def _path_parts(path: str) -> list[tuple[str, bool]]:
    parts: list[tuple[str, bool]] = []
    for raw in path.split("."):
        is_array = raw.endswith("[]")
        name = raw[:-2] if is_array else raw
        parts.append((name, is_array))
    return parts


def _object_node(root: dict[str, Any], raw_node: dict[str, Any]) -> dict[str, Any]:
    node, _ = _unwrap_node(root, raw_node)
    if node.get("type") == "array":
        items = node.get("items")
        if isinstance(items, dict):
            node, _ = _unwrap_node(root, items)
    return node


def _locate_property(
    root: dict[str, Any],
    path: str,
    *,
    create: bool,
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    current = root
    parts = _path_parts(path)
    for index, (name, is_array) in enumerate(parts):
        current = _object_node(root, current)
        properties = current.get("properties")
        if not isinstance(properties, dict):
            if not create:
                return None
            properties = {}
            current["type"] = "object"
            current["properties"] = properties
            current.setdefault("additionalProperties", False)
        child = properties.get(name)
        is_last = index == len(parts) - 1
        if child is None:
            if not create:
                return None
            if is_last:
                child = {}
            elif is_array:
                child = {
                    "type": "array",
                    "items": {"type": "object", "properties": {}, "additionalProperties": False},
                }
            else:
                child = {"type": "object", "properties": {}, "additionalProperties": False}
            properties[name] = child
        if not isinstance(child, dict):
            return None
        if is_last:
            return current, name, child
        if is_array and child.get("type") != "array":
            if not create:
                return None
            child.clear()
            child.update(
                {
                    "type": "array",
                    "items": {"type": "object", "properties": {}, "additionalProperties": False},
                }
            )
        current = child
    return None


def _schema_for_type(field_type: str, description: str) -> dict[str, Any]:
    if field_type.startswith("list["):
        item_type = field_type[5:-1]
        payload: dict[str, Any] = {"type": "array", "items": {"type": item_type}}
    elif field_type == "date":
        payload = {"type": "string", "format": "date"}
    else:
        payload = {"type": field_type}
    if description:
        payload["description"] = description
    return payload


def apply_schema_fields(
    base_schema: dict[str, Any] | None,
    *,
    name: str,
    fields: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    schema = deepcopy(base_schema) if base_schema else {
        "title": name,
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    desired: dict[str, dict[str, Any]] = {}
    for raw in fields:
        path = str(raw.get("path") or "").strip()
        field_type = str(raw.get("type") or "string").strip().lower()
        if not _COLUMN_PATH_RE.fullmatch(path):
            raise ValueError(
                f"Invalid column path '{path}'. Use letters, numbers, underscores, dots, and [] for object lists."
            )
        if field_type not in _SUPPORTED_TYPES:
            raise ValueError(f"Unsupported type '{field_type}' for {path}.")
        if path in desired:
            raise ValueError(f"Duplicate column path: {path}")
        desired[path] = {
            "path": path,
            "type": field_type,
            "required": bool(raw.get("required")),
            "description": str(raw.get("description") or "").strip(),
        }
    if not desired:
        raise ValueError("A schema must contain at least one leaf column.")

    existing = {str(item["path"]): item for item in flatten_schema_fields(schema)}
    for path in existing.keys() - desired.keys():
        located = _locate_property(schema, path, create=False)
        if located is None:
            continue
        parent, key, _node = located
        properties = parent.get("properties")
        if isinstance(properties, dict):
            properties.pop(key, None)
        required = parent.get("required")
        if isinstance(required, list):
            parent["required"] = [item for item in required if item != key]

    for path, spec in desired.items():
        located = _locate_property(schema, path, create=True)
        if located is None:
            raise ValueError(f"Could not create column path: {path}")
        parent, key, node = located
        previous_type = existing.get(path, {}).get("type")
        if previous_type != spec["type"]:
            node.clear()
            node.update(_schema_for_type(spec["type"], spec["description"]))
        else:
            if spec["description"]:
                node["description"] = spec["description"]
            else:
                node.pop("description", None)
        required = parent.get("required")
        if not isinstance(required, list):
            required = []
        if spec["required"] and key not in required:
            required.append(key)
        if not spec["required"]:
            required = [item for item in required if item != key]
        parent["required"] = sorted(str(item) for item in required)

    def prune_empty_branches(raw_node: dict[str, Any], prefix: str = "") -> None:
        node = _object_node(schema, raw_node)
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return
        for key, child in list(properties.items()):
            if not isinstance(child, dict):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            child_node, _ = _unwrap_node(schema, child)
            item_node = None
            if child_node.get("type") == "array" and isinstance(
                child_node.get("items"), dict
            ):
                item_node, _ = _unwrap_node(schema, child_node["items"])
            nested = (
                child_node
                if isinstance(child_node.get("properties"), dict)
                else item_node
                if isinstance(item_node, dict)
                and isinstance(item_node.get("properties"), dict)
                else None
            )
            nested_prefix = f"{path}[]" if nested is item_node else path
            has_desired = any(
                candidate == path
                or candidate.startswith(f"{path}.")
                or candidate.startswith(f"{path}[].")
                for candidate in desired
            )
            if not has_desired:
                properties.pop(key, None)
                required = node.get("required")
                if isinstance(required, list):
                    node["required"] = [item for item in required if item != key]
                continue
            if nested is not None:
                prune_empty_branches(child, nested_prefix)

    prune_empty_branches(schema)

    schema["title"] = name
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    return schema


@dataclass
class SchemaService:
    store: JobStore
    bucket_name: str = ""
    schemas_prefix: str = "schemas"

    def __post_init__(self) -> None:
        self.bootstrap_builtin_schemas()

    def bootstrap_builtin_schemas(self) -> None:
        for name, model in list_output_schemas().items():
            schema_json = model.model_json_schema()
            family_id = f"builtin_{_safe_family_name(name)}"
            record = {
                "version_id": _version_id(
                    family_id=family_id,
                    name=name,
                    schema_json=schema_json,
                ),
                "family_id": family_id,
                "name": name,
                "version_number": 1,
                "parent_version_id": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "created_by": "built-in",
                "prompt_name": name.lower(),
                "source": "builtin",
                "schema_json": schema_json,
            }
            self.store.upsert_schema_version(record)
        if not self.store.schema_state(_ACTIVE_SCHEMA_KEY):
            frontpage = self.resolve_version("FrontPage")
            if frontpage:
                self.store.set_schema_state(_ACTIVE_SCHEMA_KEY, frontpage["version_id"])

    @property
    def _prefix(self) -> str:
        return normalize_prefix(self.schemas_prefix)

    def _index_object_name(self) -> str:
        return f"{self._prefix}index.json"

    def _version_object_name(self, version_id: str) -> str:
        return f"{self._prefix}versions/{version_id}.json"

    def _bucket(self):
        return build_storage_bucket(self.bucket_name or None)

    def _cloud_index(self) -> dict[str, Any] | None:
        if not self.bucket_name.strip():
            return None
        blob = self._bucket().blob(self._index_object_name())
        try:
            text = blob.download_as_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            label = f"{type(exc).__name__}: {exc}".lower()
            if "notfound" in label or "not found" in label or "404" in label:
                return {}
            raise
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Cloud schema index is not a JSON object.")
        return payload

    def sync_from_cloud(self) -> dict[str, Any]:
        if not self.bucket_name.strip():
            return {"status": "local_only", "error": ""}
        try:
            payload = self._cloud_index()
            if payload == {}:
                pushed = self.push_to_cloud()
                if pushed.get("error"):
                    return pushed
                return {"status": "initialized", "error": ""}
            if payload is None:
                return {"status": "local_only", "error": ""}
            for record in payload.get("versions") or []:
                if isinstance(record, dict) and isinstance(record.get("schema_json"), dict):
                    expected_id = _version_id(
                        family_id=str(record.get("family_id") or ""),
                        name=str(record.get("name") or ""),
                        schema_json=record["schema_json"],
                    )
                    if str(record.get("version_id") or "") != expected_id:
                        raise ValueError(
                            "Cloud schema index contains a version whose ID does not match its content."
                        )
                    imported = dict(record)
                    imported["source"] = str(record.get("source") or "cloud")
                    self.store.upsert_schema_version(imported)
            active_id = str(payload.get("active_version_id") or "")
            if active_id and self.store.schema_version(active_id):
                self.store.set_schema_state(_ACTIVE_SCHEMA_KEY, active_id)
            return {"status": "synced", "error": ""}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def push_to_cloud(self) -> dict[str, Any]:
        if not self.bucket_name.strip():
            return {"status": "local_only", "error": ""}
        try:
            bucket = self._bucket()
            records = self.store.list_schema_versions()
            for record in records:
                bucket.blob(self._version_object_name(record["version_id"])).upload_from_string(
                    json.dumps(record, indent=2, ensure_ascii=False),
                    content_type="application/json",
                )
            index = {
                "schema_version": 1,
                "updated_at": utc_now_iso(),
                "active_version_id": self.store.schema_state(_ACTIVE_SCHEMA_KEY),
                "versions": records,
            }
            bucket.blob(self._index_object_name()).upload_from_string(
                json.dumps(index, indent=2, ensure_ascii=False),
                content_type="application/json",
            )
            return {"status": "synced", "error": ""}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def list_versions(self, *, sync_cloud: bool = True) -> dict[str, Any]:
        sync = self.sync_from_cloud() if sync_cloud else {"status": "not_checked", "error": ""}
        active_id = self.store.schema_state(_ACTIVE_SCHEMA_KEY)
        versions = []
        for record in self.store.list_schema_versions():
            item = dict(record)
            item["is_active"] = item["version_id"] == active_id
            item["fields"] = flatten_schema_fields(item["schema_json"])
            item["field_count"] = len(item["fields"])
            versions.append(item)
        versions.sort(
            key=lambda item: (
                not bool(item["is_active"]),
                str(item["name"]).lower(),
                -int(item["version_number"]),
            )
        )
        return {
            "versions": versions,
            "active_version_id": active_id,
            "cloud_sync": sync,
        }

    def resolve_version(self, identifier: str = "") -> dict[str, Any]:
        value = str(identifier or "").strip()
        if value:
            direct = self.store.schema_version(value)
            if direct:
                return direct
        records = self.store.list_schema_versions()
        active_id = self.store.schema_state(_ACTIVE_SCHEMA_KEY)
        matches = [
            item
            for item in records
            if not value or str(item.get("name") or "").lower() == value.lower()
        ]
        if not matches:
            return {}
        active = next((item for item in matches if item["version_id"] == active_id), None)
        return active or max(matches, key=lambda item: int(item.get("version_number") or 0))

    def validation_field_paths_by_version(self) -> dict[str, set[str]]:
        return {
            str(record["version_id"]): dataset_schema_field_paths(record)
            for record in self.store.list_schema_versions()
        }

    def create_version(
        self,
        *,
        name: str,
        fields: Iterable[dict[str, Any]],
        created_by: str,
        parent_version_id: str = "",
        make_active: bool = False,
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("Enter a schema name.")
        sync = self.sync_from_cloud()
        parent = self.store.schema_version(parent_version_id) if parent_version_id else {}
        if parent_version_id and not parent:
            raise ValueError(f"Schema version not found: {parent_version_id}")
        family_id = str(parent.get("family_id") or "") or (
            f"schema_{_safe_family_name(clean_name)}_{secrets.token_hex(4)}"
        )
        family_versions = [
            item
            for item in self.store.list_schema_versions()
            if item.get("family_id") == family_id
        ]
        version_number = max(
            (int(item.get("version_number") or 0) for item in family_versions),
            default=0,
        ) + 1
        schema_json = apply_schema_fields(
            parent.get("schema_json") if parent else None,
            name=clean_name,
            fields=fields,
        )
        version_id = _version_id(
            family_id=family_id,
            name=clean_name,
            schema_json=schema_json,
        )
        existing = self.store.schema_version(version_id)
        if existing:
            raise ValueError(
                "This schema is identical to an existing version. Change a column before saving."
            )
        record = {
            "version_id": version_id,
            "family_id": family_id,
            "name": clean_name,
            "version_number": version_number,
            "parent_version_id": str(parent.get("version_id") or ""),
            "created_at": utc_now_iso(),
            "created_by": str(created_by or "unknown"),
            "prompt_name": str(parent.get("prompt_name") or "frontpage"),
            "source": "local",
            "schema_json": schema_json,
        }
        self.store.upsert_schema_version(record)
        if make_active:
            self.store.set_schema_state(_ACTIVE_SCHEMA_KEY, version_id)
        push = self.push_to_cloud()
        item = dict(record)
        item["fields"] = flatten_schema_fields(schema_json)
        item["field_count"] = len(item["fields"])
        item["is_active"] = make_active
        return {"version": item, "cloud_pull": sync, "cloud_sync": push}

    def set_active(self, version_id: str) -> dict[str, Any]:
        self.sync_from_cloud()
        record = self.store.schema_version(version_id)
        if not record:
            raise ValueError(f"Schema version not found: {version_id}")
        self.store.set_schema_state(_ACTIVE_SCHEMA_KEY, version_id)
        return {"active_version_id": version_id, "cloud_sync": self.push_to_cloud()}
