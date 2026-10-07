"""Entity render labels from explicit frozen asset declarations only."""
from functools import lru_cache
import json
from pathlib import Path


ASSET_CATALOG = Path(__file__).resolve().parents[2] / 'Config/LowAltitude/asset_catalog.json'


@lru_cache(maxsize=1)
def asset_labels():
    payload = json.loads(ASSET_CATALOG.read_text(encoding='utf-8-sig'))
    labels = {}
    for row in payload['assets']:
        identity, label = row['logical_asset_id'], row.get('label_class')
        if not isinstance(identity, str) or not identity or identity in labels:
            raise ValueError('asset catalog contains missing or duplicate identity')
        if not isinstance(label, str) or not label:
            raise ValueError(f'asset catalog lacks an explicit label_class: {identity}')
        labels[identity] = label
    return labels


def label_for_declared_asset(logical_asset_id):
    if not isinstance(logical_asset_id, str) or not logical_asset_id:
        raise ValueError('entity requires a declared logical_asset_id')
    try:
        return asset_labels()[logical_asset_id]
    except KeyError as exc:
        raise ValueError(f'entity asset has no typed catalog declaration: {logical_asset_id}') from exc
