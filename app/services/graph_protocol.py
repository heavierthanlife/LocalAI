"""Unified graph protocol — all spider-web endpoints return this shape."""

from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    label: str
    type: str  # company|person|document|finding|law|template|project|entity|file
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str  # collusion_signal|text_similar|violates|cites|impacted_by|...
    weight: float = 1.0


# Type → default color (used by frontend when metadata doesn't override)
NODE_TYPE_COLORS = {
    'company': '#e74c3c',
    'person': '#f39c12',
    'document': '#3498db',
    'finding': '#e67e22',
    'law': '#2c3e50',
    'template': '#27ae60',
    'project': '#9b59b6',
    'entity': '#1abc9c',
    'file': '#2980b9',
}

# Edge type → default color
EDGE_TYPE_COLORS = {
    'collusion_signal': '#e74c3c',
    'text_similar': '#e67e22',
    'shared_entity': '#f39c12',
    'price_anomaly': '#c0392b',
    'violates': '#e74c3c',
    'cites': '#3498db',
    'found_in': '#9b59b6',
    'impacted_by': '#f39c12',
    'violated_by': '#e67e22',
    'links_to': '#95a5a6',
}


def to_graph_response(nodes: list[GraphNode], edges: list[GraphEdge], web_type: str) -> dict:
    return {
        'nodes': [
            {
                'id': n.id,
                'label': n.label,
                'type': n.type,
                'color': n.metadata.pop('color', NODE_TYPE_COLORS.get(n.type, '#95a5a6')),
                **n.metadata,
            }
            for n in nodes
        ],
        'edges': [
            {
                'id': f'{e.source}-{e.target}-{e.label}',
                'source': e.source,
                'target': e.target,
                'label': e.label,
                'weight': e.weight,
                'color': EDGE_TYPE_COLORS.get(e.label, '#95a5a6'),
            }
            for e in edges
        ],
        'stats': {
            'node_count': len(nodes),
            'edge_count': len(edges),
        },
        'web_type': web_type,
    }
