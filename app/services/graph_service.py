"""Knowledge graph service (U12).

Builds a graph from interconnected compliance data:
  Nodes: law, article, case, template, project
  Edges: CONTAINS (law→article), CITES (case→article),
         RELATED_TO (case→template), HAS_FINDING (project→case)

Frontend: Cytoscape.js force-directed layout.
"""

import logging
from app.database import get_db_connection

logger = logging.getLogger(__name__)

NODE_TYPES = {
    'law': {'color': '#2980b9', 'shape': 'rectangle'},
    'article': {'color': '#8e44ad', 'shape': 'ellipse'},
    'case': {'color': '#e67e22', 'shape': 'round-rectangle'},
    'template': {'color': '#27ae60', 'shape': 'diamond'},
    'project': {'color': '#2c3e50', 'shape': 'hexagon'},
}


def _make_node(node_type: str, node_id: str, label: str, extra: dict = None) -> dict:
    info = NODE_TYPES.get(node_type, {'color': '#95a5a6', 'shape': 'ellipse'})
    node = {
        'id': f'{node_type}-{node_id}',
        'type': node_type,
        'label': label[:80],
        'color': info['color'],
        'shape': info['shape'],
    }
    if extra:
        node.update(extra)
    return node


def _make_edge(source_id: str, target_id: str, edge_type: str, label: str = '') -> dict:
    return {
        'source': source_id,
        'target': target_id,
        'type': edge_type,
        'label': label,
    }


def get_graph_data(center_type: str = None, center_id: int = None,
                   max_nodes: int = 100, depth: int = 2) -> dict:
    """Build knowledge graph data.

    Args:
        center_type: 'case', 'template', 'law', 'article', 'project' — anchor
        center_id: key ID for the center node
        max_nodes: cap node count
        depth: traversal depth from center

    Returns:
        {nodes: [...], edges: [...]}
    """
    nodes_dict = {}
    edges_list = []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if center_type and center_id:
                _build_centered_graph(cur, center_type, center_id, nodes_dict, edges_list, depth)
            else:
                _build_overview_graph(cur, nodes_dict, edges_list, max_nodes)

    # Trim
    nodes = list(nodes_dict.values())[:max_nodes]
    node_ids = {n['id'] for n in nodes}
    edges = [e for e in edges_list if e['source'] in node_ids and e['target'] in node_ids][:max_nodes * 2]

    return {
        'nodes': nodes,
        'edges': edges,
        'total_nodes': len(nodes),
        'total_edges': len(edges),
    }


def _build_overview_graph(cur, nodes_dict: dict, edges_list: list, max_nodes: int):
    """Build overview: most connected nodes."""
    # Laws + articles
    cur.execute("""
        SELECT l.id, l.law_name, a.id, a.article_label
        FROM laws l
        JOIN law_articles a ON a.law_id = l.id
        WHERE l.is_active = TRUE
        ORDER BY l.id LIMIT 20
    """)
    for row in cur.fetchall():
        lid, lname, aid, alabel = row
        nodes_dict[f'law-{lid}'] = _make_node('law', lid, lname)
        nodes_dict[f'article-{aid}'] = _make_node('article', aid, alabel)
        edges_list.append(_make_edge(f'law-{lid}', f'article-{aid}', 'CONTAINS'))

    # Cases + their law links
    cur.execute("""
        SELECT ac.id, ac.title, cl.article_id, la.article_label, la.law_id
        FROM audit_cases ac
        JOIN case_law_links cl ON cl.case_id = ac.id
        JOIN law_articles la ON la.id = cl.article_id
        ORDER BY ac.id LIMIT 30
    """)
    case_ids = set()
    for row in cur.fetchall():
        cid, ctitle, aid, alabel, lid = row
        case_ids.add(cid)
        nodes_dict[f'case-{cid}'] = _make_node('case', cid, ctitle)
        edges_list.append(_make_edge(f'case-{cid}', f'article-{aid}', 'CITES'))

    # Cases + their template links
    cur.execute("""
        SELECT ac.id, ct.template_id, bt.name
        FROM audit_cases ac
        JOIN case_template_links ct ON ct.case_id = ac.id
        JOIN bid_templates bt ON bt.id = ct.template_id
        WHERE ac.id = ANY(%s) LIMIT 50
    """, (list(case_ids) if case_ids else [],))
    for row in cur.fetchall():
        cid, tid, tname = row
        nodes_dict[f'template-{tid}'] = _make_node('template', tid, tname)
        edges_list.append(_make_edge(f'case-{cid}', f'template-{tid}', 'RELATED_TO'))

    # Templates
    cur.execute("""
        SELECT bt.id, bt.name FROM bid_templates bt
        WHERE bt.is_active = TRUE ORDER BY bt.id LIMIT 10
    """)
    for row in cur.fetchall():
        tid, tname = row
        nodes_dict[f'template-{tid}'] = _make_node('template', tid, tname)

    # Projects with cases
    cur.execute("""
        SELECT DISTINCT ar.project_id, ac.id
        FROM audit_runs ar
        JOIN audit_cases ac ON ac.source_run_id = ar.id
        WHERE ar.project_id IS NOT NULL LIMIT 20
    """)
    for row in cur.fetchall():
        pid, cid = row
        nodes_dict[f'project-{pid}'] = _make_node('project', pid, f'项目 #{pid}')
        edges_list.append(_make_edge(f'project-{pid}', f'case-{cid}', 'HAS_FINDING'))

    # Trim to max_nodes
    if len(nodes_dict) > max_nodes:
        ids = list(nodes_dict.keys())[:max_nodes]
        nodes_dict = {k: v for k, v in nodes_dict.items() if k in ids}
        keep_ids = set(nodes_dict.keys())
        edges_list = [e for e in edges_list if e['source'] in keep_ids and e['target'] in keep_ids]


def _build_centered_graph(cur, center_type: str, center_id: int,
                          nodes_dict: dict, edges_list: list, depth: int):
    """Build graph centered on a specific node, traversing depth levels."""
    center_key = f'{center_type}-{center_id}'

    if center_type == 'case':
        cur.execute("SELECT id, title FROM audit_cases WHERE id = %s", (center_id,))
        row = cur.fetchone()
        if row:
            nodes_dict[center_key] = _make_node('case', row[0], row[1])

        if depth >= 1:
            cur.execute("""
                SELECT cl.article_id, la.article_label, la.law_id, l.law_name
                FROM case_law_links cl
                JOIN law_articles la ON la.id = cl.article_id
                JOIN laws l ON l.id = la.law_id
                WHERE cl.case_id = %s
            """, (center_id,))
            for r in cur.fetchall():
                aid, alabel, lid, lname = r
                nodes_dict[f'law-{lid}'] = _make_node('law', lid, lname)
                nodes_dict[f'article-{aid}'] = _make_node('article', aid, alabel)
                edges_list.append(_make_edge(f'law-{lid}', f'article-{aid}', 'CONTAINS'))
                edges_list.append(_make_edge(center_key, f'article-{aid}', 'CITES'))

            cur.execute("""
                SELECT ct.template_id, bt.name
                FROM case_template_links ct
                JOIN bid_templates bt ON bt.id = ct.template_id
                WHERE ct.case_id = %s
            """, (center_id,))
            for r in cur.fetchall():
                tid, tname = r
                nodes_dict[f'template-{tid}'] = _make_node('template', tid, tname)
                edges_list.append(_make_edge(center_key, f'template-{tid}', 'RELATED_TO'))

            cur.execute("SELECT source_run_id FROM audit_cases WHERE id = %s", (center_id,))
            r = cur.fetchone()
            if r and r[0]:
                rid = r[0]
                cur.execute("SELECT project_id FROM audit_runs WHERE id = %s", (rid,))
                pr = cur.fetchone()
                if pr and pr[0]:
                    pid = pr[0]
                    nodes_dict[f'project-{pid}'] = _make_node('project', pid, f'项目 #{pid}')
                    edges_list.append(_make_edge(f'project-{pid}', center_key, 'HAS_FINDING'))

    elif center_type == 'template':
        cur.execute("SELECT id, name FROM bid_templates WHERE id = %s", (center_id,))
        row = cur.fetchone()
        if row:
            nodes_dict[center_key] = _make_node('template', row[0], row[1])

        if depth >= 1:
            cur.execute("""
                SELECT ct.case_id, ac.title
                FROM case_template_links ct
                JOIN audit_cases ac ON ac.id = ct.case_id
                WHERE ct.template_id = %s LIMIT 20
            """, (center_id,))
            for r in cur.fetchall():
                cid, ctitle = r
                nodes_dict[f'case-{cid}'] = _make_node('case', cid, ctitle)
                edges_list.append(_make_edge(f'case-{cid}', center_key, 'RELATED_TO'))

    elif center_type == 'law':
        cur.execute("SELECT id, law_name FROM laws WHERE id = %s", (center_id,))
        row = cur.fetchone()
        if row:
            nodes_dict[center_key] = _make_node('law', row[0], row[1])

        if depth >= 1:
            cur.execute("""
                SELECT la.id, la.article_label FROM law_articles la
                WHERE la.law_id = %s LIMIT 30
            """, (center_id,))
            for r in cur.fetchall():
                aid, alabel = r
                nodes_dict[f'article-{aid}'] = _make_node('article', aid, alabel)
                edges_list.append(_make_edge(center_key, f'article-{aid}', 'CONTAINS'))

            cur.execute("""
                SELECT la.id, la.article_label, cl.case_id, ac.title
                FROM law_articles la
                JOIN case_law_links cl ON cl.article_id = la.id
                JOIN audit_cases ac ON ac.id = cl.case_id
                WHERE la.law_id = %s LIMIT 20
            """, (center_id,))
            for r in cur.fetchall():
                aid, alabel, cid, ctitle = r
                nodes_dict[f'case-{cid}'] = _make_node('case', cid, ctitle)
                edges_list.append(_make_edge(f'case-{cid}', f'article-{aid}', 'CITES'))

    elif center_type == 'project':
        pid = center_id
        nodes_dict[f'project-{pid}'] = _make_node('project', pid, f'项目 #{pid}')

        if depth >= 1:
            cur.execute("""
                SELECT ac.id, ac.title FROM audit_cases ac
                JOIN audit_runs ar ON ar.id = ac.source_run_id
                WHERE ar.project_id = %s LIMIT 20
            """, (pid,))
            for r in cur.fetchall():
                cid, ctitle = r
                nodes_dict[f'case-{cid}'] = _make_node('case', cid, ctitle)
                edges_list.append(_make_edge(f'project-{pid}', f'case-{cid}', 'HAS_FINDING'))
