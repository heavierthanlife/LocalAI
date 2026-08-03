"""Spider-web knowledge graph API endpoints."""
from flask import Blueprint, request, session

from app.services.graph_protocol import GraphNode, GraphEdge, to_graph_response
from app.utils.helpers import ok, err
from app.database import get_db_connection

graph_bp = Blueprint('graph', __name__, url_prefix='/api/graph')


@graph_bp.route('/collusion')
def collusion_graph():
    project_id = request.args.get('project_id', '').strip()
    threshold = float(request.args.get('threshold', 0.5))
    if not project_id:
        return err('project_id is required', 'MISSING_PARAM', 400)

    task_ids = _get_project_task_ids(project_id)
    if not task_ids:
        return err('No batch tasks found for this project', 'NOT_FOUND', 404)

    nodes = {}
    edges = {}

    _merge_entity_relationships(task_ids, nodes, edges)
    _merge_quote_anomalies(task_ids, nodes, edges)
    _merge_text_similarity(task_ids, nodes, edges, threshold)
    _merge_typo_cross(task_ids, nodes, edges)

    return ok(to_graph_response(
        list(nodes.values()), list(edges.values()), 'collusion'
    ))


@graph_bp.route('/compliance')
def compliance_graph():
    project_id = request.args.get('project_id', '').strip()
    if not project_id:
        return err('project_id is required', 'MISSING_PARAM', 400)

    task_ids = _get_project_task_ids(project_id)
    nodes = {}
    edges = {}

    _merge_audit_findings(task_ids, nodes, edges)
    _merge_template_links(task_ids, nodes, edges)
    _merge_law_citations(task_ids, nodes, edges)

    return ok(to_graph_response(
        list(nodes.values()), list(edges.values()), 'compliance'
    ))


@graph_bp.route('/law-impact')
def law_impact_graph():
    law_id = request.args.get('law_id', '').strip()
    depth = int(request.args.get('depth', 3))
    if not law_id:
        return err('law_id is required', 'MISSING_PARAM', 400)

    nodes = {}
    edges = {}
    try:
        from app.services.law_monitor import compute_impact
        impact = compute_impact(law_id)
        _build_law_impact_graph(law_id, impact, depth, nodes, edges, current_depth=0)
    except ImportError:
        return err('Law monitor not available', 'NOT_AVAILABLE', 503)
    except Exception as e:
        return err(str(e)[:200], 'INTERNAL_ERROR', 500)

    return ok(to_graph_response(
        list(nodes.values()), list(edges.values()), 'law_impact'
    ))


@graph_bp.route('/citation')
def citation_graph():
    root_path = request.args.get('path', '').strip()
    project_id = request.args.get('project_id', type=int)
    depth = int(request.args.get('depth', 2))

    if project_id and not root_path:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name, description FROM projects WHERE id = %s", (project_id,))
                    row = cur.fetchone()
                    if row:
                        root_path = (row[0] or '').strip()
        except Exception:
            pass
        if not root_path:
            return ok(to_graph_response([], [], 'citation', {'note': 'no_project_data'}))

    if not root_path:
        return err('path or project_id parameter is required', 'MISSING_PARAM', 400)

    try:
        from app.services.wiki_entity_service import get_entity_graph
        result = get_entity_graph(root_path, depth=depth)
        return ok(result)
    except ImportError:
        return err('Entity graph service not available', 'NOT_AVAILABLE', 503)
    except Exception as e:
        return err(str(e)[:200], 'INTERNAL_ERROR', 500)


@graph_bp.route('/types')
def graph_types():
    return ok({
        'types': [
            {'id': 'collusion', 'label': '围串标蛛网', 'description': 'Bidder collusion signals'},
            {'id': 'compliance', 'label': '违规模式蛛网', 'description': 'Compliance finding recurrence'},
            {'id': 'law_impact', 'label': '法规变更蛛网', 'description': 'Law impact propagation'},
            {'id': 'citation', 'label': '文档引用蛛网', 'description': 'Document citation web'},
        ]
    })


# ── Helper functions ──

def _get_project_task_ids(project_id):
    task_ids = set()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for table in ['batch_comparison_results', 'quote_anomaly_results', 'entity_relationships']:
                try:
                    cur.execute(
                        f"SELECT task_id FROM {table} WHERE project_id = %s AND task_id IS NOT NULL",
                        (project_id,))
                    task_ids.update(r['task_id'] for r in cur.fetchall())
                except Exception:
                    pass

            try:
                cur.execute("""
                    SELECT t.task_id FROM typo_detection_results t
                    INNER JOIN audit_runs ar ON ar.task_id = t.task_id
                    WHERE ar.project_id = %s AND t.task_id IS NOT NULL
                """, (project_id,))
                task_ids.update(r['task_id'] for r in cur.fetchall())
            except Exception:
                pass

            if not task_ids:
                cur.execute("SELECT user_id FROM projects WHERE id = %s", (project_id,))
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "SELECT task_id FROM batch_comparison_results "
                        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 50",
                        (row[0],))
                    task_ids.update(r['task_id'] for r in cur.fetchall())

    return list(task_ids)


def _merge_entity_relationships(task_ids, nodes, edges):
    if not task_ids:
        return
    placeholders = ','.join(['%s'] * len(task_ids))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT source_entity, target_entity, relation_type, confidence, risk_flag, doc_name "
                f"FROM entity_relationships WHERE task_id IN ({placeholders})",
                task_ids,
            )
            for row in cur.fetchall():
                src, tgt, rel_type, conf, risk_flag, doc_name = (
                    row['source_entity'], row['target_entity'],
                    row['relation_type'], row['confidence'],
                    row['risk_flag'], row.get('doc_name', ''),
                )
                se = src.strip()
                te = tgt.strip()
                if not se or not te:
                    continue

                for name in [se, te]:
                    ntype = 'company' if '公司' in name or '集团' in name or '有限' in name else 'person'
                    nid = f'{ntype}-{name}'
                    if nid not in nodes:
                        nodes[nid] = GraphNode(
                            id=nid, label=name, type=ntype,
                            metadata={'risk_flag': risk_flag} if risk_flag else {},
                        )

                eid = f'{src}-{tgt}-{rel_type}'
                weight = float(conf) if conf else 0.5
                edges[eid] = GraphEdge(
                    source=src, target=tgt, label=rel_type, weight=weight,
                )

                if doc_name and doc_name.strip():
                    dn = doc_name.strip()
                    did = f'file-{dn}'
                    if did not in nodes:
                        nodes[did] = GraphNode(id=did, label=dn, type='file')
                    edges[f'doc-{dn}-{src}'] = GraphEdge(
                        source=dn, target=src, label='mentions', weight=0.3,
                    )


def _merge_quote_anomalies(task_ids, nodes, edges):
    if not task_ids:
        return
    placeholders = ','.join(['%s'] * len(task_ids))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT doc_name, risk_score, cross_same_rate, cross_clustering, max_cross_risk, same_rate_flag, clustering_flag "
                f"FROM quote_anomaly_results WHERE task_id IN ({placeholders}) "
                f"AND (cross_same_rate = TRUE OR cross_clustering = TRUE)",
                task_ids,
            )
            rows = cur.fetchall()

    bidder_names = list({r['doc_name'] for r in rows})
    for name in bidder_names:
        nid = f'bidder-{name}'
        if nid not in nodes:
            nodes[nid] = GraphNode(id=nid, label=name, type='company')

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a['cross_same_rate'] or b['cross_same_rate'] or a['cross_clustering'] or b['cross_clustering']:
                weight = max(
                    float(a.get('max_cross_risk', 0)) / 100,
                    float(b.get('max_cross_risk', 0)) / 100,
                    0.3,
                )
                eid = f"{a['doc_name']}-{b['doc_name']}-price_anomaly"
                edges[eid] = GraphEdge(
                    source=a['doc_name'], target=b['doc_name'],
                    label='price_anomaly', weight=min(weight, 1.0),
                )


def _merge_text_similarity(task_ids, nodes, edges, threshold):
    if not task_ids:
        return
    placeholders = ','.join(['%s'] * len(task_ids))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT task_id, file_a, file_b, similarity, max_risk "
                f"FROM batch_pair_results WHERE task_id IN ({placeholders}) "
                f"AND similarity >= %s",
                task_ids + [threshold],
            )
            for row in cur.fetchall():
                fa, fb, sim, risk = row['file_a'], row['file_b'], row['similarity'], row['max_risk']

                for fn in [fa, fb]:
                    nid = f'file-{fn}'
                    if nid not in nodes:
                        nodes[nid] = GraphNode(id=nid, label=fn, type='document')

                eid = f'{fa}-{fb}-text_similar'
                weight = max(float(sim) / 100, 0.1) if sim else 0.5
                edges[eid] = GraphEdge(
                    source=fa, target=fb, label='text_similar', weight=min(weight, 1.0),
                )


def _merge_typo_cross(task_ids, nodes, edges):
    if not task_ids:
        return
    placeholders = ','.join(['%s'] * len(task_ids))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT doc_name, suspect_text FROM typo_detection_results "
                f"WHERE task_id IN ({placeholders})",
                task_ids,
            )
            rows = cur.fetchall()

    doc_typos = {}
    for row in rows:
        dn = row['doc_name']
        doc_typos.setdefault(dn, []).append(row['suspect_text'])

    doc_names = list(doc_typos.keys())
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            a_set = set(doc_typos[doc_names[i]])
            b_set = set(doc_typos[doc_names[j]])
            common = a_set & b_set
            if common:
                nid_a = f'file-{doc_names[i]}'
                nid_b = f'file-{doc_names[j]}'
                if nid_a not in nodes:
                    nodes[nid_a] = GraphNode(id=nid_a, label=doc_names[i], type='document')
                if nid_b not in nodes:
                    nodes[nid_b] = GraphNode(id=nid_b, label=doc_names[j], type='document')

                eid = f'{doc_names[i]}-{doc_names[j]}-typo_cross'
                weight = min(len(common) * 0.1, 1.0)
                edges[eid] = GraphEdge(
                    source=doc_names[i], target=doc_names[j],
                    label='collusion_signal', weight=weight,
                )


def _merge_audit_findings(task_ids, nodes, edges):
    if not task_ids:
        return
    placeholders = ','.join(['%s'] * len(task_ids))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT task_id, check_file_name, summary_json "
                f"FROM audit_file_results WHERE task_id IN ({placeholders})",
                task_ids,
            )
            for row in cur.fetchall():
                fn = row['check_file_name']
                nid = f'file-{fn}'
                if nid not in nodes:
                    nodes[nid] = GraphNode(id=nid, label=fn, type='document')

                import json
                summary = row.get('summary_json') or {}
                if isinstance(summary, str):
                    try:
                        summary = json.loads(summary)
                    except (json.JSONDecodeError, TypeError):
                        summary = {}

                findings = summary.get('findings', []) if isinstance(summary, dict) else []
                for f in findings:
                    if isinstance(f, dict):
                        fid = f.get('id', f.get('rule', str(hash(str(f)))[:8]))
                        fname = f.get('name', f.get('title', fid))
                        f_desc = f.get('description', '')
                        f_severity = f.get('severity', 'info')
                    else:
                        fid = str(hash(str(f)))[:8]
                        fname = str(f)[:60]
                        f_desc = ''
                        f_severity = 'info'

                    fnid = f'finding-{fid}'
                    if fnid not in nodes:
                        nodes[fnid] = GraphNode(
                            id=fnid, label=fname, type='finding',
                            metadata={'severity': f_severity, 'desc': f_desc[:200]},
                        )
                    eid = f'{fnid}-{fn}-found_in'
                    edges[eid] = GraphEdge(
                        source=fnid, target=fn, label='found_in', weight=0.6,
                    )


def _merge_template_links(task_ids, nodes, edges):
    if not task_ids:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT template_id, case_id FROM case_template_links "
                "WHERE audit_task_id = ANY(%s) OR audit_task_id IN "
                "(SELECT unnest(%s::text[]))",
                (task_ids, task_ids),
            )
            for row in cur.fetchall():
                tid = f'template-{row["template_id"]}'
                cid = f'case-{row["case_id"]}'
                if tid not in nodes:
                    nodes[tid] = GraphNode(id=tid, label=f'Template #{row["template_id"]}', type='template')
                if cid not in nodes:
                    nodes[cid] = GraphNode(id=cid, label=f'Case #{row["case_id"]}', type='finding')
                eid = f'{tid}-{cid}-violated_by'
                edges[eid] = GraphEdge(source=tid, target=cid, label='violated_by', weight=0.5)


def _merge_law_citations(task_ids, nodes, edges):
    if not task_ids:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT law_id, case_id FROM case_law_links "
                "WHERE audit_task_id = ANY(%s)",
                (task_ids,),
            )
            for row in cur.fetchall():
                lid = f'law-{row["law_id"]}'
                cid = f'case-{row["case_id"]}'
                if lid not in nodes:
                    nodes[lid] = GraphNode(id=lid, label=str(row['law_id']), type='law')
                if cid not in nodes:
                    nodes[cid] = GraphNode(id=cid, label=f'Case #{row["case_id"]}', type='finding')
                eid = f'{lid}-{cid}-cites'
                edges[eid] = GraphEdge(source=lid, target=cid, label='cites', weight=0.4)


def _build_law_impact_graph(law_id, impact, max_depth, nodes, edges, current_depth):
    if current_depth >= max_depth:
        return

    lid = f'law-{law_id}'
    if lid not in nodes:
        nodes[lid] = GraphNode(id=lid, label=str(law_id), type='law')

    if isinstance(impact, dict):
        affected_laws = impact.get('affected_laws') or impact.get('laws') or []
        affected_cases = impact.get('affected_cases') or impact.get('cases') or []
        affected_templates = impact.get('affected_templates') or impact.get('templates') or []

        for al in affected_laws:
            child_law_id = al if isinstance(al, str) else al.get('id', str(al))
            eid = f'{law_id}-{child_law_id}-impacted_by'
            edges[eid] = GraphEdge(source=law_id, target=child_law_id, label='impacted_by', weight=0.7)
            _build_law_impact_graph(child_law_id, {}, max_depth, nodes, edges, current_depth + 1)

        for ac in affected_cases:
            case_id = ac if isinstance(ac, str) else ac.get('id', str(ac))
            eid = f'{law_id}-{case_id}-cites'
            edges[eid] = GraphEdge(source=law_id, target=case_id, label='cites', weight=0.5)
            nid = f'case-{case_id}'
            if nid not in nodes:
                nodes[nid] = GraphNode(id=nid, label=str(case_id), type='finding')

        for at_item in affected_templates:
            tid = at_item if isinstance(at_item, str) else at_item.get('id', str(at_item))
            eid = f'{law_id}-{tid}-violated_by'
            edges[eid] = GraphEdge(source=law_id, target=tid, label='violated_by', weight=0.4)
            nid = f'template-{tid}'
            if nid not in nodes:
                nodes[nid] = GraphNode(id=nid, label=str(tid), type='template')
