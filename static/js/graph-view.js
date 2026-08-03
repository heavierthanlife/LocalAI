/**
 * graph-view.js — Unified Cytoscape.js renderer for all spider-web types.
 * Depends on: cytoscape (global), loaded via CDN before this script.
 */
(function () {
    'use strict';

    const SHAPES_BY_TYPE = {
        'company': 'rectangle',
        'person': 'ellipse',
        'document': 'round-rectangle',
        'file': 'round-rectangle',
        'finding': 'diamond',
        'law': 'hexagon',
        'template': 'triangle',
        'project': 'barrel',
        'entity': 'tag',
    };

    function _buildStyle(opts) {
        const style = [
            { selector: 'node', css: {
                'label': 'data(label)', 'text-valign': 'center',
                'text-halign': 'center', 'font-size': '8px',
                'background-color': 'data(color)', 'color': '#fff',
                'text-wrap': 'wrap', 'text-max-width': '100px',
                'width': '30px', 'height': '30px',
                'border-width': 2, 'border-color': '#fff',
            }},
            { selector: 'edge', css: {
                'width': 'mapData(weight, 0, 1, 1, 4)',
                'line-color': 'data(color)',
                'target-arrow-color': 'data(color)',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier', 'opacity': 0.6,
            }},
            { selector: 'edge[label]', css: { 'font-size': '6px', 'color': '#95a5a6' }},
            { selector: ':selected', css: {
                'border-color': '#f39c12', 'border-width': 3,
            }},
        ];

        if (opts && opts.shapes) {
            for (var type in opts.shapes) {
                if (opts.shapes.hasOwnProperty(type)) {
                    style.push({
                        selector: 'node[type="' + type + '"]',
                        css: { 'shape': opts.shapes[type] },
                    });
                }
            }
        } else {
            for (var nt in SHAPES_BY_TYPE) {
                if (SHAPES_BY_TYPE.hasOwnProperty(nt)) {
                    style.push({
                        selector: 'node[type="' + nt + '"]',
                        css: { 'shape': SHAPES_BY_TYPE[nt] },
                    });
                }
            }
        }

        return style;
    }

    function _buildElements(graphData) {
        var elements = [];
        if (graphData.nodes) {
            graphData.nodes.forEach(function (n) {
                elements.push({
                    data: {
                        id: n.id, label: n.label,
                        type: n.type, color: n.color || '#95a5a6',
                        metadata: n,
                    },
                });
            });
        }
        if (graphData.edges) {
            graphData.edges.forEach(function (e) {
                elements.push({
                    data: {
                        id: e.id, source: e.source, target: e.target,
                        label: e.label, weight: e.weight || 0.5,
                        color: e.color || '#95a5a6',
                    },
                });
            });
        }
        return elements;
    }

    function _showNodeDetailModal(meta) {
        var overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.3);z-index:10001;display:flex;align-items:center;justify-content:center;';
        overlay.onclick = function(e) { if (e.target === overlay) { overlay.remove(); } };

        var card = document.createElement('div');
        card.style.cssText = 'background:var(--card-bg,#fff);border-radius:10px;padding:16px 20px;max-width:420px;width:90%;max-height:70vh;overflow-y:auto;box-shadow:0 6px 24px rgba(0,0,0,0.18);';
        var typeLabels = {
            'company': '🏢 投标人', 'person': '👤 人员', 'document': '📄 文档', 'file': '📁 文件',
            'finding': '⚠️ 发现', 'law': '📜 法规', 'template': '📋 模板',
            'project': '📂 项目', 'entity': '🏷️ 实体',
        };

        var typeLabel = meta.type ? (typeLabels[meta.type] || meta.type) : '节点';
        var title = meta.label || meta.id || '未命名节点';

        var html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">';
        html += '<strong style="font-size:0.85rem;">' + typeLabel + '</strong>';
        html += '<button onclick="this.closest(\'div[style*=\\\'fixed\\\']\').remove()" style="background:none;border:none;font-size:1.2rem;cursor:pointer;line-height:1;">&times;</button>';
        html += '</div>';
        html += '<h4 style="margin:0 0 10px;font-size:0.92rem;word-break:break-all;">' + title + '</h4>';

        var fields = ['score', 'risk_score', 'max_risk', 'weight', 'signal_count', 'source_doc',
                      'description', 'category', 'path', 'law_id', 'article_count',
                      'entity_count', 'finding_count', 'node_count', 'edge_count', 'file_names'];

        var shown = [];
        for (var f in meta) {
            if (meta.hasOwnProperty(f) && f !== 'label' && f !== 'id' && f !== 'type' && f !== 'color') {
                var val = meta[f];
                if (val === null || val === undefined || val === '') continue;
                if (typeof val === 'object') val = JSON.stringify(val).substring(0, 200);
                else val = String(val).substring(0, 200);
                if (val.length > 100) val = val.substring(0, 100) + '...';
                html += '<div style="font-size:0.72rem;margin-bottom:4px;">';
                html += '<span style="color:var(--card-muted,#95a5a6);">' + f + '</span>';
                html += '<span style="margin-left:8px;word-break:break-all;">' + val + '</span>';
                html += '</div>';
                shown.push(f);
            }
        }
        if (shown.length === 0) {
            html += '<p style="font-size:0.72rem;color:var(--card-muted,#95a5a6);">暂无详细信息</p>';
        }

        card.innerHTML = html;
        overlay.appendChild(card);
        document.body.appendChild(overlay);

        var closeBtn = card.querySelector('button');
        if (closeBtn) {
            closeBtn.onclick = function() { overlay.remove(); };
        }
    }

    /**
     * Render a spider-web graph into a DOM element.
     *
     * @param {string|Element} container - DOM element or ID
     * @param {object} graphData - { nodes: [...], edges: [...], stats: {...}, web_type: '...' }
     * @param {object} [options]
     * @param {object} [options.shapes] - { type: shapeName } overrides
     * @param {function} [options.onNodeTap] - (nodeData, cyInstance) => void
     * @param {function} [options.onNodeDblTap] - (nodeData, cyInstance) => void
     * @param {string} [options.layout] - 'cose' (default) | 'breadthfirst' | 'concentric'
     * @returns {object|null} Cytoscape instance or null on failure
     */
    window.renderGraph = function (container, graphData, options) {
        if (typeof cytoscape === 'undefined') {
            console.warn('[renderGraph] Cytoscape.js not loaded');
            return null;
        }
        if (!graphData || (!graphData.nodes && !graphData.edges)) {
            console.warn('[renderGraph] No graph data');
            return null;
        }

        options = options || {};
        var el = typeof container === 'string' ? document.getElementById(container) : container;
        if (!el) {
            console.warn('[renderGraph] Container not found:', container);
            return null;
        }

        el.innerHTML = '';
        el.style.position = 'relative';

        var nodeCount = (graphData.nodes || []).length;
        var elements = _buildElements(graphData);
        var layoutName = options.layout || 'cose';

        // ── Search/Filter Toolbar (only when > 30 nodes) ──
        var filterBar = null;
        var _activeTypeFilter = null;
        if (nodeCount > 30) {
            filterBar = document.createElement('div');
            filterBar.style.cssText = 'display:flex;gap:6px;margin-bottom:4px;align-items:center;flex-wrap:wrap;';

            var searchInput = document.createElement('input');
            searchInput.type = 'text';
            searchInput.placeholder = '🔍 搜索节点...';
            searchInput.style.cssText = 'padding:3px 8px;border:1px solid var(--card-border);border-radius:4px;font-size:0.72rem;width:160px;';

            var typeChips = document.createElement('div');
            typeChips.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;';

            // Collect unique types
            var types = {};
            (graphData.nodes || []).forEach(function (n) { if (n.type) types[n.type] = true; });
            var typeList = Object.keys(types);

            typeList.forEach(function (t) {
                var chip = document.createElement('button');
                chip.textContent = t;
                chip.style.cssText = 'padding:1px 6px;border:1px solid var(--card-border);border-radius:10px;font-size:0.6rem;cursor:pointer;background:var(--card-bg);';
                chip.dataset.type = t;
                typeChips.appendChild(chip);
            });

            var clearBtn = document.createElement('button');
            clearBtn.textContent = '清除';
            clearBtn.style.cssText = 'padding:1px 8px;border:1px solid var(--card-border);border-radius:10px;font-size:0.6rem;cursor:pointer;background:var(--card-bg);';

            filterBar.appendChild(searchInput);
            filterBar.appendChild(typeChips);
            filterBar.appendChild(clearBtn);
            el.appendChild(filterBar);
        }

        var graphWrapper = document.createElement('div');
        graphWrapper.style.cssText = 'position:relative;width:100%;' + (filterBar ? 'height:calc(100% - 32px);' : 'height:100%;');
        el.appendChild(graphWrapper);

        var cy = cytoscape({
            container: graphWrapper,
            elements: elements,
            style: _buildStyle(options),
            layout: { name: layoutName, animate: true, fit: true, padding: 20 },
            wheelSensitivity: 0.3,
        });

        // ── Search/filter logic ──
        if (filterBar) {
            function _applyFilter() {
                cy.nodes().style('opacity', 1);
                cy.edges().style('opacity', 0.6);

                var q = searchInput.value.trim().toLowerCase();
                var visibleNodes = cy.nodes();
                var hiddenNodes = cy.collection();

                // Type filter
                if (_activeTypeFilter) {
                    hiddenNodes = visibleNodes.filter(function (n) {
                        return n.data('type') !== _activeTypeFilter;
                    });
                    visibleNodes = visibleNodes.difference(hiddenNodes);
                }

                // Text filter
                if (q) {
                    var textHidden = visibleNodes.filter(function (n) {
                        return (n.data('label') || '').toLowerCase().indexOf(q) === -1;
                    });
                    hiddenNodes = hiddenNodes.union(textHidden);
                    visibleNodes = visibleNodes.difference(textHidden);
                }

                // Dim hidden
                hiddenNodes.style('opacity', 0.08);
                cy.edges().style('opacity', function (e) {
                    return (e.source().style('opacity') < 0.2 && e.target().style('opacity') < 0.2) ? 0.05 : 0.6;
                });
            }

            searchInput.addEventListener('input', _applyFilter);

            typeChips.querySelectorAll('button').forEach(function (chip) {
                chip.addEventListener('click', function () {
                    if (_activeTypeFilter === chip.dataset.type) {
                        _activeTypeFilter = null;
                        chip.style.background = 'var(--card-bg)';
                        chip.style.color = '';
                        chip.style.fontWeight = '';
                    } else {
                        typeChips.querySelectorAll('button').forEach(function (c) {
                            c.style.background = 'var(--card-bg)';
                            c.style.color = '';
                            c.style.fontWeight = '';
                        });
                        _activeTypeFilter = chip.dataset.type;
                        chip.style.background = '#1e293b';
                        chip.style.color = 'white';
                        chip.style.fontWeight = '600';
                    }
                    _applyFilter();
                });
            });

            clearBtn.addEventListener('click', function () {
                searchInput.value = '';
                _activeTypeFilter = null;
                typeChips.querySelectorAll('button').forEach(function (c) {
                    c.style.background = 'var(--card-bg)';
                    c.style.color = '';
                    c.style.fontWeight = '';
                });
                cy.nodes().style('opacity', 1);
                cy.edges().style('opacity', 0.6);
            });
        }

        if (options.onNodeTap) {
            cy.on('tap', 'node', function (evt) {
                var node = evt.target;
                options.onNodeTap(node.data('metadata'), cy);
            });
        }

        if (options.onNodeDblTap) {
            cy.on('dbltap', 'node', function (evt) {
                var node = evt.target;
                options.onNodeDblTap(node.data('metadata'), cy);
            });
        } else {
            cy.on('dbltap', 'node', function (evt) {
                var node = evt.target;
                _showNodeDetailModal(node.data('metadata'));
            });
        }

        if (graphData.stats) {
            var statsEl = document.createElement('div');
            statsEl.style.cssText = 'position:absolute;top:4px;right:8px;font-size:0.65rem;color:#95a5a6;background:rgba(255,255,255,0.85);padding:2px 6px;border-radius:4px;z-index:10;pointer-events:none;';
            statsEl.textContent = graphData.stats.node_count + ' nodes / ' + graphData.stats.edge_count + ' edges';
            graphWrapper.style.position = 'relative';
            graphWrapper.appendChild(statsEl);
        }

        return cy;
    };

})();
