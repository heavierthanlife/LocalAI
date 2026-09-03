/* Material Symbols icon helper — standalone global (FIX-016 后续).

   Loaded BEFORE all business scripts (see templates/index.html) so every JS file
   can call _icon()/_iconMd() regardless of load order. Font CSS in
   static/css/icons.css (self-hosted woff2 in static/fonts/).

   Usage:
     var html = _icon('✅');        // → <span class="msi msi-sm">check_circle</span>
     var html = _icon('description'); // pass a Material Symbols ligature name directly
*/
(function () {
    var MSI = {
        '📥': 'download', '📊': 'bar_chart', '🔀': 'swap_horiz', '⚖️': 'balance',
        '🤖': 'smart_toy', '🖼️': 'image', '🛡️': 'verified_user', '🔴': 'error',
        '⚠️': 'warning', '✅': 'check_circle', '❌': 'cancel', '⏳': 'hourglass_empty',
        '👥': 'groups', '🕸️': 'hub', '📋': 'list_alt', '🔑': 'key', '🔍': 'search',
        '📄': 'description', '📁': 'folder', '📂': 'folder_open', '🗑️': 'delete',
        '🔄': 'refresh', '➕': 'add', '✏️': 'edit', '⚙️': 'settings', '🔧': 'build',
        '📝': 'edit_note', '💬': 'chat', '🧠': 'psychology', '💰': 'payments',
        '📈': 'trending_up', '📉': 'trending_down', '📚': 'menu_book', '📜': 'article',
        '🏢': 'domain', '👤': 'person', '🏗️': 'construction', '📅': 'calendar_month',
        '🔗': 'link', '💾': 'save', '📤': 'upload', '🕐': 'schedule', '⏱️': 'timer',
        '🚫': 'block', '⛔': 'block', '🔥': 'local_fire_department', '💡': 'lightbulb',
        '⭐': 'star', '★': 'star', '📌': 'push_pin', '📍': 'location_on',
        '⏭️': 'skip_next', '⏭': 'skip_next', '⚡': 'bolt', '💼': 'work',
        '🏛️': 'account_balance', '🏷️': 'sell', '🆕': 'new_releases', '🌐': 'public',
        '📖': 'menu_book', '📓': 'note', '📑': 'table_rows', '✉️': 'mail',
        '📨': 'send', '🗺️': 'map', '🧹': 'cleaning_services', '♻️': 'recycling',
        '🔨': 'hammer', '🎛️': 'tune', '🔬': 'science', '🧪': 'science',
        '📎': 'attach_file', '🔔': 'notifications', '✕': 'close', '✖': 'close',
        '☰': 'menu', '👑': 'workspace_premium', '👁️': 'visibility', '🩺': 'monitor_heart',
        '📷': 'photo_camera', '📹': 'videocam', '📱': 'smartphone', '⌨️': 'keyboard',
        '📦': 'inventory_2', '🗂️': 'folder_special', '🗄️': 'database', '🗓️': 'calendar_month',
        '🔑': 'key', '🏁': 'flag', '🎉': 'celebration', '✨': 'auto_awesome',
        '🛠️': 'construction', '⏹': 'stop', '💭': 'chat', '↺': 'restart_alt',
        '⬇️': 'south', '⬇': 'south', '⬆️': 'north', '⬆': 'north',
        '✓': 'check', '√': 'check', '➤': 'chevron_right', '☰': 'menu',
        '📝': 'edit_note', '🚨': 'notifications_active', '🌓': 'dark_mode',
        '📏': 'straighten', '📐': 'architecture',
        '🗑': 'delete', '🗄': 'database', '🗂': 'folder_special', '🖼': 'image',
        '🏷': 'sell', '🏛': 'account_balance', '🎛': 'tune', '✍️': 'edit',
        '✍': 'edit', '👍': 'thumb_up', '👎': 'thumb_down',
        '🟢': 'monitoring'
    };

    function _esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    function _icon(name) {
        var glyph = MSI[name] || name;
        return '<span class="msi msi-sm" aria-hidden="true">' + _esc(glyph) + '</span>';
    }
    function _iconMd(name) {
        var glyph = MSI[name] || name;
        return '<span class="msi msi-md" aria-hidden="true">' + _esc(glyph) + '</span>';
    }

    // FIX-016 后续: 统一折叠箭头工具（M5）。用 Material Symbols expand_more + CSS 旋转
    // 替代 textContent='▶'/'▼'。调用: _toggleArrow(el, isCollapsed)
    function _toggleArrow(el, collapsed) {
        if (!el) return;
        el.innerHTML = '<span class="msi msi-arrow' + (collapsed ? ' collapsed' : '') + '" aria-hidden="true">expand_more</span>';
    }

    // Expose globals for all business scripts (loaded after this file)
    window.MSI = MSI;
    window._icon = _icon;
    window._iconMd = _iconMd;
    window._toggleArrow = _toggleArrow;
})();
