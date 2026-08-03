# UI Upload Fixes — Design Spec

**Date:** 2026-07-06 | **Branch:** master

## Scope

Three independent UI bug fixes in `static/js/app.js` + `static/css/app.css`:
1. File duplicate dialog: batch review panel + "yes to all" + lazy content compare
2. Progress bar: 4 missing `finishProgress` calls + success/error visual differentiation
3. Drag-drop watermark: enter/leave counter to fix stuck overlay

---

## Fix #1: File Duplicate Dialog

### Backend (`app/routes/admin.py`)

- Add name-conflict query alongside existing hash check at line 928:
  ```sql
  SELECT id, original_name, stored_path, version, folder_id, file_size, file_hash
  FROM project_files WHERE project_id = %s AND original_name = %s
  ```
- Return `conflicts` array in response with type tags: `hash_match` vs `name_match`
- New endpoint `POST /admin/projects/<id>/files/compare-content` — takes two file_ids, returns extracted text from both, with progress events via SSE or polling

### Frontend (`static/js/app.js`)

- Replace `showDuplicateFileOptions` (line 6040) single-modal with `showBatchConflictPanel(conflicts)`:
  - Batch header: "N conflicts found" + **"Keep all existing"** + **"Replace all with new"**
  - Per-pair row: metadata columns (name, size, version, date) side-by-side
  - Per-pair actions: "Keep existing" / "Replace" / "Rename" buttons
  - For `name_match` type: **"Compare content"** button → inline progress → diff view
- `uploadFileToFolder` returns conflicts to caller instead of handling inline
- Bulk upload in knowledge lab (line 7065) queues results, invokes panel on completion

### CSS

- Conflict panel: fixed-position overlay, scrollable pair list, side-by-side metadata cards
- Inline diff view: split pane with scroll-sync, additions/deletions highlighting

---

## Fix #2: Never-Ending Progress Bar

### `finishProgress` (line 190)

- Use `success` param: green (`#16a34a`) auto-hide 600ms on true; red (`#ef4444`) on false
- Error state: sticky toast with message + dismiss "我知道了" button, no auto-hide
- Rename: `finishProgress(success, message)` → success=true means green+fade, success=false means red+sticky+dismiss button

### Missing `finishProgress` calls

| # | Location | Fix |
|---|----------|-----|
| 1 | `uploadFileToFolder` duplicate path (line 6000-6035) | `finishProgress(false, '发现重复文件')` after conflict resolution, before reload |
| 2 | `uploadFileToFolder` general failure (line 6037) | `finishProgress(false, data.error \|\| '上传失败')` before alert |
| 3 | `uploadFileToFolder` no try/catch | Wrap fetch in try/catch, `finishProgress(false, '网络错误')` in catch |
| 4 | `watchTaskProgress` SSE hang (line 10757) | 30s idle timeout: `setTimeout` reset on each `onmessage`; on fire → `es.close()` + `finishProgress(false, '连接超时')` |

---

## Fix #3: Drag-Drop Watermark

### `static/js/app.js` (line 8790-8817)

Replace body-level class toggle with counter:

```js
let _dragCounter = 0;
['dragenter', 'dragover', 'dragleave', 'drop', 'dragend'].forEach(evt => {
    document.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
});
document.addEventListener('dragenter', () => {
    _dragCounter++;
    chatInterface.classList.add('drag-over');
});
document.addEventListener('dragleave', () => {
    _dragCounter--;
    if (_dragCounter <= 0) { _dragCounter = 0; chatInterface.classList.remove('drag-over'); }
});
document.addEventListener('drop', () => {
    _dragCounter = 0;
    chatInterface.classList.remove('drag-over');
});
document.addEventListener('dragend', () => {
    _dragCounter = 0;
    chatInterface.classList.remove('drag-over');
});
```

No CSS changes needed — the existing `.drag-over::after` works correctly once the class toggles reliably.

---

## Self-Review

- No TBDs or placeholders
- Fix #1 backend change is minimal (1 new query, 1 new endpoint); frontend is the bulk
- Fix #2 changes are surgical — 4 explicit calls, 1 function behavior change
- Fix #3 is a 15-line rewrite of the drag listeners
- All changes confined to `static/js/app.js`, `static/css/app.css`, `app/routes/admin.py`
