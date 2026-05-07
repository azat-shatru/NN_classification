/* Drag-and-drop reordering for .slide-order-row elements in the Charts Portal. */
(function () {
    var _dragging = null;

    function nearestRow(el) {
        return el && el.closest ? el.closest('.slide-order-row') : null;
    }

    document.addEventListener('dragstart', function (e) {
        var row = nearestRow(e.target);
        if (!row) return;
        _dragging = row;
        /* opacity change deferred so the drag image captures the original look */
        setTimeout(function () { if (_dragging) _dragging.style.opacity = '0.4'; }, 0);
        e.dataTransfer.effectAllowed = 'move';
    }, true);

    document.addEventListener('dragend', function () {
        if (_dragging) { _dragging.style.opacity = '1'; }
        _dragging = null;
        /* Flash the container border to hint the user should click Confirm */
        var container = document.getElementById('cp2c-slide-rows');
        if (container) {
            container.style.outline = '2px solid #6366f1';
            setTimeout(function () { container.style.outline = ''; }, 800);
        }
    }, true);

    document.addEventListener('dragover', function (e) {
        var row = nearestRow(e.target);
        if (!row || !_dragging || row === _dragging) return;
        var container = _dragging.parentNode;
        if (!container || row.parentNode !== container) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        var siblings = Array.from(container.querySelectorAll('.slide-order-row'));
        var fromIdx  = siblings.indexOf(_dragging);
        var toIdx    = siblings.indexOf(row);
        if (fromIdx < 0 || toIdx < 0) return;
        if (fromIdx < toIdx) {
            container.insertBefore(_dragging, row.nextSibling);
        } else {
            container.insertBefore(_dragging, row);
        }
    }, true);
})();
