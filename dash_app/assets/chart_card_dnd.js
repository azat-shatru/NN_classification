/* Drag-and-drop reordering for .cp2c-chart-card elements inside #cp2c-cards-container */
(function () {
    var _dragging = null;

    function nearestCard(el) {
        return el && el.closest ? el.closest('.cp2c-chart-card') : null;
    }

    document.addEventListener('dragstart', function (e) {
        var card = nearestCard(e.target);
        if (!card) return;
        _dragging = card;
        setTimeout(function () { if (_dragging) _dragging.style.opacity = '0.45'; }, 0);
        e.dataTransfer.effectAllowed = 'move';
    }, true);

    document.addEventListener('dragend', function () {
        if (_dragging) { _dragging.style.opacity = '1'; }
        _dragging = null;
        /* Hint the user to click Save Order */
        var btn = document.getElementById('cp2c-save-card-order-btn');
        if (btn) {
            btn.style.outline = '2px solid #6366f1';
            setTimeout(function () { btn.style.outline = ''; }, 1000);
        }
    }, true);

    document.addEventListener('dragover', function (e) {
        var card = nearestCard(e.target);
        if (!card || !_dragging || card === _dragging) return;
        var container = document.getElementById('cp2c-cards-container');
        if (!container) return;
        if (card.parentNode !== container) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        var siblings = Array.from(container.querySelectorAll('.cp2c-chart-card'));
        var fromIdx  = siblings.indexOf(_dragging);
        var toIdx    = siblings.indexOf(card);
        if (fromIdx < 0 || toIdx < 0) return;

        /* Use mouse x-position to decide insert-before vs insert-after within the row */
        var rect    = card.getBoundingClientRect();
        var midX    = rect.left + rect.width / 2;
        var insertBeforeTarget = e.clientX < midX;

        if (insertBeforeTarget) {
            container.insertBefore(_dragging, card);
        } else {
            container.insertBefore(_dragging, card.nextSibling);
        }
    }, true);
})();
