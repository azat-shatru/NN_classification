/* Drag-and-drop for Variable Mapping option chips.
   Uses document-level event delegation so listeners survive Dash re-renders. */

(function () {
    'use strict';

    var dragged = null;

    document.addEventListener('dragstart', function (e) {
        var chip = e.target.closest('.opt-chip');
        if (!chip) return;
        dragged = chip;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', chip.getAttribute('data-opt') || '');
        /* Defer class so browser can take the drag snapshot first */
        requestAnimationFrame(function () {
            if (dragged) dragged.classList.add('opt-chip--dragging');
        });
    }, false);

    document.addEventListener('dragend', function () {
        if (dragged) {
            dragged.classList.remove('opt-chip--dragging');
            dragged = null;
        }
        document.querySelectorAll('.opt-dropzone--over').forEach(function (z) {
            z.classList.remove('opt-dropzone--over');
        });
    }, false);

    document.addEventListener('dragover', function (e) {
        var zone = e.target.closest('.opt-dropzone');
        if (!zone) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        document.querySelectorAll('.opt-dropzone--over').forEach(function (z) {
            if (z !== zone) z.classList.remove('opt-dropzone--over');
        });
        zone.classList.add('opt-dropzone--over');
    }, false);

    document.addEventListener('dragleave', function (e) {
        var zone = e.target.closest('.opt-dropzone');
        if (!zone) return;
        if (!zone.contains(e.relatedTarget)) {
            zone.classList.remove('opt-dropzone--over');
        }
    }, false);

    document.addEventListener('drop', function (e) {
        var zone = e.target.closest('.opt-dropzone');
        if (!zone || !dragged) return;
        e.preventDefault();
        zone.classList.remove('opt-dropzone--over');
        zone.appendChild(dragged);
        dragged.classList.remove('opt-chip--dragging');
    }, false);
}());
