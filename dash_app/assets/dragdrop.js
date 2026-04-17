/* Drag-and-drop for Variable Mapping option chips.
   Uses document-level event delegation so listeners survive Dash re-renders. */

(function () {
    'use strict';

    var dragged = null;

    // Prevent Bootstrap accordion toggle when clicking the question-delete button.
    //
    // Bubble order: target → … → React root → document.body → document → window
    // React (Dash) listens on the React root element, Bootstrap listens on document.
    // By stopping propagation on document.body (bubble phase), we block Bootstrap's
    // document-level handler while letting React already-see the event.
    document.body.addEventListener('click', function (e) {
        if (e.target.closest('.vm-del-q-btn')) {
            e.stopPropagation();
        }
    }, false);

    document.addEventListener('dragstart', function (e) {
        // Don't drag when clicking the chip-delete or question-delete buttons
        if (e.target.closest('.opt-chip-del') || e.target.closest('.vm-del-q-btn')) {
            e.preventDefault();
            return;
        }
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


    // ── Double-click chip text to edit inline ─────────────────────────────────

    document.addEventListener('dblclick', function (e) {
        var chip = e.target.closest('.opt-chip');
        if (!chip) return;
        // Don't activate if clicking the × button
        if (e.target.closest('.opt-chip-del')) return;
        // Already editing?
        if (chip.querySelector('.opt-chip-input')) return;

        var currentText = chip.getAttribute('data-opt') || '';
        var delBtn = chip.querySelector('.opt-chip-del');

        // Replace chip text node with an input
        var input = document.createElement('input');
        input.type = 'text';
        input.value = currentText;
        input.className = 'opt-chip-input';

        // Remove the text node(s) from the chip, keep the del button
        Array.from(chip.childNodes).forEach(function (n) {
            if (n !== delBtn) chip.removeChild(n);
        });
        chip.insertBefore(input, delBtn);
        input.focus();
        input.select();
        chip.draggable = false;

        function commit() {
            var newText = input.value.trim();
            if (!newText) newText = currentText;
            chip.removeChild(input);
            chip.insertBefore(document.createTextNode(newText), delBtn);
            chip.setAttribute('data-opt', newText);
            // Update the del button's id so the clientside delete still works
            if (delBtn && delBtn.id) {
                try {
                    var idObj = JSON.parse(delBtn.id);
                    var sep = idObj.index.indexOf('||');
                    if (sep !== -1) {
                        idObj.index = idObj.index.substring(0, sep + 2) + newText;
                        delBtn.id = JSON.stringify(idObj);
                    }
                } catch (ex) {}
            }
            chip.draggable = true;
        }

        input.addEventListener('keydown', function (ev) {
            if (ev.key === 'Enter')  { ev.preventDefault(); commit(); }
            if (ev.key === 'Escape') { ev.preventDefault();
                chip.removeChild(input);
                chip.insertBefore(document.createTextNode(currentText), delBtn);
                chip.draggable = true;
            }
        });
        input.addEventListener('blur', commit);
    }, false);


    // ── "Add options" button: toggle paste panel ──────────────────────────────

    document.addEventListener('click', function (e) {
        var btn = e.target.closest('.vm-add-opts-btn');
        if (!btn) return;
        var qcode = btn.getAttribute('data-qcode');
        var panel = document.querySelector('.vm-paste-row[data-qcode="' + qcode + '"]');
        if (!panel) return;
        var hidden = panel.style.display === 'none' || panel.style.display === '';
        panel.style.display = hidden ? 'table-row' : 'none';
        if (hidden) {
            var ta = panel.querySelector('.vm-opts-textarea');
            if (ta) { ta.value = ''; ta.focus(); }
        }
    }, false);


    // ── "Add to pool" button: parse textarea and create chips ─────────────────

    function _makeChip(text, qcode) {
        var span = document.createElement('span');
        span.className = 'opt-chip opt-chip-pool';
        span.draggable = true;
        span.setAttribute('data-opt', text);
        span.appendChild(document.createTextNode(text));

        // Minimal del button (no Dash id — purely DOM-managed custom chip)
        var del = document.createElement('button');
        del.className = 'opt-chip-del';
        del.textContent = '×';
        del.addEventListener('click', function (e) {
            e.stopPropagation();
            span.remove();
        });
        span.appendChild(del);
        return span;
    }

    document.addEventListener('click', function (e) {
        var btn = e.target.closest('.vm-opts-add-btn');
        if (!btn) return;
        var qcode = btn.getAttribute('data-qcode');
        var panel = btn.closest('.vm-paste-row');
        if (!panel) return;
        var ta = panel.querySelector('.vm-opts-textarea');
        if (!ta) return;

        var lines = ta.value.split('\n')
            .map(function (l) { return l.trim(); })
            .filter(function (l) { return l.length > 0; });

        if (lines.length === 0) return;

        // Find the pool drop-zone in the same table
        var table = panel.closest('table');
        if (!table) return;
        var pool = table.querySelector('.opt-dropzone[data-col="__pool__"]');
        if (!pool) return;

        lines.forEach(function (text) {
            pool.appendChild(_makeChip(text, qcode));
        });

        ta.value = '';
        panel.style.display = 'none';
    }, false);

}());
