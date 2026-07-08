(function() {
    var canvas = document.getElementById('digitCanvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var clearBtn = document.getElementById('clearBtn');
    var checkBtn = document.getElementById('checkBtn');
    var nextBtn = document.getElementById('nextBtn');
    var retryBtn = document.getElementById('retryBtn');
    var targetDigitEl = document.getElementById('targetDigit');
    var feedbackPlaceholder = document.getElementById('feedbackPlaceholder');
    var feedbackResult = document.getElementById('feedbackResult');
    var feedbackMessage = document.getElementById('feedbackMessage');
    var feedbackIcon = document.getElementById('feedbackIcon');
    var writtenDigit = document.getElementById('writtenDigit');
    var writtenConfidence = document.getElementById('writtenConfidence');

    var drawing = false;
    var hasDrawn = false;
    var currentTarget = 3;
    var isChecking = false;

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    function getPos(e) {
        var rect = canvas.getBoundingClientRect();
        var scaleX = canvas.width / rect.width;
        var scaleY = canvas.height / rect.height;
        var clientX = e.touches ? e.touches[0].clientX : e.clientX;
        var clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
    }

    function startDraw(e) {
        e.preventDefault();
        if (isChecking) return;
        drawing = true;
        hasDrawn = true;
        var pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        checkBtn.disabled = true;
        feedbackPlaceholder.style.display = 'flex';
        feedbackResult.style.display = 'none';
    }

    function draw(e) {
        e.preventDefault();
        if (!drawing) return;
        var pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function stopDraw(e) {
        e.preventDefault();
        drawing = false;
        ctx.beginPath();
        if (hasDrawn && !isChecking) {
            checkBtn.disabled = false;
        }
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDraw);

    function clearCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        checkBtn.disabled = true;
        feedbackPlaceholder.style.display = 'flex';
        feedbackResult.style.display = 'none';
    }

    clearBtn.addEventListener('click', clearCanvas);
    clearCanvas();

    function loadTarget() {
        fetch('/next_target')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                currentTarget = data.target;
                targetDigitEl.textContent = currentTarget;
                targetDigitEl.className = 'target-digit';
                clearCanvas();
                loadStats();
            });
    }

    function loadStats() {
        fetch('/stats')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                for (var d = 0; d < 10; d++) {
                    var s = data.per_digit[d];
                    var pctEl = document.getElementById('pct-' + d);
                    var ringEl = document.getElementById('ring-' + d);
                    var cell = document.querySelector('.progress-cell[data-digit="' + d + '"]');
                    if (s.total === 0) {
                        pctEl.textContent = '--';
                        ringEl.style.strokeDashoffset = '125.6';
                        cell.className = 'progress-cell progress-none';
                    } else {
                        pctEl.textContent = s.accuracy + '%';
                        var offset = 125.6 - (s.accuracy / 100) * 125.6;
                        ringEl.style.strokeDashoffset = offset;
                        if (s.accuracy >= 80) {
                            cell.className = 'progress-cell progress-good';
                        } else if (s.accuracy >= 50) {
                            cell.className = 'progress-cell progress-ok';
                        } else {
                            cell.className = 'progress-cell progress-bad';
                        }
                    }
                }
            });
    }

    function checkWriting() {
        if (checkBtn.disabled || isChecking) return;
        isChecking = true;
        checkBtn.disabled = true;
        checkBtn.textContent = 'Checking...';

        var imageData = canvas.toDataURL('image/png');

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, targetDigit: currentTarget })
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            feedbackPlaceholder.style.display = 'none';
            feedbackResult.style.display = 'block';
            feedbackMessage.textContent = data.message || 'Check result';
            writtenDigit.textContent = data.prediction || '?';
            writtenConfidence.textContent = (data.confidence || 0) + '%';

            if (data.correct) {
                feedbackIcon.innerHTML = '<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12L11 15L16 9" stroke-linecap="round" stroke-linejoin="round"/></svg>';
                feedbackResult.className = 'feedback-result feedback-correct';
                targetDigitEl.className = 'target-digit target-correct';
                nextBtn.style.display = 'inline-block';
                retryBtn.style.display = 'none';
            } else {
                feedbackIcon.innerHTML = '<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#f85149" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15.5 9.5L9.5 15.5M9.5 9.5L15.5 15.5" stroke-linecap="round"/></svg>';
                feedbackResult.className = 'feedback-result feedback-wrong';
                targetDigitEl.className = 'target-digit target-wrong';
                nextBtn.style.display = 'none';
                retryBtn.style.display = 'inline-block';
            }

            loadStats();
        })
        .catch(function(err) {
            feedbackPlaceholder.style.display = 'none';
            feedbackResult.style.display = 'block';
            feedbackMessage.textContent = 'Could not reach the server. Check your connection.';
            feedbackIcon.innerHTML = '<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#d29922" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01" stroke-linecap="round"/></svg>';
            feedbackResult.className = 'feedback-result feedback-wrong';
            writtenDigit.textContent = '?';
            writtenConfidence.textContent = '0%';
            nextBtn.style.display = 'none';
            retryBtn.style.display = 'inline-block';
        })
        .finally(function() {
            isChecking = false;
            checkBtn.textContent = 'Check My Writing';
        });
    }

    checkBtn.addEventListener('click', checkWriting);

    nextBtn.addEventListener('click', function() {
        loadTarget();
    });

    retryBtn.addEventListener('click', function() {
        clearCanvas();
        feedbackResult.style.display = 'none';
        feedbackPlaceholder.style.display = 'flex';
        targetDigitEl.className = 'target-digit';
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !checkBtn.disabled && !isChecking) {
            checkWriting();
        }
        if ((e.key === 'c' || e.key === 'C') && !isChecking) {
            retryBtn.click();
        }
    });

    loadTarget();
})();
