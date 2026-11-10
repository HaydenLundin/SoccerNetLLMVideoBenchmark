/**
 * SoccerNet VLM Interactive Demonstration Application
 * Synchronized multimodal video playback, canvas waveform rendering,
 * sliding window telemetry, and Chain-of-Thought inspection.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    matches: [],
    models: {},
    currentMatchId: 0,
    currentHalf: 1,
    currentChunkIndex: 0,
    currentModelKey: 'mece_m15',
    chunkData: null,
    scorecardData: null,
    highlights: [],
    isPlaying: false,
    playbackSpeed: 1.0,
    currentFilter: 'all'
  };

  // DOM Elements
  const videoPlayer = document.getElementById('video-player');
  const matchSelect = document.getElementById('match-select');
  const halfSelect = document.getElementById('half-select');
  const chunkSelect = document.getElementById('chunk-select');
  const modelSelect = document.getElementById('model-select');
  const highlightsContainer = document.getElementById('highlights-container');

  const btnPlayPause = document.getElementById('btn-play-pause');
  const iconPlay = document.getElementById('icon-play');
  const iconPause = document.getElementById('icon-pause');
  const btnBack5 = document.getElementById('btn-back-5');
  const btnFwd5 = document.getElementById('btn-fwd-5');
  const timeCurrent = document.getElementById('time-current');
  const timeDuration = document.getElementById('time-duration');
  const globalGameTime = document.getElementById('global-game-time');
  const overlayClock = document.getElementById('overlay-clock');
  const speedButtons = document.querySelectorAll('.btn-speed');

  const timelineStage = document.getElementById('timeline-stage');
  const timelineNeedle = document.getElementById('timeline-needle');
  const trackGtContent = document.getElementById('track-gt-content');
  const trackPredContent = document.getElementById('track-pred-content');
  const trackWindowsContent = document.getElementById('track-windows-content');
  const dspCanvas = document.getElementById('dsp-canvas');

  const chunkTimerBadge = document.getElementById('chunk-timer-badge');
  const chunkGtCount = document.getElementById('chunk-gt-count');
  const chunkPredCount = document.getElementById('chunk-pred-count');

  const windowStatusBadge = document.getElementById('window-status-badge');
  const metricWindowBounds = document.getElementById('metric-window-bounds');
  const metricTriggerReason = document.getElementById('metric-trigger-reason');
  const metricWhistlePeak = document.getElementById('metric-whistle-peak');
  const metricCrowdOptical = document.getElementById('metric-crowd-optical');
  const cotModelBadge = document.getElementById('cot-model-badge');
  const cotText = document.getElementById('cot-text');

  const eventsTableBody = document.getElementById('events-table-body');
  const filterButtons = document.querySelectorAll('.btn-filter');

  const scorecardModal = document.getElementById('scorecard-modal');
  const btnOpenScorecard = document.getElementById('btn-open-scorecard');
  const btnCloseScorecard = document.getElementById('btn-close-scorecard');
  const scorecardTableBody = document.getElementById('scorecard-table-body');

  // Utility formatting
  function formatTime(seconds) {
    if (isNaN(seconds) || seconds === null) return '00:00.0';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ds = Math.floor((seconds % 1) * 10);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ds}`;
  }

  function formatClockOverlay(seconds) {
    if (isNaN(seconds) || seconds === null) return '[00:00]';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `[${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}]`;
  }

  // 1. Initial Data Fetching
  async function init() {
    try {
      const [matchesRes, highlightsRes] = await Promise.all([
        fetch('/api/matches').then(r => r.json()),
        fetch('/api/highlights').then(r => r.json())
      ]);

      state.matches = matchesRes.matches || [];
      state.models = matchesRes.models || {};
      state.highlights = highlightsRes.highlights || [];

      populateSelectors();
      renderHighlights();
      setupEventListeners();

      // Load initial chunk
      await loadChunkData();
    } catch (err) {
      console.error('Initialization error:', err);
    }
  }

  function populateSelectors() {
    // Populate Matches
    matchSelect.innerHTML = '';
    state.matches.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.game_id;
      opt.textContent = `Game ${m.game_id}: ${m.title} (${m.date})`;
      matchSelect.appendChild(opt);
    });

    // Populate Models
    modelSelect.innerHTML = '';
    Object.entries(state.models).forEach(([key, cfg]) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = cfg.name;
      modelSelect.appendChild(opt);
    });

    updateChunkSelectOptions();
  }

  function updateChunkSelectOptions() {
    const curMatch = state.matches.find(m => m.game_id === state.currentMatchId);
    chunkSelect.innerHTML = '';
    if (curMatch) {
      const count = curMatch.chunk_counts[state.currentHalf] || 5;
      for (let i = 0; i < count; i++) {
        const opt = document.createElement('option');
        opt.value = i;
        const startMin = i * 10;
        opt.textContent = `Chunk ${String(i).padStart(2, '0')} (~${startMin}:00 - ${startMin + 10}:00)`;
        chunkSelect.appendChild(opt);
      }
    }
    chunkSelect.value = state.currentChunkIndex;
  }

  function renderHighlights() {
    highlightsContainer.innerHTML = '';
    state.highlights.forEach(h => {
      const chip = document.createElement('button');
      chip.className = 'highlight-chip';
      chip.textContent = h.title;
      chip.title = `${h.description} (${h.badge})`;
      chip.addEventListener('click', () => jumpToHighlight(h));
      highlightsContainer.appendChild(chip);
    });
  }

  async function jumpToHighlight(h) {
    state.currentMatchId = h.game_id;
    state.currentHalf = h.half;
    state.currentChunkIndex = h.chunk_index;

    matchSelect.value = h.game_id;
    halfSelect.value = h.half;
    updateChunkSelectOptions();
    chunkSelect.value = h.chunk_index;

    await loadChunkData();

    // Seek video to relative timestamp
    setTimeout(() => {
      if (videoPlayer.duration && !isNaN(videoPlayer.duration)) {
        videoPlayer.currentTime = Math.min(h.rel_sec, videoPlayer.duration);
      }
    }, 400);
  }

  // 2. Load Chunk Data
  async function loadChunkData() {
    const url = `/api/chunk_data?game_id=${state.currentMatchId}&half=${state.currentHalf}&chunk_index=${state.currentChunkIndex}&model_set=${state.currentModelKey}`;
    try {
      const res = await fetch(url);
      const data = await res.json();
      state.chunkData = data;

      // Update video source
      const videoSrc = data.chunk.video_url;
      if (videoPlayer.getAttribute('src') !== videoSrc) {
        videoPlayer.src = videoSrc;
        videoPlayer.load();
      }

      // Update badges
      const dur = data.chunk.duration_sec || 600.0;
      timeDuration.textContent = formatTime(dur);
      chunkTimerBadge.textContent = `00:00 / ${formatTime(dur)}`;
      chunkGtCount.textContent = `${data.ground_truth.length} GT`;
      chunkPredCount.textContent = `${data.predictions.length} Preds`;

      // Render Tracks & Canvas
      renderTimelineTracks();
      drawDspCanvas();
      renderEventsTable();
      updateCurrentWindow(0.0);
    } catch (err) {
      console.error('Failed to load chunk data:', err);
    }
  }

  // 3. Render Timeline Markers & Windows
  function renderTimelineTracks() {
    if (!state.chunkData) return;
    const dur = state.chunkData.chunk.duration_sec || 600.0;

    // Render Ground Truth Track
    trackGtContent.innerHTML = '';
    state.chunkData.ground_truth.forEach(gt => {
      const pct = (gt.rel_sec / dur) * 100;
      const marker = document.createElement('div');
      marker.className = 'event-marker marker-gt';

      if (gt.label === 'Goal') marker.classList.add('marker-goal');
      else if (gt.label.includes('Red')) marker.classList.add('marker-red');
      else if (gt.label === 'Substitution') marker.classList.add('marker-sub');

      marker.style.left = `${pct}%`;
      marker.textContent = gt.label;
      marker.title = `GT: ${gt.label} @ ${gt.rel_sec}s (${gt.gameTime})`;

      marker.addEventListener('click', (e) => {
        e.stopPropagation();
        videoPlayer.currentTime = gt.rel_sec;
      });
      trackGtContent.appendChild(marker);
    });

    // Render Predictions Track
    trackPredContent.innerHTML = '';
    state.chunkData.predictions.forEach(p => {
      const pct = (p.rel_sec / dur) * 100;
      const marker = document.createElement('div');
      marker.className = 'event-marker marker-pred';

      if (p.label === 'Goal') marker.classList.add('marker-goal');
      else if (p.label.includes('Red')) marker.classList.add('marker-red');
      else if (p.label === 'Substitution') marker.classList.add('marker-sub');

      marker.style.left = `${pct}%`;
      marker.textContent = `${p.label} (${(p.confidence * 100).toFixed(0)}%)`;
      marker.title = `Pred: ${p.label} @ ${p.rel_sec}s | Conf: ${p.confidence} | Model: ${p.model}`;

      marker.addEventListener('click', (e) => {
        e.stopPropagation();
        videoPlayer.currentTime = p.rel_sec;
      });
      trackPredContent.appendChild(marker);
    });

    // Render 60s Sliding Windows Track
    trackWindowsContent.innerHTML = '';
    state.chunkData.windows.forEach(w => {
      const leftPct = (w.start_sec / dur) * 100;
      const widthPct = ((w.end_sec - w.start_sec) / dur) * 100;

      const seg = document.createElement('div');
      seg.className = `window-segment ${w.is_active ? 'window-active' : 'window-skipped'}`;
      seg.style.left = `${leftPct}%`;
      seg.style.width = `${widthPct}%`;
      seg.dataset.windowIndex = w.window_index;
      seg.title = `Window ${w.window_index} (${w.start_sec}s - ${w.end_sec}s): ${w.trigger_reason}`;

      seg.addEventListener('click', (e) => {
        e.stopPropagation();
        videoPlayer.currentTime = w.start_sec;
      });
      trackWindowsContent.appendChild(seg);
    });
  }

  // 4. Draw Canvas DSP Waves (Whistle z-score, Crowd dB, Optical MAD)
  function drawDspCanvas() {
    if (!state.chunkData || !state.chunkData.dsp) return;
    const dsp = state.chunkData.dsp;
    const dur = state.chunkData.chunk.duration_sec || 600.0;

    const ctx = dspCanvas.getContext('2d');
    const width = dspCanvas.width;
    const height = dspCanvas.height;
    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.strokeStyle = '#1e2638';
    ctx.lineWidth = 1;
    for (let y = 20; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // 1. Draw Optical Cut MAD (Cyan, bottom third, 0-50 MAD)
    if (dsp.optical_mad && dsp.optical_mad.length > 0) {
      const optLen = dsp.optical_mad.length;
      ctx.beginPath();
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.2;
      for (let i = 0; i < optLen; i++) {
        const t = i * 1.0;
        const x = (t / dur) * width;
        const val = Math.min(dsp.optical_mad[i], 50.0);
        const y = height - (val / 50.0) * (height * 0.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Optical Cut threshold line (MAD = 30)
      const optCutY = height - (30.0 / 50.0) * (height * 0.4);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
      ctx.moveTo(0, optCutY);
      ctx.lineTo(width, optCutY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2. Draw Crowd dB (Emerald Green, middle band, 0-15 dB)
    if (dsp.crowd_db && dsp.crowd_db.length > 0) {
      const cLen = dsp.crowd_db.length;
      ctx.beginPath();
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.2;
      for (let i = 0; i < cLen; i++) {
        const t = i * 0.5;
        const x = (t / dur) * width;
        const val = Math.max(0, Math.min(dsp.crowd_db[i], 15.0));
        const y = height * 0.65 - (val / 15.0) * (height * 0.35);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Crowd threshold line (+6 dB)
      const crowdY = height * 0.65 - (6.0 / 15.0) * (height * 0.35);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
      ctx.moveTo(0, crowdY);
      ctx.lineTo(width, crowdY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 3. Draw Whistle z-scores (Crimson, top third, 0-10 z)
    if (dsp.whistle_z && dsp.whistle_z.length > 0) {
      const wLen = dsp.whistle_z.length;
      ctx.beginPath();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.6;
      for (let i = 0; i < wLen; i++) {
        const t = i * 0.5;
        const x = (t / dur) * width;
        const val = Math.max(0, Math.min(dsp.whistle_z[i], 10.0));
        const y = height * 0.45 - (val / 10.0) * (height * 0.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Whistle threshold line (z = 4.5)
      const whistleY = height * 0.45 - (4.5 / 10.0) * (height * 0.4);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
      ctx.moveTo(0, whistleY);
      ctx.lineTo(width, whistleY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 5. Update Playhead Needle & Telemetry
  function updatePlayhead(time) {
    if (!state.chunkData) return;
    const dur = state.chunkData.chunk.duration_sec || 600.0;
    const startSec = state.chunkData.chunk.actual_start_sec || 0.0;

    // Update readouts
    timeCurrent.textContent = formatTime(time);
    overlayClock.textContent = formatClockOverlay(time);

    const matchSec = startSec + time;
    const matchMin = Math.floor(matchSec / 60);
    const matchRem = Math.floor(matchSec % 60);
    globalGameTime.textContent = `(Match: ${state.currentHalf} - ${String(matchMin).padStart(2, '0')}:${String(matchRem).padStart(2, '0')})`;

    // Needle position
    const timelineWidth = timelineStage.clientWidth - 110;
    const pct = Math.max(0, Math.min(time / dur, 1.0));
    const needleLeft = 110 + pct * timelineWidth;
    timelineNeedle.style.left = `${needleLeft}px`;

    // Highlight current window in track
    updateCurrentWindow(time);
  }

  function updateCurrentWindow(time) {
    if (!state.chunkData || !state.chunkData.windows) return;
    const windows = state.chunkData.windows;

    let activeWin = null;
    for (const w of windows) {
      if (time >= w.start_sec && time <= w.end_sec) {
        activeWin = w;
        break;
      }
    }

    if (!activeWin && windows.length > 0) {
      activeWin = windows[windows.length - 1];
    }

    // Toggle segment highlight class
    document.querySelectorAll('.window-segment').forEach(seg => {
      seg.classList.remove('window-current');
      if (activeWin && parseInt(seg.dataset.windowIndex) === activeWin.window_index) {
        seg.classList.add('window-current');
      }
    });

    if (activeWin) {
      metricWindowBounds.textContent = `${activeWin.start_sec}s – ${activeWin.end_sec}s`;
      metricTriggerReason.textContent = activeWin.trigger_reason;

      if (activeWin.is_active) {
        windowStatusBadge.className = 'badge badge-active';
        windowStatusBadge.textContent = 'Active Play';
      } else {
        windowStatusBadge.className = 'badge badge-skipped';
        windowStatusBadge.textContent = 'Skipped (0.001s CPU)';
      }

      metricWhistlePeak.textContent = `z = ${activeWin.max_whistle_z.toFixed(2)}`;
      metricCrowdOptical.textContent = `+${activeWin.max_crowd_db.toFixed(1)} dB / ${activeWin.max_optical_mad.toFixed(0)} MAD`;

      if (activeWin.commentary && activeWin.commentary.length > 0) {
        cotText.textContent = `"${activeWin.commentary}"`;
      } else {
        cotText.textContent = `"[No commentary emitted - inactive midfield window filtered out on CPU]"`;
      }
    }
  }

  // 6. Events Table
  function renderEventsTable() {
    if (!state.chunkData) return;
    const gtEvents = state.chunkData.ground_truth || [];
    const predEvents = state.chunkData.predictions || [];

    eventsTableBody.innerHTML = '';

    // Merge into combined list
    const combined = [];
    gtEvents.forEach(gt => {
      combined.push({
        type: 'GT',
        label: gt.label,
        rel_sec: gt.rel_sec,
        gameTime: gt.gameTime,
        conf: '1.00',
        model: 'Ground Truth',
        team: gt.team
      });
    });

    predEvents.forEach(p => {
      // Find closest GT event for drift
      let minDrift = null;
      let closestGT = null;
      gtEvents.forEach(gt => {
        if (gt.label === p.label) {
          const d = p.rel_sec - gt.rel_sec;
          if (minDrift === null || Math.abs(d) < Math.abs(minDrift)) {
            minDrift = d;
            closestGT = gt;
          }
        }
      });

      combined.push({
        type: 'PRED',
        label: p.label,
        rel_sec: p.rel_sec,
        gameTime: p.gameTime,
        conf: p.confidence.toFixed(2),
        model: p.model,
        drift: minDrift !== null ? minDrift : null
      });
    });

    combined.sort((a, b) => a.rel_sec - b.rel_sec);

    combined.forEach(ev => {
      const tr = document.createElement('tr');
      const timeStr = formatTime(ev.rel_sec);

      let driftHtml = '—';
      if (ev.type === 'PRED' && ev.drift !== null) {
        const sign = ev.drift >= 0 ? '+' : '';
        const isTight = Math.abs(ev.drift) <= 1.0;
        const colorClass = isTight ? 'style="color:#10b981;font-weight:700;"' : 'style="color:#60a5fa;"';
        driftHtml = `<span ${colorClass}>${sign}${ev.drift.toFixed(2)}s</span>`;
      }

      tr.innerHTML = `
        <td class="time-cell">${timeStr}</td>
        <td><strong>${ev.label}</strong></td>
        <td><span class="badge-tag ${ev.type === 'GT' ? 'tag-gt' : 'tag-pred'}">${ev.type === 'GT' ? 'Ground Truth' : ev.model}</span></td>
        <td>${ev.conf}</td>
        <td>${driftHtml}</td>
        <td><button class="btn-jump">Jump</button></td>
      `;

      tr.querySelector('.btn-jump').addEventListener('click', () => {
        videoPlayer.currentTime = ev.rel_sec;
      });

      eventsTableBody.appendChild(tr);
    });
  }

  // 7. Official Scorecard Modal
  async function openScorecard() {
    scorecardModal.style.display = 'flex';
    if (!state.scorecardData) {
      try {
        const res = await fetch('/api/scorecard');
        state.scorecardData = await res.json();
      } catch (err) {
        console.error('Failed to load scorecard:', err);
        return;
      }
    }

    renderScorecardTable();
  }

  function renderScorecardTable() {
    if (!state.scorecardData) return;
    const perClass = state.scorecardData.per_class;
    const classes = [
      'Penalty', 'Kick-off', 'Goal', 'Substitution', 'Offside',
      'Shots on target', 'Shots off target', 'Clearance', 'Ball out of play',
      'Throw-in', 'Foul', 'Indirect free-kick', 'Direct free-kick', 'Corner',
      'Yellow card', 'Red card', 'Yellow->red card'
    ];

    scorecardTableBody.innerHTML = '';
    classes.forEach(cls => {
      const q = perClass.qwen_loose ? (perClass.qwen_loose[cls] || 0.0) : 0.0;
      const l = perClass.lfm_loose ? (perClass.lfm_loose[cls] || 0.0) : 0.0;
      const m = perClass.mini_loose ? (perClass.mini_loose[cls] || 0.0) : 0.0;
      const base = perClass.base_loose ? (perClass.base_loose[cls] || 0.0) : 0.0;
      const m15 = perClass.m15_loose ? (perClass.m15_loose[cls] || 0.0) : 0.0;
      const s2 = perClass.stage2_loose ? (perClass.stage2_loose[cls] || 0.0) : 0.0;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${cls}</td>
        <td>${q > 0 ? q.toFixed(4) : '<span style="color:#64748b;">0.0000</span>'}</td>
        <td>${l > 0 ? l.toFixed(4) : '<span style="color:#64748b;">0.0000</span>'}</td>
        <td>${m > 0 ? m.toFixed(4) : '<span style="color:#64748b;">0.0000</span>'}</td>
        <td>${base.toFixed(4)}</td>
        <td>${m15.toFixed(4)}</td>
        <td class="cell-highlight">${s2.toFixed(4)}</td>
      `;
      scorecardTableBody.appendChild(tr);
    });
  }

  // 8. Event Listeners
  function setupEventListeners() {
    // Selectors
    matchSelect.addEventListener('change', () => {
      state.currentMatchId = parseInt(matchSelect.value);
      state.currentChunkIndex = 0;
      updateChunkSelectOptions();
      loadChunkData();
    });

    halfSelect.addEventListener('change', () => {
      state.currentHalf = parseInt(halfSelect.value);
      state.currentChunkIndex = 0;
      updateChunkSelectOptions();
      loadChunkData();
    });

    chunkSelect.addEventListener('change', () => {
      state.currentChunkIndex = parseInt(chunkSelect.value);
      loadChunkData();
    });

    modelSelect.addEventListener('change', () => {
      state.currentModelKey = modelSelect.value;
      loadChunkData();
    });

    // Video Transport
    btnPlayPause.addEventListener('click', togglePlayPause);
    btnBack5.addEventListener('click', () => { videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5); });
    btnFwd5.addEventListener('click', () => { videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 5); });

    videoPlayer.addEventListener('timeupdate', () => {
      updatePlayhead(videoPlayer.currentTime);
    });

    videoPlayer.addEventListener('play', () => {
      state.isPlaying = true;
      iconPlay.style.display = 'none';
      iconPause.style.display = 'block';
    });

    videoPlayer.addEventListener('pause', () => {
      state.isPlaying = false;
      iconPlay.style.display = 'block';
      iconPause.style.display = 'none';
    });

    // Speed Controls
    speedButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        speedButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.playbackSpeed = parseFloat(btn.dataset.speed);
        videoPlayer.playbackRate = state.playbackSpeed;
      });
    });

    // Timeline Scrubbing via Click
    timelineStage.addEventListener('click', (e) => {
      if (!state.chunkData) return;
      const rect = timelineStage.getBoundingClientRect();
      const clickX = e.clientX - rect.left - 110;
      const totalWidth = rect.width - 110;
      if (clickX >= 0 && totalWidth > 0) {
        const pct = Math.max(0, Math.min(clickX / totalWidth, 1.0));
        const dur = state.chunkData.chunk.duration_sec || 600.0;
        videoPlayer.currentTime = pct * dur;
      }
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayPause();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - (e.shiftKey ? 5 : 1));
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + (e.shiftKey ? 5 : 1));
      }
    });

    // Scorecard Modal
    btnOpenScorecard.addEventListener('click', openScorecard);
    btnCloseScorecard.addEventListener('click', () => { scorecardModal.style.display = 'none'; });
    scorecardModal.addEventListener('click', (e) => {
      if (e.target === scorecardModal) scorecardModal.style.display = 'none';
    });
  }

  function togglePlayPause() {
    if (videoPlayer.paused) {
      videoPlayer.play();
    } else {
      videoPlayer.pause();
    }
  }

  // Start app
  init();
});
