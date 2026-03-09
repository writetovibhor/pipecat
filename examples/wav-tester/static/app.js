/**
 * Pipecat WAV Tester — Frontend JavaScript
 *
 * Flow:
 *  1. Fetch /files → render file list in sidebar
 *  2. User clicks a file → fetch /audio/<path>, decode with Web Audio API,
 *     resample to 16 kHz mono Float32
 *  3. User clicks Connect → open WebSocket to /ws
 *  4. User presses Space / clicks Play:
 *       • Playing  → read 10 ms chunks from the audio buffer, convert to
 *                    Int16, send over WebSocket at real-time rate (clock-
 *                    compensated setTimeout, not setInterval); animate viz
 *       • Pausing  → stop chunk loop; send ~600 ms of silence so pipecat's
 *                    VAD detects end-of-speech and triggers STT → LLM → TTS
 *  5. Receive binary frame from server → raw PCM Int16 bytes → schedule
 *     playback via AudioContext; animate output visualizer
 *  6. Receive text frame from server → JSON message → update transcript strip
 */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────────
const SAMPLE_RATE   = 16000;   // Hz — must match pipecat transport config
const CHUNK_SAMPLES = 160;     // 10 ms @ 16 kHz
const CHUNK_MS      = 10;      // interval between sends (ms)
const SILENCE_MS    = 600;     // silence burst duration on pause (ms)
const VIZ_BARS      = 32;      // number of equaliser bars

// ── State ──────────────────────────────────────────────────────────────────
let ws               = null;
let audioCtx         = null;   // 16 kHz AudioContext for TTS output
let monitorCtx       = null;   // native-rate AudioContext for input monitoring
let pcmBuffer        = null;   // Float32Array — full file @ 16 kHz mono
let playhead         = 0;      // current position in samples
let isPlaying        = false;
let chunkTimer       = null;   // setTimeout handle
let playStartTime    = null;   // performance.now() when playback started
let samplesSent      = 0;      // samples sent since playback started (for clock compensation)
let nextPlayTime     = 0;      // next scheduled output audio time
let monitorNextTime  = 0;      // next scheduled input monitor time
let selectedFile     = null;   // { name, url }
let silenceTimer     = null;   // for sending silence burst
let activeFilter     = 'none'; // 'none' | 'deepfilter' | 'quail'
// ── End-of-conversation state ──────────────────────────────────────────────
let wavPlaybackDone   = false;  // true after file reaches end
let botStartedAfterEOF = false; // bot spoke at least once after EOF
let eocTimer          = null;   // fires if no bot speech within EOC_NO_RESPONSE_MS
const EOC_NO_RESPONSE_MS = 8000; // ms to wait for bot response before declaring EOC

// ── DOM refs ───────────────────────────────────────────────────────────────
const connectBtn      = () => document.getElementById('connect-btn');
const playBtn         = () => document.getElementById('play-btn');
const playIcon        = () => document.getElementById('play-icon');
const fileNameLabel   = () => document.getElementById('file-name');
const timeDisplay     = () => document.getElementById('time-display');
const progressFill    = () => document.getElementById('progress-fill');
const botStatus       = () => document.getElementById('bot-status');
const wsBadge         = () => document.getElementById('ws-status');
const wsStatusText    = () => document.getElementById('ws-status-text');

// ── Metrics data store (for JSON export) ──────────────────────────────────
let _metricsSnapshot = null;   // latest session-level aggregates from server
let _turnHistory     = [];     // one entry per turn, augmented with quality on arrival

// ── Snapshot comparison store ───────────────────────────────────────────────
let _snapshots = [];  // up to 2 entries: { label, filterName, session, snr, noiseFloor, clipPct }

// ── Audio quality tracking ─────────────────────────────────────────────────
let speechRmsSum  = 0, speechRmsCount  = 0;  // RMS during VAD-detected speech
let noiseRmsSum   = 0, noiseRmsCount   = 0;  // RMS during silence while playing
let isSpeechActive = false;                  // updated by user_speaking events
let clipCount      = 0, totalSamplesSent = 0; // clipping: samples at ±32767

function updateAudioQuality(int16Data) {
  const rms = computeRms(int16Data);
  if (isSpeechActive) {
    speechRmsSum += rms; speechRmsCount++;
  } else if (isPlaying) {
    noiseRmsSum += rms; noiseRmsCount++;
  }

  // Count clipped samples
  totalSamplesSent += int16Data.length;
  for (let i = 0; i < int16Data.length; i++) {
    if (int16Data[i] >= 32767 || int16Data[i] <= -32768) clipCount++;
  }

  // Refresh SNR + clipping display every ~500ms (every 50 chunks)
  if ((speechRmsCount + noiseRmsCount) % 50 === 1) renderAudioKpis();
}

function renderAudioKpis() {
  const speechRms = speechRmsCount > 0 ? speechRmsSum / speechRmsCount : 0;
  const noiseRms  = noiseRmsCount  > 0 ? noiseRmsSum  / noiseRmsCount  : 0;

  const noiseEl = document.getElementById('kv-noise');
  if (noiseRms > 0) {
    noiseEl.textContent = noiseRms.toFixed(4);
    noiseEl.className = 'kpi-value ' + (noiseRms < 0.01 ? 'good' : noiseRms < 0.05 ? 'warn' : 'bad');
  }

  const snrEl = document.getElementById('kv-snr');
  if (speechRms > 0 && noiseRms > 0) {
    const snr = 20 * Math.log10(speechRms / noiseRms);
    snrEl.textContent = snr.toFixed(1) + ' dB';
    snrEl.className = 'kpi-value ' + (snr > 20 ? 'good' : snr > 10 ? 'warn' : 'bad');
  }

  // Clipping rate
  const clipEl = document.getElementById('kv-clip');
  if (clipEl && totalSamplesSent > 0) {
    const rate = clipCount / totalSamplesSent * 100;
    clipEl.textContent = rate < 0.001 ? '0%' : rate.toFixed(3) + '%';
    clipEl.className = 'kpi-value ' + (rate < 0.1 ? 'good' : rate < 1 ? 'warn' : 'bad');
  }
}

function resetAudioQuality() {
  speechRmsSum = speechRmsCount = noiseRmsSum = noiseRmsCount = 0;
  clipCount = totalSamplesSent = 0;
  ['kv-snr', 'kv-noise', 'kv-clip'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.textContent = '—'; el.className = 'kpi-value'; }
  });
}

// ── Logging ────────────────────────────────────────────────────────────────
let _logScrollPending = false;

function log(msg, level = 'info') {
  const logEl = document.getElementById('log');
  const entry = document.createElement('div');
  entry.className = `log-entry log-${level}`;
  const now = new Date();
  const ts  = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
  entry.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg">${msg}</span>`;
  logEl.appendChild(entry);
  // Batch scroll updates — one rAF per burst of messages avoids forced reflow per entry
  if (!_logScrollPending) {
    _logScrollPending = true;
    requestAnimationFrame(() => {
      logEl.scrollTop = logEl.scrollHeight;
      _logScrollPending = false;
    });
  }
}

function clearLog() { document.getElementById('log').innerHTML = ''; }

function downloadLog() {
  const entries = document.querySelectorAll('#log .log-entry');
  if (!entries.length) return;

  const lines = Array.from(entries).map(el => {
    const ts  = el.querySelector('.log-time')?.textContent ?? '';
    const msg = el.querySelector('.log-msg')?.textContent  ?? '';
    return `[${ts}] ${msg}`;
  });

  const stem = selectedFile ? selectedFile.name.replace(/\.wav$/i, '') : 'session';
  const ts   = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `log_${stem}_${ts}.txt`;
  a.click();
  URL.revokeObjectURL(url);

  const btn  = document.getElementById('download-log-btn');
  const orig = btn.textContent;
  btn.textContent = '✓ Saved!';
  btn.style.color = 'var(--green)';
  setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 2000);
}

// ── File list ──────────────────────────────────────────────────────────────
async function loadFileList() {
  const listEl = document.getElementById('file-list');
  try {
    const resp = await fetch('/files');
    const data = await resp.json();

    if (data.error) {
      listEl.innerHTML = `<p class="hint" style="color:#ff5a5a">${data.error}</p>`;
      return;
    }
    if (!data.files.length) {
      listEl.innerHTML = '<p class="hint">No .wav files found in INPUT_DIR.</p>';
      return;
    }

    // Group by directory
    const groups = {};
    for (const f of data.files) {
      const dir = f.dir || '(root)';
      (groups[dir] = groups[dir] || []).push(f);
    }

    listEl.innerHTML = '';
    for (const [dir, files] of Object.entries(groups)) {
      const grp = document.createElement('div');
      grp.className = 'dir-group';
      if (dir !== '(root)') {
        grp.innerHTML = `<div class="dir-label">${escHtml(dir)}</div>`;
      }
      for (const f of files) {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.dataset.url  = f.url;
        item.dataset.name = f.name;
        item.innerHTML = `
          <span class="file-icon">🎵</span>
          <span class="file-item-name" title="${escHtml(f.name)}">${escHtml(f.name)}</span>
          <span class="file-size">${formatBytes(f.size)}</span>`;
        item.addEventListener('click', () => onFileClick(item, f));
        grp.appendChild(item);
      }
      listEl.appendChild(grp);
    }
    log(`Found ${data.files.length} file(s)`, 'ok');
  } catch (err) {
    listEl.innerHTML = `<p class="hint" style="color:#ff5a5a">Failed to load files: ${err.message}</p>`;
    log(`File list error: ${err.message}`, 'error');
  }
}

// ── Select & load WAV ──────────────────────────────────────────────────────
async function onFileClick(itemEl, file) {
  // Highlight
  document.querySelectorAll('.file-item').forEach(el => el.classList.remove('selected'));
  itemEl.classList.add('selected');

  // Stop current playback
  if (isPlaying) pausePlayback();
  pcmBuffer = null;
  playhead  = 0;
  resetAudioQuality();
  resetEocState();

  selectedFile = file;
  fileNameLabel().textContent = file.name;
  timeDisplay().textContent   = '0:00 / 0:00';
  progressFill().style.width  = '0%';
  playBtn().disabled = true;

  log(`Loading ${file.name} …`);
  try {
    const resp   = await fetch(file.url);
    const ab     = await resp.arrayBuffer();
    const decoded = await decodeAndResample(ab);
    pcmBuffer = decoded;
    playhead  = 0;

    const duration = pcmBuffer.length / SAMPLE_RATE;
    fileNameLabel().textContent = file.name;
    timeDisplay().textContent   = `0:00 / ${formatTime(duration)}`;

    playBtn().disabled = (ws === null || ws.readyState !== WebSocket.OPEN);
    log(`Ready: ${file.name} (${formatTime(duration)})`, 'ok');
  } catch (err) {
    log(`Failed to load ${file.name}: ${err.message}`, 'error');
  }
}

async function decodeAndResample(arrayBuffer) {
  // 1. Create a temporary AudioContext to decode the WAV
  const tmpCtx  = new AudioContext();
  const decoded = await tmpCtx.decodeAudioData(arrayBuffer);
  await tmpCtx.close();

  // 2. Resample + down-mix to mono 16 kHz via OfflineAudioContext
  const targetLen = Math.ceil(decoded.duration * SAMPLE_RATE);
  const offline   = new OfflineAudioContext(1, targetLen, SAMPLE_RATE);

  // Down-mix to mono: merge channels via ChannelMergerNode
  const src = offline.createBufferSource();
  src.buffer = decoded;

  // If stereo+, use a channel splitter → merger to average channels
  if (decoded.numberOfChannels > 1) {
    const splitter = offline.createChannelSplitter(decoded.numberOfChannels);
    const merger   = offline.createChannelMerger(1);
    const gainVal  = 1 / decoded.numberOfChannels;
    src.connect(splitter);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const g = offline.createGain();
      g.gain.value = gainVal;
      splitter.connect(g, ch, 0);
      g.connect(merger, 0, 0);
    }
    merger.connect(offline.destination);
  } else {
    src.connect(offline.destination);
  }

  src.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);   // Float32Array, 16 kHz mono
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function toggleConnect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    disconnectWs();
  } else {
    connectWs();
  }
}

function connectWs() {
  if (!audioCtx) {
    audioCtx     = new AudioContext({ sampleRate: SAMPLE_RATE });
    nextPlayTime = 0;
  }
  if (!monitorCtx) {
    monitorCtx      = new AudioContext();  // native sample rate — browser resamples 16 kHz input
    monitorNextTime = 0;
  }

  setWsStatus('connecting');
  log('Connecting to WebSocket…');

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const recordOriginal = document.getElementById('record-original-cb')?.checked ? 'true' : 'false';
  ws = new WebSocket(`${protocol}//${location.host}/ws?record_original=${recordOriginal}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setWsStatus('connected');
    connectBtn().textContent = 'Disconnect';
    const recOrig = document.getElementById('record-original-cb')?.checked;
    log(`WebSocket connected — recording: filtered input + bot output${recOrig ? ' + original input' : ''}`, 'ok');
    if (pcmBuffer) playBtn().disabled = false;
    loadFileList();  // refresh file list on connect
  };

  ws.onclose = (ev) => {
    setWsStatus('disconnected');
    connectBtn().textContent = 'Connect';
    playBtn().disabled = true;
    if (isPlaying) pausePlayback(false);  // pause without sending silence
    // Reset filter selector — new connection always starts with no filter active
    activeFilter = 'none';
    const fSel = document.getElementById('filter-select');
    if (fSel) fSel.value = 'none';
    showFilterConfigPanel('none');
    resetFilterStatus();
    log(`WebSocket closed (code ${ev.code})`, ev.wasClean ? 'info' : 'warn');
    ws = null;
  };

  ws.onerror = () => {
    log('WebSocket error', 'error');
  };

  ws.onmessage = handleServerMessage;
}

function disconnectWs() {
  if (isPlaying) pausePlayback(false);
  resetEocState();
  if (ws) ws.close();
}

// ── Filter selection ─────────────────────────────────────────────────────────
function showFilterConfigPanel(filterName) {
  document.getElementById('deepfilter-config').style.display = filterName === 'deepfilter' ? '' : 'none';
  document.getElementById('quail-config').style.display      = filterName === 'quail'      ? '' : 'none';
}

function onFilterSelectChange(filterName) {
  activeFilter = filterName;
  showFilterConfigPanel(filterName);
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  if (filterName === 'deepfilter') {
    ws.send(JSON.stringify({ type: 'filter_select', filter: 'deepfilter' }));
    log('DeepFilterNet noise suppression enabled', 'ok');
  } else if (filterName === 'quail') {
    const modelId = document.getElementById('quail-model-select')?.value || 'quail-vf-l-16khz';
    ws.send(JSON.stringify({ type: 'filter_select', filter: 'quail', model_id: modelId }));
    log(`Quail speech enhancement enabled (${modelId})`, 'ok');
  } else {
    ws.send(JSON.stringify({ type: 'filter_select', filter: 'none' }));
    log('Audio filter disabled', 'info');
  }
}

function onQuailModelChange() {
  if (activeFilter !== 'quail' || !ws || ws.readyState !== WebSocket.OPEN) return;
  const modelId = document.getElementById('quail-model-select')?.value || 'quail-vf-l-16khz';
  ws.send(JSON.stringify({ type: 'quail_update', model_id: modelId }));
  log(`Quail model updated: ${modelId}`, 'info');
}

function onAttenLimChange(value) {
  const slider = document.getElementById('atten-slider');
  const label  = document.getElementById('atten-value');
  if (value === null) {
    // Unlimited suppression
    if (label) label.textContent = '∞';
    if (slider) slider.value = 40;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'deepfilter_atten_lim', value: null }));
      log('DeepFilter attenuation limit: unlimited (full suppression)', 'info');
    }
  } else {
    const db = parseFloat(value);
    if (label) label.textContent = db + ' dB';
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'deepfilter_atten_lim', value: db }));
    }
  }
}

function handleServerMessage(ev) {
  if (ev.data instanceof ArrayBuffer) {
    // Binary → TTS audio (raw PCM Int16)
    const int16 = new Int16Array(ev.data);
    scheduleOutputAudio(int16);
    outputViz.feed(int16);
  } else if (typeof ev.data === 'string') {
    // Text → JSON status/transcription
    try {
      const msg = JSON.parse(ev.data);
      handleJsonMessage(msg);
    } catch (_) { /* ignore */ }
  }
}

function handleJsonMessage(msg) {
  switch (msg.type) {
    case 'transcription':
      log(`You: "${msg.text}"`, 'info');
      break;
    case 'bot_text':
      log(`Bot: "${msg.text}"`, 'bot');
      break;
    case 'user_speaking':
      isSpeechActive = !!msg.value;
      botStatus().textContent = msg.value ? '' : 'Thinking…';
      break;
    case 'bot_speaking':
      botStatus().textContent = msg.value ? '🔊 Speaking…' : '';
      log(msg.value ? 'Bot started speaking' : 'Bot stopped speaking', 'bot');
      if (msg.value && wavPlaybackDone) {
        // Bot started speaking after file ended — cancel no-response timer
        if (eocTimer) { clearTimeout(eocTimer); eocTimer = null; }
        botStartedAfterEOF = true;
      } else if (!msg.value && wavPlaybackDone && botStartedAfterEOF) {
        // Bot finished speaking after file ended → conversation complete
        log('✅ Conversation complete', 'ok');
        wavPlaybackDone    = false;
        botStartedAfterEOF = false;
      }
      break;
    case 'filter_status':
      onFilterStatus(msg);
      break;
    case 'metrics_update':
      onMetricsUpdate(msg);
      break;
    case 'turn_quality':
      onTurnQuality(msg);
      break;
    default:
      log(`Server: ${JSON.stringify(msg)}`, 'info');
  }
}

// ── Filter status bar ───────────────────────────────────────────────────────

const FSTATUS_LABELS = {
  pending:  '—',
  loading:  'loading…',
  ready:    'ready',
  inactive: 'ready',
  active:   'active',
  error:    'error',
  disabled: 'not installed',
};

function onFilterStatus(msg) {
  const { filter, state, detail } = msg;

  // Show the bar the first time we get any status.
  const bar = document.getElementById('filter-status-bar');
  if (bar) bar.style.display = '';

  const pill = document.getElementById('fstatus-' + filter);
  const txt  = document.getElementById('fstatus-text-' + filter);
  if (!pill || !txt) return;

  pill.dataset.state = state;
  const label = (state === 'error' && detail) ? detail
              : (FSTATUS_LABELS[state] ?? state);
  txt.textContent = label;
  pill.title = detail || '';
}

function resetFilterStatus() {
  const bar = document.getElementById('filter-status-bar');
  if (bar) bar.style.display = 'none';
  ['deepfilter', 'quail'].forEach(name => {
    const pill = document.getElementById('fstatus-' + name);
    const txt  = document.getElementById('fstatus-text-' + name);
    if (pill) pill.dataset.state = 'pending';
    if (txt)  txt.textContent = '—';
  });
}

// ── Metrics panel ──────────────────────────────────────────────────────────
function fmtMs(ms) {
  if (ms == null) return '—';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : Math.round(ms) + 'ms';
}

function latencyClass(ms) {
  if (ms == null) return '';
  if (ms < 400)  return 'good';
  if (ms < 900)  return 'warn';
  return 'bad';
}

function setKpi(id, value, cls = '') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  el.className = 'kpi-value' + (cls ? ' ' + cls : '');
}

function fmtConf(v) {
  if (v == null) return '—';
  return (v * 100).toFixed(1) + '%';
}
function confClass(v) {
  if (v == null) return '';
  if (v >= 0.90) return 'good';
  if (v >= 0.70) return 'warn';
  return 'bad';
}
function wpmClass(v) {
  if (v == null) return '';
  if (v >= 80 && v <= 180) return 'good';
  if (v <= 220) return 'warn';
  return 'bad';
}
function wordsClass(v) {
  if (v == null || v === 0) return '';
  if (v <= 30) return 'good';
  if (v <= 60) return 'warn';
  return 'bad';
}
function qualityClass(avg) {
  if (avg == null) return '';
  if (avg >= 4.5) return 'good';
  if (avg >= 3.0) return 'warn';
  return 'bad';
}

function setKpiStat(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = fmtMs(val);
}

function onMetricsUpdate(msg) {
  const { turn, session } = msg;
  if (!turn) return;

  // Update turn badge
  const badge = document.getElementById('metrics-turn-badge');
  if (badge) badge.textContent = `Turn ${session.turn_count}`;

  // Row 1: latency KPIs — show session mean as primary value, p50/p99 as sub-stats
  setKpi('kv-rtt', fmtMs(session.rtt_mean), latencyClass(session.rtt_mean));
  setKpiStat('kp50-rtt', session.rtt_p50); setKpiStat('kp99-rtt', session.rtt_p99);

  setKpi('kv-stt', fmtMs(session.stt_mean), latencyClass(session.stt_mean));
  setKpiStat('kp50-stt', session.stt_p50); setKpiStat('kp99-stt', session.stt_p99);

  setKpi('kv-llm', fmtMs(session.llm_mean), latencyClass(session.llm_mean));
  setKpiStat('kp50-llm', session.llm_p50); setKpiStat('kp99-llm', session.llm_p99);

  setKpi('kv-tts', fmtMs(session.tts_mean), latencyClass(session.tts_mean));
  setKpiStat('kp50-tts', session.tts_p50); setKpiStat('kp99-tts', session.tts_p99);

  // Row 2: conversation health
  setKpi('kv-interrupts', session.total_interruptions,
         session.total_interruptions === 0 ? 'good' : 'warn');

  // Row 3: speech + content — session averages
  if (session.avg_stt_confidence != null)
    setKpi('kv-stt-conf', fmtConf(session.avg_stt_confidence), confClass(session.avg_stt_confidence));
  if (session.avg_speaking_rate_wpm != null)
    setKpi('kv-wpm', Math.round(session.avg_speaking_rate_wpm) + ' wpm', wpmClass(session.avg_speaking_rate_wpm));
  if (session.avg_bot_word_count != null)
    setKpi('kv-response-len', Math.round(session.avg_bot_word_count) + ' w', wordsClass(session.avg_bot_word_count));

  // Row 4: Active filter stats (shown whenever a filter is active)
  if (msg.filter && msg.filter.active && msg.filter.active !== 'none') {
    const f = msg.filter;
    const filterRow = document.getElementById('filter-stats-row');
    if (filterRow) filterRow.style.display = '';
    if (f.avg_latency_ms != null)
      setKpi('kv-filter-latency', fmtMs(f.avg_latency_ms), f.avg_latency_ms < 10 ? 'good' : f.avg_latency_ms < 30 ? 'warn' : 'bad');
    if (f.avg_noise_reduction_db != null) {
      document.getElementById('kpi-filter-noise-db').style.display = '';
      setKpi('kv-filter-noise-db', f.avg_noise_reduction_db.toFixed(1) + ' dB',
             f.avg_noise_reduction_db > 3 ? 'good' : f.avg_noise_reduction_db > 0 ? '' : 'warn');
    } else {
      // Noise reduction only available for DeepFilterNet — hide for Quail
      document.getElementById('kpi-filter-noise-db').style.display = 'none';
    }
    // Active filter name + model
    const activeEl = document.getElementById('kv-filter-active');
    const modelEl  = document.getElementById('kv-filter-model');
    if (activeEl) activeEl.textContent = f.active === 'deepfilter' ? 'DeepFilterNet' : 'Quail';
    if (modelEl)  modelEl.textContent  = f.model || '';
    // store for export
    if (_metricsSnapshot) _metricsSnapshot.filter = f;
  }

  // Store for JSON export
  _metricsSnapshot = { ...session };
  if (!_turnHistory.find(t => t.turn_id === turn.turn_id)) {
    _turnHistory.push({
      turn_id:           turn.turn_id,
      user_text:         turn.user_text,
      bot_text:          turn.bot_text,
      rtt_ms:            turn.rtt_ms,
      stt_ms:            turn.stt_ms,
      llm_ms:            turn.llm_ms,
      tts_ms:            turn.tts_ms,
      stt_confidence:    turn.stt_confidence,
      speaking_rate_wpm: turn.speaking_rate_wpm,
      bot_word_count:    turn.bot_word_count,
      quality:           null,
    });
  }

  // Add a new row to the turn table (quality cell starts as pending)
  const tbody = document.getElementById('turn-tbody');
  if (tbody) {
    const tr = document.createElement('tr');
    tr.id = `turn-row-${turn.turn_id}`;
    tr.innerHTML = `
      <td>${turn.turn_id}</td>
      <td class="user-said" title="${escHtml(turn.user_text)}">"${escHtml(turn.user_text)}"</td>
      <td class="${latencyClass(turn.rtt_ms)}">${fmtMs(turn.rtt_ms)}</td>
      <td class="${latencyClass(turn.stt_ms)}">${fmtMs(turn.stt_ms)}</td>
      <td class="${latencyClass(turn.llm_ms)}">${fmtMs(turn.llm_ms)}</td>
      <td class="${latencyClass(turn.tts_ms)}">${fmtMs(turn.tts_ms)}</td>
      <td class="${confClass(turn.stt_confidence)}">${fmtConf(turn.stt_confidence)}</td>
      <td class="${wpmClass(turn.speaking_rate_wpm)}">${turn.speaking_rate_wpm != null ? Math.round(turn.speaking_rate_wpm) : '—'}</td>
      <td class="${wordsClass(turn.bot_word_count)}">${turn.bot_word_count || '—'}</td>
      <td id="tq-${turn.turn_id}"><span class="quality-pending">…</span></td>`;
    tbody.appendChild(tr);
    tr.scrollIntoView({ block: 'nearest' });
  }
}

function onTurnQuality(msg) {
  const { turn_id, score, issue } = msg;

  // Persist quality in data store
  const entry = _turnHistory.find(t => t.turn_id === turn_id);
  if (entry) entry.quality = { score, issue };

  // Update quality cell in the table
  const cell = document.getElementById(`tq-${turn_id}`);
  if (cell) {
    if (issue === 'error' || score === 0) {
      cell.innerHTML = `<span style="color:var(--red);font-size:11px">ERR</span>`;
    } else {
      const stars = '★'.repeat(score) + '☆'.repeat(5 - score);
      const issueStr = issue && issue !== 'none'
        ? ` <span style="color:var(--text-muted);font-size:10px">(${issue})</span>` : '';
      cell.innerHTML = `<span class="quality-stars">${stars}</span>${issueStr}`;
    }
  }

  // Update avg quality KPI
  const rows = document.querySelectorAll('#turn-tbody tr');
  const scores = [];
  rows.forEach(r => {
    const c = r.querySelector('[id^="tq-"]');
    if (c) {
      const s = (c.querySelector('.quality-stars') || {}).textContent || '';
      const n = (s.match(/★/g) || []).length;
      if (n > 0) scores.push(n);
    }
  });
  if (scores.length) {
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const stars = '★'.repeat(Math.round(avg)) + '☆'.repeat(5 - Math.round(avg));
    setKpi('kv-quality', `${stars} ${avg.toFixed(1)}`, qualityClass(avg));
  }

  log(`Quality turn ${turn_id}: ${score}/5${issue !== 'none' ? ' — ' + issue : ''}`, 'info');
}

function resetMetrics() {
  _metricsSnapshot = null;
  _turnHistory = [];
  ['kv-rtt','kv-stt','kv-llm','kv-tts','kv-quality',
   'kv-stt-conf','kv-wpm','kv-response-len'].forEach(id => setKpi(id, '—'));
  ['kp50-rtt','kp99-rtt','kp50-stt','kp99-stt',
   'kp50-llm','kp99-llm','kp50-tts','kp99-tts'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = '—';
  });
  setKpi('kv-interrupts', '0', 'good');
  const badge = document.getElementById('metrics-turn-badge');
  if (badge) badge.textContent = 'Turn 0';
  const tbody = document.getElementById('turn-tbody');
  if (tbody) tbody.innerHTML = '';
  resetAudioQuality();
}

function setWsStatus(state) {
  const badge = wsBadge();
  badge.classList.remove('ws-connected', 'ws-disconnected', 'ws-connecting');
  badge.classList.add(`ws-${state}`);
  wsStatusText().textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

// ── Input audio monitoring (local playback of the WAV being sent) ──────────
function scheduleInputAudio(float32Data) {
  if (!monitorCtx) return;
  if (monitorCtx.state === 'suspended') monitorCtx.resume();

  // AudioBuffer at 16 kHz — monitorCtx resamples to native rate on playback
  const buf = monitorCtx.createBuffer(1, float32Data.length, SAMPLE_RATE);
  buf.copyToChannel(new Float32Array(float32Data), 0);

  const src = monitorCtx.createBufferSource();
  src.buffer = buf;
  src.connect(monitorCtx.destination);

  const now = monitorCtx.currentTime;
  if (monitorNextTime < now + 0.02) monitorNextTime = now + 0.02;
  src.start(monitorNextTime);
  monitorNextTime += buf.duration;
}

// ── Output audio scheduling ────────────────────────────────────────────────
function scheduleOutputAudio(int16Data) {
  if (!audioCtx) return;

  // Resume AudioContext if suspended (browser autoplay policy)
  if (audioCtx.state === 'suspended') audioCtx.resume();

  const float32 = int16ToFloat32(int16Data);
  const buf     = audioCtx.createBuffer(1, float32.length, SAMPLE_RATE);
  buf.copyToChannel(float32, 0);

  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (nextPlayTime < now + 0.02) {
    nextPlayTime = now + 0.02;   // 20 ms look-ahead buffer
  }
  src.start(nextPlayTime);
  nextPlayTime += buf.duration;
}

// ── Playback control ───────────────────────────────────────────────────────
function togglePlay() {
  if (!pcmBuffer || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (isPlaying) {
    pausePlayback(true);
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (!pcmBuffer) return;

  // Resume AudioContext (required after user gesture)
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

  monitorNextTime = 0;   // reset so first chunk schedules from now
  isPlaying = true;
  playBtn().classList.add('playing');
  playIcon().textContent = '⏸';
  log('▶ Playback started — sending audio to Pipecat');

  // Cancel any pending silence burst
  if (silenceTimer) { clearInterval(silenceTimer); silenceTimer = null; }

  playStartTime = performance.now();
  samplesSent   = 0;
  scheduleNextChunk();
}

function scheduleNextChunk() {
  if (!isPlaying) return;
  // Compute when the next chunk should be sent based on the wall clock,
  // so drift in the JS event loop doesn't cause audio to outrun real-time.
  const nextSendMs = playStartTime + (samplesSent + CHUNK_SAMPLES) * 1000 / SAMPLE_RATE;
  const delayMs    = Math.max(0, nextSendMs - performance.now());
  chunkTimer = setTimeout(() => {
    sendChunk();
    scheduleNextChunk();
  }, delayMs);
}

function pausePlayback(sendSilence = true) {
  isPlaying = false;
  clearTimeout(chunkTimer);
  chunkTimer    = null;
  playStartTime = null;
  samplesSent   = 0;
  playBtn().classList.remove('playing');
  playIcon().textContent = '▶';

  if (sendSilence && ws && ws.readyState === WebSocket.OPEN) {
    log('⏸ Paused — sending silence to trigger VAD end-of-speech');
    startSilenceBurst();
  } else {
    log('⏸ Paused');
  }
}

function sendChunk() {
  if (!pcmBuffer || !ws || ws.readyState !== WebSocket.OPEN) {
    pausePlayback(false);
    return;
  }

  if (playhead >= pcmBuffer.length) {
    log('📼 End of file — waiting for agent response…', 'ok');
    pausePlayback(true);   // send silence at end too
    playhead = 0;
    updateProgress();
    onWavPlaybackComplete();
    return;
  }

  const end      = Math.min(playhead + CHUNK_SAMPLES, pcmBuffer.length);
  const chunk    = pcmBuffer.subarray(playhead, end);
  const int16    = float32ToInt16(chunk);
  ws.send(int16.buffer);

  scheduleInputAudio(chunk);
  inputViz.feed(int16);
  updateAudioQuality(int16);
  const sent = end - playhead;
  playhead    += sent;
  samplesSent += sent;
  updateProgress();
}

// Send SILENCE_MS of silence so pipecat VAD detects end-of-speech
function startSilenceBurst() {
  const totalSamples = Math.round(SAMPLE_RATE * SILENCE_MS / 1000);
  let   sent         = 0;
  const silence      = new Int16Array(CHUNK_SAMPLES);  // zeros

  silenceTimer = setInterval(() => {
    if (sent >= totalSamples || !ws || ws.readyState !== WebSocket.OPEN) {
      clearInterval(silenceTimer);
      silenceTimer = null;
      return;
    }
    ws.send(silence.buffer);
    sent += CHUNK_SAMPLES;
  }, CHUNK_MS);
}

// ── Progress UI ────────────────────────────────────────────────────────────
function updateProgress() {
  if (!pcmBuffer) return;
  const pct      = Math.min(1, playhead / pcmBuffer.length);
  const current  = playhead / SAMPLE_RATE;
  const total    = pcmBuffer.length / SAMPLE_RATE;
  progressFill().style.width = `${(pct * 100).toFixed(2)}%`;
  timeDisplay().textContent  = `${formatTime(current)} / ${formatTime(total)}`;
}

// ── End-of-conversation detection ─────────────────────────────────────────
function onWavPlaybackComplete() {
  wavPlaybackDone    = true;
  botStartedAfterEOF = false;
  // If no bot speech starts within EOC_NO_RESPONSE_MS, consider it done.
  if (eocTimer) clearTimeout(eocTimer);
  eocTimer = setTimeout(() => {
    eocTimer = null;
    if (wavPlaybackDone && !botStartedAfterEOF) {
      log('ℹ No agent response after file end — conversation complete', 'info');
      wavPlaybackDone = false;
    }
  }, EOC_NO_RESPONSE_MS);
}

function resetEocState() {
  wavPlaybackDone    = false;
  botStartedAfterEOF = false;
  if (eocTimer) { clearTimeout(eocTimer); eocTimer = null; }
}

// ── PCM conversion helpers ─────────────────────────────────────────────────
function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}

function int16ToFloat32(int16) {
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    f32[i] = int16[i] / 32768.0;
  }
  return f32;
}

// ── Visualizer ─────────────────────────────────────────────────────────────
class BarVisualizer {
  constructor(canvasId, color) {
    this.canvas  = document.getElementById(canvasId);
    this.ctx     = this.canvas.getContext('2d');
    this.color   = color;
    this.n       = VIZ_BARS;
    this.heights = new Float32Array(this.n);   // smoothed bar heights [0..1]
    this.targets = new Float32Array(this.n);   // target bar heights
    this.rafId   = null;
    this._resize();
    this._startLoop();

    // Keep canvas size in sync
    new ResizeObserver(() => this._resize()).observe(this.canvas.parentElement);
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const w = parent.clientWidth - 32;
    const h = 70;
    this.canvas.width  = Math.max(w, 60);
    this.canvas.height = h;
  }

  /** Feed a chunk of Int16 audio samples to update bar heights. */
  feed(int16Data) {
    const rms = computeRms(int16Data);

    // Spread energy across bars with some random variation for visual interest
    for (let i = 0; i < this.n; i++) {
      const rand  = 0.5 + Math.random() * 0.6;          // 0.5 – 1.1
      const shape = Math.sin((i / this.n) * Math.PI);   // bell shape
      this.targets[i] = Math.min(1, rms * rand * 1.4 * (0.4 + shape * 0.6));
    }
  }

  _startLoop() {
    const SMOOTH_UP   = 0.35;
    const SMOOTH_DOWN = 0.12;
    const DECAY       = 0.88;

    const tick = () => {
      // Smooth heights toward targets, then decay targets
      for (let i = 0; i < this.n; i++) {
        const diff   = this.targets[i] - this.heights[i];
        const alpha  = diff > 0 ? SMOOTH_UP : SMOOTH_DOWN;
        this.heights[i] += diff * alpha;
        this.targets[i] *= DECAY;
      }
      this._draw();
      this.rafId = requestAnimationFrame(tick);
    };
    tick();
  }

  _draw() {
    const { canvas, ctx, heights, color, n } = this;
    const W = canvas.width, H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const gap = 2;
    const bw  = (W - gap * (n - 1)) / n;

    for (let i = 0; i < n; i++) {
      const h = Math.max(2, heights[i] * (H - 4));
      const x = i * (bw + gap);
      const y = H - h;

      // Gradient from accent colour (top) to dimmer (bottom)
      const grad = ctx.createLinearGradient(0, y, 0, H);
      grad.addColorStop(0, color);
      grad.addColorStop(1, color.replace(')', ', 0.25)').replace('rgb', 'rgba'));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x, y, bw, h, Math.min(bw / 2, 3));
      ctx.fill();
    }
  }
}

function computeRms(int16Data) {
  if (!int16Data.length) return 0;
  let sum = 0;
  for (let i = 0; i < int16Data.length; i++) {
    const s = int16Data[i] / 32768;
    sum += s * s;
  }
  return Math.sqrt(sum / int16Data.length);
}

// ── Utility helpers ────────────────────────────────────────────────────────
function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatBytes(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Metric info tooltip ────────────────────────────────────────────────────

const METRIC_INFO = {
  rtt: {
    label: 'Round-Trip Latency (RTT)',
    formula: 'T(user stops speaking) → T(bot first audio)',
    description: 'Total end-to-end pipeline delay from the moment the user finishes speaking to the moment the bot produces its first audio sample.',
    why: 'Drives perceived conversational naturalness. Human turn-taking typically expects a response within 200–800 ms. Beyond 1.5 s, users perceive the system as "thinking too long" and may speak again.',
    ranges: [
      { label: 'Excellent', cls: 'good',    value: '< 800 ms' },
      { label: 'Acceptable',cls: 'warn',    value: '800 ms – 1.5 s' },
      { label: 'Slow',      cls: 'bad',     value: '> 1.5 s' },
    ],
    note: 'RTT ≈ STT + LLM + TTS latencies. Optimise the slowest stage first.',
  },
  stt: {
    label: 'Speech-to-Text Latency',
    formula: 'T(user stops speaking) → T(transcript received)',
    description: 'Time for the STT service (Deepgram) to return a transcript after the user stops speaking. Includes network round-trip to the STT cloud API.',
    why: 'High STT latency is directly added to RTT. It also delays VAD-triggered LLM calls, making the whole pipeline feel sluggish.',
    ranges: [
      { label: 'Excellent', cls: 'good',    value: '< 250 ms' },
      { label: 'Acceptable',cls: 'warn',    value: '250 – 600 ms' },
      { label: 'Slow',      cls: 'bad',     value: '> 600 ms' },
    ],
    note: 'Streaming STT can reduce this significantly by returning partial transcripts before the user finishes.',
  },
  llm: {
    label: 'LLM Time-to-First-Token',
    formula: 'T(transcript received) → T(first LLM token)',
    description: 'Time from when the full transcript is sent to the LLM until the first response token is streamed back. Does not include full generation time.',
    why: 'TTFB determines when TTS can begin; minimising it shortens RTT. A fast TTFB makes the bot feel responsive even if full generation takes longer.',
    ranges: [
      { label: 'Excellent', cls: 'good',    value: '< 300 ms' },
      { label: 'Acceptable',cls: 'warn',    value: '300 ms – 800 ms' },
      { label: 'Slow',      cls: 'bad',     value: '> 800 ms' },
    ],
    note: 'Smaller/quantised models have lower TTFB. Gemini Flash is optimised for streaming.',
  },
  tts: {
    label: 'TTS Time-to-First-Byte (TTFB)',
    formula: 'T(LLM response complete) → T(first audio byte)',
    description: 'Time from when the LLM finishes generating its full response to when the TTS service (Cartesia) returns the first audio chunk.',
    why: 'TTS latency adds directly to the end of the pipeline. Low TTFB makes the bot voice feel "instant"; high TTFB causes an awkward silence after thinking.',
    ranges: [
      { label: 'Excellent', cls: 'good',    value: '< 150 ms' },
      { label: 'Acceptable',cls: 'warn',    value: '150 – 400 ms' },
      { label: 'Slow',      cls: 'bad',     value: '> 400 ms' },
    ],
    note: 'Neural TTS can be pipelined — some services start synthesising before LLM finishes, reducing perceived latency.',
  },
  snr: {
    label: 'Input Signal-to-Noise Ratio',
    formula: '20 × log₁₀(speech RMS / noise RMS)',
    description: 'Ratio of speech signal power to background noise power, measured in decibels. Speech RMS is accumulated while VAD detects the user talking; noise RMS is accumulated during silence.',
    why: 'Low SNR increases STT word-error rate and can fool the VAD into missing speech boundaries or firing false positives. Poor SNR is the single biggest cause of mis-transcriptions.',
    ranges: [
      { label: 'Clean',     cls: 'good',    value: '> 20 dB' },
      { label: 'Moderate',  cls: 'warn',    value: '10 – 20 dB' },
      { label: 'Noisy',     cls: 'bad',     value: '< 10 dB' },
    ],
    note: 'Telephone audio typically achieves 15–25 dB. Studio recordings exceed 40 dB.',
  },
  noise: {
    label: 'Noise Floor',
    formula: 'Mean RMS amplitude during silence (non-speech) segments',
    description: 'Average root-mean-square amplitude of the audio signal when no speech is detected. Measured as a normalised float (0 = silence, 1 = full scale).',
    why: 'A high noise floor degrades both VAD accuracy and STT transcription quality. It can also cause the VAD to remain in "speech" state continuously, preventing turn boundaries from firing.',
    ranges: [
      { label: 'Clean',     cls: 'good',    value: '< 0.01' },
      { label: 'Moderate',  cls: 'warn',    value: '0.01 – 0.05' },
      { label: 'Noisy',     cls: 'bad',     value: '> 0.05' },
    ],
    note: 'Typical phone line noise floor is 0.005–0.02. Air-conditioning or crowds can push it above 0.05.',
  },
  interrupts: {
    label: 'Interruptions',
    formula: 'Count of turns where user speech overlaps bot audio',
    description: 'Number of times the user started speaking while the bot was still producing audio. Detected when a VAD speech-start event arrives while the bot-speaking flag is active.',
    why: 'Frequent interruptions are a signal of conversation friction — the bot may be giving overly long responses, the user may be frustrated, or barge-in handling is not working correctly.',
    ranges: [
      { label: 'Normal',    cls: 'good',    value: '0 – 1 per session' },
      { label: 'Elevated',  cls: 'warn',    value: '2 – 4 per session' },
      { label: 'High',      cls: 'bad',     value: '5 + per session' },
    ],
    note: 'Some interruptions are natural. Zero is not always ideal — it may mean users are passively listening rather than engaging.',
  },
  'stt-conf': {
    label: 'STT Word Confidence',
    formula: 'mean(word.confidence) across all words in utterance',
    description: 'Average per-word confidence score returned by Deepgram. Each word in the transcript is assigned a confidence value between 0 and 1 by the acoustic model.',
    why: 'Low confidence correlates directly with higher word-error rate. Words below ~0.6 are likely mis-transcribed, which corrupts the LLM prompt and degrades response quality.',
    ranges: [
      { label: 'High',     cls: 'good', value: '≥ 90%' },
      { label: 'Moderate', cls: 'warn', value: '70 – 89%' },
      { label: 'Low',      cls: 'bad',  value: '< 70%' },
    ],
    note: 'Confidence is acoustic-model certainty, not semantic correctness. A confidently wrong word (e.g. "flight" → "fright") will still score high.',
  },
  wpm: {
    label: 'Input Speaking Rate',
    formula: 'word_count / (last_word.end − first_word.start) × 60',
    description: 'Words per minute of the caller\'s speech, computed from Deepgram word-level timestamps. Only covers the voiced portion of each utterance, not pauses.',
    why: 'Very fast speech (> 200 WPM) increases STT error rate and may overwhelm the LLM with run-on text. Very slow speech (< 80 WPM) can suggest hesitation, poor audio, or non-native speaker difficulty.',
    ranges: [
      { label: 'Normal',  cls: 'good', value: '100 – 180 wpm' },
      { label: 'Fast',    cls: 'warn', value: '180 – 220 wpm' },
      { label: 'Too fast',cls: 'bad',  value: '> 220 wpm' },
      { label: 'Slow',    cls: 'warn', value: '< 80 wpm' },
    ],
    note: 'Average conversational English is ~130 wpm. Phone support calls typically run 120–160 wpm.',
  },
  clip: {
    label: 'Input Clipping Rate',
    formula: 'samples at ±32767 / total samples × 100%',
    description: 'Fraction of PCM Int16 audio samples that hit the ADC ceiling (±32767). Clipping is hard saturation — the waveform is truncated, creating harmonic distortion that no DSP can recover.',
    why: 'Even 0.1% clipping is audible as crackling and significantly degrades STT accuracy, especially for fricatives (s, f, sh). It usually indicates gain staging problems at the recording device.',
    ranges: [
      { label: 'Clean',    cls: 'good', value: '< 0.1%' },
      { label: 'Moderate', cls: 'warn', value: '0.1 – 1%' },
      { label: 'Severe',   cls: 'bad',  value: '> 1%' },
    ],
    note: 'Clipping is irreversible. Fix at the source: lower microphone gain or use a compressor before ADC.',
  },
  'response-len': {
    label: 'Average Bot Response Length',
    formula: 'mean word count of bot_text across all turns',
    description: 'Average number of words in the bot\'s spoken responses. Longer responses take more time to synthesise and hear, increasing effective RTT from the caller\'s perspective.',
    why: 'Voice UX research shows callers lose attention after ~30 words (~10 seconds at 180 wpm). Concise answers improve satisfaction; verbose answers increase barge-in and repeat requests.',
    ranges: [
      { label: 'Concise',  cls: 'good', value: '≤ 30 words' },
      { label: 'Moderate', cls: 'warn', value: '31 – 60 words' },
      { label: 'Verbose',  cls: 'bad',  value: '> 60 words' },
    ],
    note: 'Prompt-tune the LLM with "Keep responses under 2 sentences" to target the concise range.',
  },
  quality: {
    label: 'Average Conversation Quality',
    formula: 'Mean score (1–5) from LLM judge across all turns',
    description: 'Each completed user↔bot exchange is scored 1–5 by a Gemini judge evaluating helpfulness, relevance, and conciseness. The score is averaged across all judged turns.',
    why: 'Latency metrics measure speed but not correctness. The quality score catches issues like off-topic answers, repeated clarification loops, unhelpful responses, and overly verbose replies.',
    ranges: [
      { label: 'Excellent', cls: 'good',    value: '4.5 – 5.0' },
      { label: 'Good',      cls: 'good',    value: '3.5 – 4.4' },
      { label: 'Acceptable',cls: 'warn',    value: '2.5 – 3.4' },
      { label: 'Poor',      cls: 'bad',     value: '< 2.5' },
    ],
    note: 'LLM-as-judge is a heuristic. Cross-validate against human evaluation for critical deployments.',
  },
  'filter-latency': {
    label: 'Filter Processing Latency',
    formula: 'avg wall-clock time per 10 ms audio chunk (active filter)',
    description: 'Average time the active audio filter (DeepFilterNet or Quail) takes to process each 10 ms audio chunk. This adds directly to audio pipeline latency.',
    why: 'If filter latency exceeds the chunk duration (~10 ms), the audio pipeline falls behind real-time. Compared across runs, it shows the overhead cost of noise suppression.',
    ranges: [
      { label: 'Real-time safe', cls: 'good', value: '< 10 ms' },
      { label: 'Marginal',       cls: 'warn', value: '10 – 30 ms' },
      { label: 'Too slow',       cls: 'bad',  value: '> 30 ms' },
    ],
    note: 'Only populated when a filter (DeepFilterNet or Quail) is active.',
  },
  'filter-noise-db': {
    label: 'DeepFilterNet Noise Reduction',
    formula: '20 × log₁₀(RMS_in / RMS_out) averaged over all processed chunks',
    description: 'Average dB reduction in signal amplitude caused by DeepFilterNet. Positive values mean the filter attenuated audio; higher dB = more aggressive suppression. Not available for Quail.',
    why: 'Shows how much noise was removed. Very high values during speech may indicate over-suppression (musical noise, artifacts). Compare STT confidence with/without filter to verify improvement.',
    ranges: [
      { label: 'Good suppression', cls: 'good', value: '3 – 15 dB' },
      { label: 'Mild',             cls: '',      value: '0 – 3 dB' },
      { label: 'Over-suppressing', cls: 'warn',  value: '> 15 dB' },
    ],
    note: 'Attenuation limit (atten_lim_db) caps this value. Set a lower limit to preserve voice naturalness.',
  },
};

let _tooltipAnchor = null;  // the ⓘ button that opened the current tooltip

function showMetricInfo(btn, metric) {
  const info = METRIC_INFO[metric];
  if (!info) return;

  const tooltip = document.getElementById('metric-tooltip');

  // If clicking the same open button → close
  if (_tooltipAnchor === btn && tooltip.classList.contains('visible')) {
    hideMetricInfo();
    return;
  }

  // Mark active button
  if (_tooltipAnchor) _tooltipAnchor.classList.remove('active');
  _tooltipAnchor = btn;
  btn.classList.add('active');

  // Build ranges rows
  const rangeRows = info.ranges.map(r => `
    <tr>
      <td><span class="tt-range-dot tt-range-${r.cls}"></span>${r.label}</td>
      <td>${r.value}</td>
    </tr>`).join('');

  tooltip.innerHTML = `
    <div class="tt-header">
      <span class="tt-title">${escHtml(info.label)}</span>
      <button class="tt-close" onclick="hideMetricInfo()" aria-label="Close">✕</button>
    </div>
    <div class="tt-formula">${escHtml(info.formula)}</div>
    <div class="tt-section">
      <div class="tt-section-label">What it measures</div>
      <div class="tt-section-body">${escHtml(info.description)}</div>
    </div>
    <div class="tt-section">
      <div class="tt-section-label">Why it matters</div>
      <div class="tt-section-body">${escHtml(info.why)}</div>
    </div>
    <div class="tt-section">
      <div class="tt-section-label">Reference ranges</div>
      <table class="tt-ranges">${rangeRows}</table>
    </div>
    ${info.note ? `<div class="tt-section-body" style="font-size:11px;color:var(--text-muted);border-top:1px solid var(--border);padding-top:8px;margin-top:4px">💡 ${escHtml(info.note)}</div>` : ''}`;

  // Position: below the button, clamped inside viewport
  tooltip.classList.add('visible');
  tooltip.setAttribute('aria-hidden', 'false');

  const btnRect = btn.getBoundingClientRect();
  const ttW = tooltip.offsetWidth;
  const ttH = tooltip.offsetHeight;
  const margin = 8;

  let left = btnRect.right - ttW;          // right-align with button
  let top  = btnRect.bottom + margin;      // below the button

  // Clamp to viewport
  left = Math.max(margin, Math.min(left, window.innerWidth  - ttW - margin));
  top  = Math.max(margin, Math.min(top,  window.innerHeight - ttH - margin));

  // If tooltip would overflow below, show above instead
  if (top + ttH > window.innerHeight - margin) {
    top = btnRect.top - ttH - margin;
  }

  tooltip.style.left = left + 'px';
  tooltip.style.top  = top  + 'px';
}

function hideMetricInfo() {
  const tooltip = document.getElementById('metric-tooltip');
  tooltip.classList.remove('visible');
  tooltip.setAttribute('aria-hidden', 'true');
  if (_tooltipAnchor) {
    _tooltipAnchor.classList.remove('active');
    _tooltipAnchor = null;
  }
}

// Close tooltip on outside click
document.addEventListener('click', (ev) => {
  const tooltip = document.getElementById('metric-tooltip');
  if (!tooltip || !tooltip.classList.contains('visible')) return;
  if (!tooltip.contains(ev.target) && !ev.target.closest('.kpi-info-btn')) {
    hideMetricInfo();
  }
}, true);

// ── Copy metrics to clipboard ──────────────────────────────────────────────
function copyMetricsJson() {
  const btn = document.getElementById('copy-metrics-btn');
  const s = _metricsSnapshot || {};

  // Compute live audio quality from client-side accumulators
  const speechRms = speechRmsCount > 0 ? speechRmsSum / speechRmsCount : null;
  const noiseRms  = noiseRmsCount  > 0 ? noiseRmsSum  / noiseRmsCount  : null;
  const snrDb     = (speechRms && noiseRms) ? +(20 * Math.log10(speechRms / noiseRms)).toFixed(1) : null;
  const clipPct   = totalSamplesSent > 0 ? +(clipCount / totalSamplesSent * 100).toFixed(4) : null;

  // Compute avg quality from turn history
  const scoredTurns = _turnHistory.filter(t => t.quality && t.quality.score > 0);
  const avgQuality  = scoredTurns.length
    ? +(scoredTurns.reduce((a, t) => a + t.quality.score, 0) / scoredTurns.length).toFixed(2)
    : null;

  const payload = {
    exported_at: new Date().toISOString(),
    file: selectedFile ? selectedFile.name : null,
    session: {
      turn_count:            s.turn_count            ?? 0,
      total_interruptions:   s.total_interruptions   ?? 0,
      rtt_ms:                { mean: s.rtt_mean,  p50: s.rtt_p50,  p99: s.rtt_p99  },
      stt_ms:                { mean: s.stt_mean,  p50: s.stt_p50,  p99: s.stt_p99  },
      llm_ms:                { mean: s.llm_mean,  p50: s.llm_p50,  p99: s.llm_p99  },
      tts_ms:                { mean: s.tts_mean,  p50: s.tts_p50,  p99: s.tts_p99  },
      avg_stt_confidence:    s.avg_stt_confidence    ?? null,
      avg_speaking_rate_wpm: s.avg_speaking_rate_wpm ?? null,
      avg_bot_words:         s.avg_bot_word_count     ?? null,
      avg_quality:           avgQuality,
      snr_db:                snrDb,
      noise_floor:           noiseRms != null ? +noiseRms.toFixed(5) : null,
      clipping_pct:          clipPct,
    },
    turns: _turnHistory.map(t => ({ ...t })),  // shallow copy
  };

  const stem = selectedFile ? selectedFile.name.replace(/\.wav$/i, '') : 'metrics';
  const ts   = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const filename = `${stem}_${ts}.json`;

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename;
  a.click();
  URL.revokeObjectURL(url);

  const orig = btn.textContent;
  btn.textContent = '✓ Saved!';
  btn.style.color = 'var(--green)';
  setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 2000);
}

// ── Snapshot comparison ─────────────────────────────────────────────────────

function takeSnapshot() {
  if (!_metricsSnapshot) { log('No metrics yet — play through at least one turn first.', 'warn'); return; }

  const idx        = _snapshots.length;
  const label      = idx === 0 ? 'A' : 'B';
  const filterName = activeFilter;

  // Compute live audio quality from current client-side accumulators
  const speechRms = speechRmsCount > 0 ? speechRmsSum / speechRmsCount : null;
  const noiseRms  = noiseRmsCount  > 0 ? noiseRmsSum  / noiseRmsCount  : null;
  const snrDb     = (speechRms && noiseRms) ? +(20 * Math.log10(speechRms / noiseRms)).toFixed(1) : null;
  const clipPct   = totalSamplesSent > 0 ? +(clipCount / totalSamplesSent * 100).toFixed(3) : null;

  const scoredTurns = _turnHistory.filter(t => t.quality && t.quality.score > 0);
  const avgQuality  = scoredTurns.length
    ? +(scoredTurns.reduce((a, t) => a + t.quality.score, 0) / scoredTurns.length).toFixed(2)
    : null;

  const snap = {
    label,
    filterName,
    session:    { ..._metricsSnapshot, avg_quality: avgQuality },
    snrDb,
    noiseFloor: noiseRms != null ? +noiseRms.toFixed(5) : null,
    clipPct,
    turnCount:  _metricsSnapshot.turn_count ?? 0,
    file:       selectedFile ? selectedFile.name : null,
  };

  if (idx < 2) {
    _snapshots.push(snap);
  } else {
    // Cycle: drop A, promote B → A, new snap → B
    _snapshots = [_snapshots[1], snap];
  }

  renderComparison();
  const btn = document.getElementById('snapshot-btn');
  const orig = btn.textContent;
  btn.textContent = `✓ Saved ${label}`;
  btn.style.color = 'var(--green)';
  setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 1500);
  const filterLabel = filterName === 'none' ? 'No filter' : filterName === 'deepfilter' ? 'DeepFilterNet' : 'Quail';
  log(`Snapshot ${label} saved (${filterLabel}, ${snap.turnCount} turns)`, 'ok');
}

function clearSnapshots() {
  _snapshots = [];
  document.getElementById('compare-panel').style.display = 'none';
  log('Snapshots cleared', 'info');
}

function renderComparison() {
  const panel = document.getElementById('compare-panel');
  if (!_snapshots.length) { panel.style.display = 'none'; return; }
  panel.style.display = '';

  const a = _snapshots[0];
  const b = _snapshots[1] || null;

  // Update column headers
  const fmtFilterLabel = (name) => name === 'none' ? 'No filter' : name === 'deepfilter' ? 'DeepFilterNet' : 'Quail';
  document.getElementById('cmp-head-a').textContent =
    `${a.label}: ${fmtFilterLabel(a.filterName)} · ${a.turnCount}t` +
    (a.file ? ` · ${a.file}` : '');
  document.getElementById('cmp-head-b').textContent = b
    ? `${b.label}: ${fmtFilterLabel(b.filterName)} · ${b.turnCount}t` +
      (b.file ? ` · ${b.file}` : '')
    : '— take Snapshot B —';

  const rows = [
    { label: 'STT Confidence',   va: fmtConf(a.session.avg_stt_confidence),             vb: b ? fmtConf(b.session.avg_stt_confidence)             : null, da: a.session.avg_stt_confidence,             db: b?.session.avg_stt_confidence,             higherBetter: true,  fmt: v => fmtConf(v) },
    { label: 'STT Latency avg',  va: fmtMs(a.session.stt_mean),                         vb: b ? fmtMs(b.session.stt_mean)                         : null, da: a.session.stt_mean,                       db: b?.session.stt_mean,                       higherBetter: false, fmt: v => fmtMs(v) },
    { label: 'STT Latency p50',  va: fmtMs(a.session.stt_p50),                          vb: b ? fmtMs(b.session.stt_p50)                          : null, da: a.session.stt_p50,                        db: b?.session.stt_p50,                        higherBetter: false, fmt: v => fmtMs(v) },
    { label: 'Round-Trip avg',   va: fmtMs(a.session.rtt_mean),                         vb: b ? fmtMs(b.session.rtt_mean)                         : null, da: a.session.rtt_mean,                       db: b?.session.rtt_mean,                       higherBetter: false, fmt: v => fmtMs(v) },
    { label: 'Round-Trip p50',   va: fmtMs(a.session.rtt_p50),                          vb: b ? fmtMs(b.session.rtt_p50)                          : null, da: a.session.rtt_p50,                        db: b?.session.rtt_p50,                        higherBetter: false, fmt: v => fmtMs(v) },
    { label: 'LLM Latency avg',  va: fmtMs(a.session.llm_mean),                         vb: b ? fmtMs(b.session.llm_mean)                         : null, da: a.session.llm_mean,                       db: b?.session.llm_mean,                       higherBetter: false, fmt: v => fmtMs(v) },
    { label: 'Input SNR',        va: a.snrDb    != null ? a.snrDb + ' dB'      : '—',  vb: b ? (b.snrDb    != null ? b.snrDb + ' dB'      : '—') : null, da: a.snrDb,                                  db: b?.snrDb,                                  higherBetter: true,  fmt: v => v.toFixed(1) + ' dB' },
    { label: 'Noise Floor',      va: a.noiseFloor != null ? a.noiseFloor.toFixed(4) : '—', vb: b ? (b.noiseFloor != null ? b.noiseFloor.toFixed(4) : '—') : null, da: a.noiseFloor,                      db: b?.noiseFloor,                             higherBetter: false, fmt: v => v.toFixed(4) },
    { label: 'Clipping Rate',    va: a.clipPct  != null ? a.clipPct + '%'      : '—',  vb: b ? (b.clipPct  != null ? b.clipPct + '%'      : '—') : null, da: a.clipPct,                                db: b?.clipPct,                                higherBetter: false, fmt: v => v.toFixed(3) + '%' },
    { label: 'LLM Quality',      va: a.session.avg_quality != null ? a.session.avg_quality.toFixed(2) + ' ★' : '—', vb: b ? (b.session.avg_quality != null ? b.session.avg_quality.toFixed(2) + ' ★' : '—') : null, da: a.session.avg_quality, db: b?.session.avg_quality, higherBetter: true, fmt: v => v.toFixed(2) + ' ★' },
  ];

  const tbody = document.getElementById('compare-tbody');
  tbody.innerHTML = rows.map(r => {
    let deltaCell = '<td class="cmp-delta">—</td>';
    if (b && r.da != null && r.db != null) {
      const raw   = r.db - r.da;
      const good  = r.higherBetter ? raw > 0 : raw < 0;
      const bad   = r.higherBetter ? raw < 0 : raw > 0;
      const cls   = good ? 'cmp-good' : (bad ? 'cmp-bad' : '');
      const sign  = raw > 0 ? '+' : '';
      deltaCell   = `<td class="cmp-delta ${cls}">${sign}${r.fmt(raw)}</td>`;
    } else if (!b) {
      deltaCell = '<td class="cmp-delta cmp-muted">await B</td>';
    }
    return `<tr>
      <td class="cmp-label">${r.label}</td>
      <td class="cmp-val">${r.va ?? '—'}</td>
      <td class="cmp-val">${r.vb ?? '—'}</td>
      ${deltaCell}
    </tr>`;
  }).join('');
}

// ── Keyboard shortcut ──────────────────────────────────────────────────────
document.addEventListener('keydown', (ev) => {
  if (ev.code === 'Space' && ev.target === document.body) {
    ev.preventDefault();
    togglePlay();
  }
});

// ── Init ───────────────────────────────────────────────────────────────────
// Instantiate visualizers (after DOM is ready)
let inputViz, outputViz;

document.addEventListener('DOMContentLoaded', () => {
  inputViz  = new BarVisualizer('input-viz',  'rgb(62, 207, 142)');
  outputViz = new BarVisualizer('output-viz', 'rgb(59, 158, 255)');

  // Delegate ⓘ button clicks
  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('.kpi-info-btn');
    if (btn) {
      ev.stopPropagation();
      showMetricInfo(btn, btn.dataset.metric);
    }
  });

  loadFileList();
  log('Ready. Click Connect to start.', 'ok');
});
