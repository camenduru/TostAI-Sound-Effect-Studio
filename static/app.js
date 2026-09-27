/* TostAI Sound Effect Studio — front-end for the MOSS-SoundEffect v2.0 web app. */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ────────────────────────── theme ────────────────────────── */

// The palette itself lives in styles.css: :root is light and
// html[data-theme="dark"] overrides the same variable names. The inline script
// in index.html has already picked one before the first paint, so setTheme()
// here only has to keep the button honest and re-read the few colours the
// canvas needs (a canvas cannot use CSS variables).
const THEME_KEY = 'tostai.sfx.theme';
// A canvas cannot use CSS variables, so the one accent it draws with is read
// back out of the stylesheet whenever the theme changes. One colour, not a
// gradient: this UI is flat, and so is the visualiser.
const themeColor = { acc: [14, 116, 144] };

function hexToRgb(value) {
  const hex = (value || '').trim().replace('#', '');
  if (hex.length === 3) {
    return [0, 1, 2].map((i) => parseInt(hex[i] + hex[i], 16));
  }
  if (hex.length >= 6) {
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  }
  return null;
}

function rgba(triple, alpha) {
  return `rgba(${triple[0]}, ${triple[1]}, ${triple[2]}, ${alpha})`;
}

function refreshThemeColors() {
  const rgb = hexToRgb(getComputedStyle(document.documentElement).getPropertyValue('--acc'));
  if (rgb && rgb.every((n) => Number.isFinite(n))) themeColor.acc = rgb;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (err) {
    /* private mode: the choice just will not persist */
  }
  refreshThemeColors();
  const button = $('#theme-toggle');
  // The label names the theme it switches TO, so "Dark" on a light page can
  // never be read as "the page is dark".
  button.textContent = theme === 'dark' ? 'Light' : 'Dark';
  button.title = theme === 'dark' ? 'switch to the light theme' : 'switch to the dark theme';
}

function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
}

const state = {
  catalog: null,
  lang: 'en',
  batch: false,
  busy: false,
  abort: null,
  audio: null, // { float32, sampleRate }
  loop: false,
  outputs: [],
  activeOutput: null,
  progressTimer: null,
};

// One audio element for the whole shelf: starting a take must stop the previous
// one, and a per-row element cannot know about the others.
let shelfAudio = null;
let shelfPlaying = null;

/* ────────────────────────── audio engine ────────────────────────── */

const engine = {
  ctx: null,
  analyser: null,
  master: null,
  sources: new Set(),
  buffer: null,
  playing: false,
  loop: false,
  startCtxTime: 0,
  startOffset: 0,
  pauseOffset: 0,
  current: null,
  sampleRate: 48000,
  requested: null,
};

function ensureCtx(sampleRate) {
  if (engine.ctx && (!sampleRate || engine.requested === sampleRate)) {
    return engine.ctx;
  }
  if (engine.ctx) engine.ctx.close();
  let ctx;
  try {
    ctx = sampleRate ? new AudioContext({ sampleRate }) : new AudioContext();
  } catch (err) {
    ctx = new AudioContext();
  }
  engine.requested = sampleRate || ctx.sampleRate;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.72;
  const master = ctx.createGain();
  master.gain.value = 0.92;
  master.connect(analyser);
  analyser.connect(ctx.destination);
  engine.ctx = ctx;
  engine.analyser = analyser;
  engine.master = master;
  engine.sampleRate = ctx.sampleRate;
  engine.sources = new Set();
  return ctx;
}

function resample(input, from, to) {
  if (from === to) return input;
  const ratio = to / from;
  const length = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i / ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = pos - lo;
    output[i] = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return output;
}

function stopSources() {
  for (const source of engine.sources) {
    try {
      source.onended = null;
      source.stop();
    } catch (err) {
      /* already stopped */
    }
  }
  engine.sources.clear();
}

function resetEngine() {
  stopSources();
  engine.buffer = null;
  engine.playing = false;
  engine.pauseOffset = 0;
  engine.current = null;
}

function makeBuffer(float32, sampleRate) {
  const ctx = ensureCtx(sampleRate);
  const data = resample(float32, sampleRate, ctx.sampleRate);
  const buffer = ctx.createBuffer(1, data.length, ctx.sampleRate);
  buffer.copyToChannel(data, 0);
  return buffer;
}

function playBuffer(offset) {
  if (!engine.buffer) return;
  const ctx = ensureCtx();
  ctx.resume();
  stopSources();
  const start = Math.min(Math.max(0, offset || 0), Math.max(0, engine.buffer.duration - 0.02));
  const source = ctx.createBufferSource();
  source.buffer = engine.buffer;
  source.loop = engine.loop;
  source.connect(engine.master);
  source.onended = () => {
    engine.sources.delete(source);
    if (engine.current === source && !engine.loop) {
      engine.playing = false;
      engine.pauseOffset = 0;
      engine.current = null;
      syncTransport();
    }
  };
  source.start(0, start);
  engine.sources.add(source);
  engine.current = source;
  engine.playing = true;
  engine.startCtxTime = ctx.currentTime;
  engine.startOffset = start;
  syncTransport();
}

function pauseBuffer() {
  if (!engine.playing) return;
  engine.pauseOffset = position();
  const source = engine.current;
  engine.playing = false;
  engine.current = null;
  if (source) {
    try {
      source.onended = null;
      source.stop();
    } catch (err) {
      /* ignore */
    }
    engine.sources.delete(source);
  }
  syncTransport();
}

function position() {
  if (!engine.ctx || !engine.buffer) return engine.pauseOffset || 0;
  if (engine.playing && engine.current) {
    const elapsed = engine.ctx.currentTime - engine.startCtxTime;
    if (engine.loop) {
      return (engine.startOffset + elapsed) % engine.buffer.duration;
    }
    return Math.min(engine.buffer.duration, engine.startOffset + elapsed);
  }
  return Math.min(engine.pauseOffset || 0, engine.buffer.duration);
}

/* ────────────────────────── encoding helpers ────────────────────────── */

function encodeWav(float32, sampleRate) {
  const length = float32.length;
  const view = new DataView(new ArrayBuffer(44 + length * 2));
  const write = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view.buffer], { type: 'audio/wav' });
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function download(source, name) {
  // A saved take is already a URL on the server; a fresh one is still a Blob.
  const objectUrl = typeof source === 'string';
  const url = objectUrl ? source : URL.createObjectURL(source);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (!objectUrl) setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/* ────────────────────────── catalog rendering ────────────────────────── */

function renderQuickChips() {
  const container = $('#quick-chips');
  container.innerHTML = '';
  const prompts = (state.catalog.quick_prompts || {})[state.lang] || [];
  for (const item of prompts) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip-btn';
    button.textContent = item.label;
    button.title = item.prompt;
    button.addEventListener('click', () => {
      $('#prompt').value = item.prompt;
      updateCounter();
    });
    container.appendChild(button);
  }
}

function renderPresetChips() {
  const container = $('#preset-chips');
  container.innerHTML = '';
  const presets = (state.catalog.sound_presets || {})[state.lang] || [];
  for (const preset of presets) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip-btn';
    button.textContent = preset.label;
    button.title = preset.prompt;
    button.addEventListener('click', () => {
      if (state.batch) {
        appendBatchLine(preset.prompt);
      } else {
        $('#prompt').value = preset.prompt;
        updateCounter();
      }
    });
    container.appendChild(button);
  }
}

function renderNegativeChips() {
  const container = $('#negative-chips');
  container.innerHTML = '';
  for (const preset of state.catalog.negative_presets || []) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip-btn';
    button.textContent = preset.label;
    button.title = preset.prompt;
    button.addEventListener('click', () => {
      $('#negative').value = preset.prompt;
    });
    container.appendChild(button);
  }
}

function renderParamDocs() {
  const list = $('#param-docs');
  list.innerHTML = '';
  for (const doc of state.catalog.param_docs || []) {
    const li = document.createElement('li');
    const b = document.createElement('b');
    b.textContent = doc.param;
    li.appendChild(b);
    li.appendChild(document.createTextNode(' — ' + doc.detail));
    list.appendChild(li);
  }
}

function renderFacts() {
  const model = state.catalog.model;
  const body = $('#facts-body');
  const rows = [
    ['Model', model.name],
    ['Architecture', model.architecture],
    ['DiT variant', model.dit_variant],
    ['Sample rate', `${model.sample_rate} Hz`],
    ['Channels', `${model.channels} (mono)`],
    ['Max duration', `${model.max_seconds} s`],
    ['Languages', model.languages.join(', ')],
    ['Device', model.device],
    ['Model dir', model.model_dir],
    ['License', model.license],
  ];
  body.innerHTML = rows
    .map(([k, v]) => `<div class="fact-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span></div>`)
    .join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* ────────────────────────── mode / language ────────────────────────── */

function setBatch(batch) {
  state.batch = batch;
  $('#batch-field').hidden = !batch;
  $('#prompt').parentElement.classList.toggle('is-dimmed', batch);
  $('#mode-toggle').textContent = batch ? 'switch to single' : 'switch to batch';
  updateBatchCount();
}

function appendBatchLine(text) {
  const field = $('#batch');
  const lines = field.value.split('\n').filter((l) => l.trim());
  if (!lines.some((l) => l === text)) lines.push(text);
  field.value = lines.join('\n');
  updateBatchCount();
}

function updateBatchCount() {
  const lines = $('#batch').value.split('\n').filter((l) => l.trim());
  $('#batch-count').textContent = `${lines.length} caption${lines.length === 1 ? '' : 's'}`;
}

function selectLang(lang) {
  state.lang = lang;
  $$('.lang-btn').forEach((btn) => btn.classList.toggle('is-active', btn.dataset.lang === lang));
  renderQuickChips();
  renderPresetChips();
  $('#prompt').placeholder = lang === 'zh'
    ? '描述你想要的声音效果——环境、生物、动作……'
    : 'Describe the sound effect — environment, creature, action…';
  updateCounter();
}

function updateCounter() {
  const text = $('#prompt').value;
  $('#prompt-count').textContent = `${text.length} chars`;
}

/* ────────────────────────── visualizer ────────────────────────── */

const viz = { canvas: null, ctx: null, data: null };

function initViz() {
  viz.canvas = $('#viz');
  viz.ctx = viz.canvas.getContext('2d');
  resizeViz();
  window.addEventListener('resize', resizeViz);
  requestAnimationFrame(drawViz);
}

function resizeViz() {
  const ratio = window.devicePixelRatio || 1;
  viz.canvas.width = viz.canvas.clientWidth * ratio;
  viz.canvas.height = viz.canvas.clientHeight * ratio;
  viz.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawViz() {
  requestAnimationFrame(drawViz);
  const ctx = viz.ctx;
  if (!ctx) return;
  const width = viz.canvas.clientWidth;
  const height = viz.canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);

  const active = engine.playing || (state.busy && engine.analyser);
  let values = null;
  if (engine.analyser && active) {
    if (!viz.data || viz.data.length !== engine.analyser.frequencyBinCount) {
      viz.data = new Uint8Array(engine.analyser.frequencyBinCount);
    }
    engine.analyser.getByteFrequencyData(viz.data);
    values = viz.data;
  }

  const bars = 72;
  const gap = 3;
  const barWidth = Math.max(2, (width - gap * (bars - 1)) / bars);
  const baseY = height * 0.55;
  ctx.fillStyle = rgba(themeColor.acc, values ? 0.9 : 0.4);
  for (let i = 0; i < bars; i++) {
    let amplitude;
    if (values) {
      const index = Math.floor((i / bars) * values.length * 0.7);
      amplitude = (values[index] / 255) * height * 0.72;
    } else {
      const wave = Math.sin(i * 0.32 + performance.now() / 1400) * 0.5 + 0.5;
      amplitude = 3 + wave * 6;
    }
    amplitude = Math.max(2, amplitude);
    const x = i * (barWidth + gap);
    const radius = Math.min(barWidth / 2, 4);
    roundRect(ctx, x, baseY - amplitude, barWidth, amplitude * 2, radius);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fill();
}

/* ────────────────────────── transport UI ────────────────────────── */

function syncTransport() {
  const hasBuffer = Boolean(engine.buffer);
  const duration = hasBuffer ? engine.buffer.duration : 0;
  const pos = hasBuffer && !state.busy ? position() : 0;

  $('#play').disabled = !hasBuffer || state.busy;
  $('#play').textContent = engine.playing ? '❚❚' : '▶';
  $('#scrubber').disabled = !hasBuffer || state.busy;
  $('#loop').disabled = !hasBuffer;
  $('#loop').classList.toggle('is-on', engine.loop);
  $('#download-wav').disabled = !state.audio;

  $('#time-total').textContent = formatTime(duration);
  $('#time-current').textContent = formatTime(pos);
  if (duration > 0 && !engine.loop) {
    $('#scrubber').value = String(Math.round((pos / duration) * 1000));
  } else if (duration > 0 && engine.loop) {
    $('#scrubber').value = String(Math.round((pos / duration) * 1000));
  } else {
    $('#scrubber').value = '0';
  }
  $('#stage-overlay').hidden = hasBuffer || state.busy;
}

function setStats({ total, duration, steps }) {
  $('#stat-total').textContent = Number.isFinite(total) ? `${(total / 1000).toFixed(2)} s` : '—';
  $('#stat-rtf').textContent =
    duration > 0 && Number.isFinite(total) ? `${(total / 1000 / duration).toFixed(2)}×` : '—';
  $('#stat-length').textContent = duration > 0 ? `${duration.toFixed(2)} s` : '—';
  $('#stat-steps').textContent = Number.isFinite(steps) ? `${steps}` : '—';
}

function showError(message) {
  const el = $('#error');
  el.textContent = message;
  el.hidden = false;
}

function clearError() {
  $('#error').hidden = true;
}

function setBusy(busy) {
  state.busy = busy;
  $('#generate').disabled = busy;
  $('#spinner').hidden = !busy;
  $('#cancel').hidden = !busy;
  $('#generate-label').textContent = busy ? 'Generating…' : 'Generate sound';
  syncTransport();
}

/* ────────────────────────── progress ────────────────────────── */

function startProgressPolling() {
  stopProgressPolling();
  $('#progress-wrap').hidden = false;
  state.progressTimer = setInterval(async () => {
    try {
      const data = await (await fetch('/api/generate/progress', { cache: 'no-store' })).json();
      if (!data.active) return;
      const pct = data.steps_total ? Math.round((data.steps_completed / data.steps_total) * 100) : 0;
      $('#progress').value = pct;
      $('#progress-text').textContent = `step ${data.steps_completed} / ${data.steps_total}`;
      $('#progress-eta').textContent = data.eta_s != null ? `eta ${Math.round(data.eta_s)}s` : '';
    } catch (err) {
      /* the poll is cosmetic; generation errors surface on the request itself */
    }
  }, 500);
}

function stopProgressPolling(done) {
  if (state.progressTimer) {
    clearInterval(state.progressTimer);
    state.progressTimer = null;
  }
  if (done) {
    $('#progress').value = 100;
    setTimeout(() => {
      $('#progress-wrap').hidden = true;
      $('#progress').value = 0;
    }, 600);
  } else {
    $('#progress-wrap').hidden = true;
    $('#progress').value = 0;
  }
}

/* ────────────────────────── generation ────────────────────────── */

function buildForm() {
  const form = new FormData();
  if (state.batch) {
    const lines = $('#batch').value.split('\n').map((l) => l.trim()).filter(Boolean);
    if (!lines.length) return null;
    form.append('prompts', lines.join('\n'));
  } else {
    const text = $('#prompt').value.trim();
    if (!text) return null;
    form.append('prompts', text);
  }
  form.append('negative_prompt', $('#negative').value.trim());
  form.append('seconds', $('#seconds').value);
  form.append('steps', $('#steps').value);
  form.append('cfg_scale', $('#cfg').value);
  form.append('sigma_shift', $('#sigma').value);
  form.append('seed', String(parseInt($('#seed').value, 10) || 0));
  form.append('duration_tag', $('#duration-tag').checked ? '1' : '0');
  form.append('save', $('#keep-take').checked ? '1' : '0');
  return form;
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || `Request failed (${response.status})`;
  } catch (err) {
    return `Request failed (${response.status})`;
  }
}

async function generate() {
  if (state.busy) return;
  clearError();

  const form = buildForm();
  if (!form) return showError('Describe the sound you want first.');
  const controller = new AbortController();
  state.abort = controller;

  resetEngine();
  setBusy(true);
  $('#meta-demo').hidden = true;
  $('#meta-mode').textContent = state.batch ? 'batch' : 'sfx';
  $('#meta-params').textContent = `${$('#seconds').value}s · ${$('#steps').value} steps · cfg ${$('#cfg').value} · seed ${$('#seed').value}`;
  setStats({ total: NaN, duration: 0, steps: NaN });
  startProgressPolling();

  const startedAt = performance.now();
  try {
    if (state.batch) {
      await runBatch(form, controller, startedAt);
    } else {
      await runSingle(form, controller, startedAt);
    }
    await loadOutputs();
  } catch (err) {
    if (err.name === 'AbortError') {
      showError('Generation cancelled.');
    } else {
      showError(err.message || String(err));
    }
  } finally {
    state.abort = null;
    setBusy(false);
    stopProgressPolling(false);
    refreshStatus();
  }
}

function metaFromResponse(response) {
  const demo = response.headers.get('X-SoundEffect-Demo') === '1';
  const warning = response.headers.get('X-SoundEffect-Warning') || '';
  $('#meta-demo').hidden = !demo;
  if (demo) {
    $('#meta-demo').title = warning || 'Served by the built-in demo synth because the model is offline.';
  }
  return demo;
}

async function runSingle(form, controller, startedAt) {
  const response = await fetch('/api/generate', {
    method: 'POST',
    body: form,
    signal: controller.signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  metaFromResponse(response);

  const sampleRate = parseInt(response.headers.get('X-SoundEffect-Sample-Rate') || '48000', 10) || 48000;
  const steps = parseInt(response.headers.get('X-SoundEffect-Steps-Done') || '0', 10);
  const raw = await response.arrayBuffer();
  const ctx = ensureCtx(sampleRate);
  const buffer = await ctx.decodeAudioData(raw.slice(0));
  const float32 = buffer.getChannelData(0).slice();
  const total = performance.now() - startedAt;

  state.audio = { float32, sampleRate: buffer.sampleRate };
  engine.buffer = makeBuffer(float32, buffer.sampleRate);
  engine.pauseOffset = 0;
  stopProgressPolling(true);
  setStats({ total, duration: buffer.duration, steps });
  playBuffer(0);
}

async function runBatch(form, controller, startedAt) {
  const response = await fetch('/api/generate', {
    method: 'POST',
    body: form,
    signal: controller.signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  metaFromResponse(response);

  const payload = await response.json();
  const sampleRate = payload.sample_rate || 48000;
  const total = performance.now() - startedAt;
  // Play the first take; every take is already on the shelf with its own
  // download link, which is the natural place to audition the rest.
  if (!payload.wavs_b64 || !payload.wavs_b64.length) {
    throw new Error('The batch returned no audio.');
  }
  const binary = atob(payload.wavs_b64[0]);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const ctx = ensureCtx(sampleRate);
  const buffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
  const float32 = buffer.getChannelData(0).slice();

  state.audio = { float32, sampleRate: buffer.sampleRate };
  engine.buffer = makeBuffer(float32, buffer.sampleRate);
  engine.pauseOffset = 0;
  stopProgressPolling(true);
  setStats({ total, duration: buffer.duration, steps: null });
  playBuffer(0);
}

/* ────────────────────────── output shelf ────────────────────────── */

// The shelf is a VIEW OF THE OUTPUT FOLDER, not a second copy of it. Every take
// the server finishes is written to disk with a JSON sidecar, so re-reading the
// list is both the refresh and the persistence -- no client-side bookkeeping to
// get out of step with what is actually on disk.

async function loadOutputs() {
  try {
    const data = await (await fetch('/api/outputs')).json();
    state.outputs = data.outputs || [];
    const dir = data.dir || 'outputs/';
    $('#outputs-dir').textContent = `saving to ${dir}`;
    $('#outputs-dir').title = dir;
  } catch (err) {
    state.outputs = [];
    $('#outputs-dir').textContent = 'output folder unavailable';
  }
  renderHistory();
}

function renderHistory() {
  const container = $('#history');
  container.innerHTML = '';
  if (!state.outputs.length) {
    container.innerHTML =
      '<p class="empty">Nothing saved yet — every take lands in the output folder and appears here.</p>';
    return;
  }
  for (const output of state.outputs) {
    const el = document.createElement('div');
    el.className = 'take' + (state.activeOutput === output.name ? ' is-playing' : '');
    el.innerHTML = `
      <button class="take-play" type="button" aria-label="Play take">▶</button>
      <div class="take-info">
        <div class="take-text"></div>
        <div class="take-sub">
          <span>${output.batch_index != null ? `#${output.batch_index + 1}` : (output.mode || 'take')}</span>
          <span>${Number(output.seconds || 0).toFixed(1)}s · ${Number(output.duration || 0).toFixed(2)}s</span>
          <span>seed ${output.seed ?? '—'}</span>
          ${output.demo ? '<span class="tag demo">demo</span>' : ''}
        </div>
      </div>
      <div class="take-actions">
        <button class="ghost small act-download" type="button">WAV</button>
        <button class="ghost small act-reuse" type="button">Reuse</button>
        <button class="ghost small act-delete" type="button" title="delete this take">✕</button>
      </div>`;
    el.querySelector('.take-text').textContent = output.prompt || output.name;

    el.querySelector('.take-play').addEventListener('click', () => toggleTake(output));
    el.querySelector('.act-download').addEventListener('click', () =>
      download(output.url, output.name)
    );
    el.querySelector('.act-reuse').addEventListener('click', () => {
      if (output.prompt) {
        if (state.batch) appendBatchLine(output.prompt);
        else $('#prompt').value = output.prompt;
      }
      if (output.negative_prompt) $('#negative').value = output.negative_prompt;
      if (output.seconds != null) $('#seconds').value = output.seconds;
      if (output.steps != null) $('#steps').value = output.steps;
      if (output.cfg_scale != null) $('#cfg').value = output.cfg_scale;
      if (output.sigma_shift != null) $('#sigma').value = output.sigma_shift;
      if (output.seed != null) $('#seed').value = output.seed;
      syncParamLabels();
      updateCounter();
      updateBatchCount();
    });
    el.querySelector('.act-delete').addEventListener('click', () => deleteOutput(output));
    container.appendChild(el);
  }
}

function toggleTake(output) {
  const alreadyPlaying = shelfPlaying && shelfPlaying.name === output.name;
  if (shelfAudio) {
    shelfAudio.onended = null;
    shelfAudio.pause();
  }
  shelfAudio = null;
  shelfPlaying = null;
  state.activeOutput = null;
  if (alreadyPlaying) {
    renderHistory();
    return;
  }
  pauseBuffer();
  shelfAudio = new Audio(output.url);
  shelfPlaying = output;
  state.activeOutput = output.name;
  shelfAudio.onended = () => {
    shelfAudio = null;
    shelfPlaying = null;
    state.activeOutput = null;
    renderHistory();
  };
  shelfAudio.play();
  renderHistory();
}

async function deleteOutput(output) {
  if (!window.confirm(`Delete ${output.name} from the output folder?`)) return;
  try {
    await fetch(`/api/outputs/${encodeURIComponent(output.name)}`, { method: 'DELETE' });
  } catch (err) {
    showError(`Could not delete ${output.name}.`);
  }
  await loadOutputs();
}

/* ────────────────────────── status ────────────────────────── */

// The indicator is a bare dot, so the wording it stands for goes into the
// tooltip and the accessible name instead of onto the page.
function setStatus(label, detail) {
  const badge = $('#status-badge');
  $('#status-text').textContent = label;
  badge.title = detail;
  badge.setAttribute('aria-label', detail);
}

async function refreshStatus() {
  const badge = $('#status-badge');
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    if (data.connected) {
      badge.className = 'badge compact is-online';
      setStatus(
        'model online',
        `Model ready (${data.detail || 'ready'}). Click to re-check.`
      );
    } else if (data.engine === 'loading') {
      badge.className = 'badge compact is-loading';
      setStatus('model loading', `Loading the pipeline: ${data.detail || '…'}`);
    } else if (data.fallback) {
      badge.className = 'badge compact is-demo';
      setStatus(
        'demo audio, model offline',
        `No model at ${data.model_dir} (${data.detail}), so takes are demo audio, ` +
          'not the model. Click to re-check.'
      );
    } else {
      badge.className = 'badge compact is-offline';
      setStatus(
        'model offline',
        `No model at ${data.model_dir} (${data.detail}). Generation will fail. Click to re-check.`
      );
    }
    $('#sample-rate-chip').textContent = `${(data.sample_rate / 1000).toFixed(1)} kHz · mono`;
  } catch (err) {
    badge.className = 'badge compact is-offline';
    setStatus('app server unreachable', 'The app server is unreachable. Click to re-check.');
  }
}

/* ────────────────────────── update ────────────────────────── */

// Pulls the latest source out of this app's own repository, then restarts the
// server so the new Python is actually loaded. The repository is public, so
// the update needs no credential -- the button just fetches.

function updateMsg(html) {
  $('#update-msg').innerHTML = html;
}

function openUpdate() {
  updateMsg('');
  $('#update-go').disabled = false;
  $('#update-rev').textContent = 'checking the installed revision…';
  fetch('/api/update')
    .then((r) => r.json())
    .then((d) => {
      if (d.repo) {
        $('#update-src').textContent = d.repo;
        $('#update-src').href = 'https://github.com/' + d.repo;
      }
      $('#update-rev').textContent = d.rev
        ? `installed ${d.short || d.rev}`
        : 'installed revision unknown (this image carries no .tostai_rev)';
    })
    .catch(() => {
      $('#update-rev').textContent = '';
    });
  $('#update-dialog').showModal();
  $('#update-go').focus();
}

function closeUpdate() {
  $('#update-dialog').close();
}

// Wait until the NEW revision is the one answering. Waiting for "any response"
// is not enough: the re-exec is delayed so the update's own response can flush,
// so for the first second or so the OLD process still answers, and a naive
// readiness check would reload straight back into the old code.
async function waitForRev(rev) {
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const r = await fetch('/api/update', { cache: 'no-store' });
      if (!r.ok) continue;
      const d = await r.json();
      if (d.rev === rev) return true;
    } catch (err) {
      /* still down */
    }
  }
  return false;
}

async function runUpdate() {
  $('#update-go').disabled = true;
  updateMsg('<div class="hint">Fetching the latest source…</div>');
  let d;
  try {
    const r = await fetch('/api/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    // Read the body as text and parse it here, rather than calling r.json(). A
    // failure before the handler runs answers with a bare text/plain "Internal
    // Server Error", and r.json() then throws a parser complaint instead of the
    // actual cause.
    const raw = await r.text();
    try {
      d = JSON.parse(raw);
    } catch (err) {
      d = {
        ok: false,
        error: `HTTP ${r.status} from the server: ${raw.slice(0, 400)}`,
      };
    }
  } catch (err) {
    updateMsg(`<div class="err">${escapeHtml(String(err))}</div>`);
    $('#update-go').disabled = false;
    return;
  }
  if (!d.ok) {
    updateMsg(`<div class="err">${escapeHtml(d.error || 'update failed')}</div>`);
    $('#update-go').disabled = false;
    return;
  }
  if (!d.updated) {
    updateMsg(`<div class="okbox">Already up to date at ${escapeHtml(d.rev.slice(0, 10))}.</div>`);
    $('#update-go').disabled = false;
    return;
  }
  updateMsg(
    `<div class="hint">Updated to ${escapeHtml(d.rev.slice(0, 10))} (${d.files} files). Restarting…</div>`
  );
  if (!(await waitForRev(d.rev))) {
    updateMsg(
      '<div class="err">The server did not come back on the new revision.\n' +
        'Check it with:  docker logs <container></div>'
    );
    $('#update-go').disabled = false;
    return;
  }
  location.reload();
}

/* ────────────────────────── wiring ────────────────────────── */

function syncParamLabels() {
  $('#seconds-value').textContent = `${Number($('#seconds').value).toFixed(1)} s`;
  $('#steps-value').textContent = $('#steps').value;
  $('#cfg-value').textContent = Number($('#cfg').value).toFixed(1);
  $('#sigma-value').textContent = Number($('#sigma').value).toFixed(1);
}

function wireEvents() {
  $('#generate').addEventListener('click', generate);
  $('#cancel').addEventListener('click', () => state.abort && state.abort.abort());
  $('#reset').addEventListener('click', () => {
    clearError();
    $('#prompt').value = '';
    $('#batch').value = '';
    $('#negative').value = '';
    $('#seed').value = '0';
    $('#seconds').value = '10';
    $('#steps').value = '100';
    $('#cfg').value = '4';
    $('#sigma').value = '5';
    $('#duration-tag').checked = true;
    $('#keep-take').checked = true;
    syncParamLabels();
    updateCounter();
    updateBatchCount();
  });

  $('#prompt').addEventListener('input', updateCounter);
  $('#prompt').addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') generate();
  });
  $('#batch').addEventListener('input', updateBatchCount);
  $('#batch').addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') generate();
  });

  $('#mode-toggle').addEventListener('click', () => setBatch(!state.batch));
  $$('.lang-btn').forEach((btn) =>
    btn.addEventListener('click', () => selectLang(btn.dataset.lang))
  );

  ['seconds', 'steps', 'cfg', 'sigma'].forEach((id) =>
    $('#' + id).addEventListener('input', syncParamLabels)
  );
  $('#seed-random').addEventListener('click', () => {
    $('#seed').value = String(Math.floor(Math.random() * 1000000));
  });

  $('#play').addEventListener('click', () => {
    if (engine.playing) pauseBuffer();
    else playBuffer(engine.pauseOffset || 0);
  });
  $('#scrubber').addEventListener('input', (event) => {
    if (!engine.buffer || state.busy) return;
    const ratio = Number(event.target.value) / 1000;
    const target = ratio * engine.buffer.duration;
    if (engine.playing) playBuffer(target);
    else {
      engine.pauseOffset = target;
      syncTransport();
    }
  });
  $('#loop').addEventListener('click', () => {
    engine.loop = !engine.loop;
    if (engine.current) engine.current.loop = engine.loop;
    syncTransport();
  });

  $('#download-wav').addEventListener('click', () => {
    if (state.audio) {
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      download(encodeWav(state.audio.float32, state.audio.sampleRate), `sfx-${stamp}.wav`);
    }
  });

  $('#refresh-outputs').addEventListener('click', loadOutputs);

  $('#status-badge').addEventListener('click', refreshStatus);
  $('#theme-toggle').addEventListener('click', toggleTheme);
  $('#facts-button').addEventListener('click', () => $('#facts-dialog').showModal());
  $('#facts-close').addEventListener('click', () => $('#facts-dialog').close());
  $('#update-button').addEventListener('click', openUpdate);
  $('#update-close').addEventListener('click', closeUpdate);
  $('#update-cancel').addEventListener('click', closeUpdate);
  $('#update-go').addEventListener('click', runUpdate);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && engine.playing) pauseBuffer();
  });

  setInterval(() => {
    if (engine.playing && !state.busy) syncTransport();
  }, 250);
  setInterval(() => {
    if (!state.busy) refreshStatus();
  }, 20000);
}

/* ────────────────────────── boot ────────────────────────── */

async function boot() {
  setTheme(document.documentElement.dataset.theme || 'light');
  initViz();
  wireEvents();
  try {
    state.catalog = await (await fetch('/api/catalog')).json();
  } catch (err) {
    showError('Could not load the catalog from the app server.');
    return;
  }
  const ranges = state.catalog.param_ranges || {};
  if (ranges.seconds) {
    $('#seconds').min = ranges.seconds.min;
    $('#seconds').max = ranges.seconds.max;
    $('#seconds').step = ranges.seconds.step;
    $('#seconds').value = ranges.seconds.default;
  }
  renderParamDocs();
  renderNegativeChips();
  renderFacts();
  selectLang('en');
  setBatch(false);
  syncParamLabels();
  updateCounter();
  updateBatchCount();
  syncTransport();
  await loadOutputs();
  refreshStatus();
}

boot();
