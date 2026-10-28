// Deckray: the deckray Python package runs in Pyodide, embeddings run in transformers.js, and optional LLM calls
// go straight from this tab to the provider the user picked. Every ingestion and question is a visible pipeline run.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
env.allowLocalModels = false;

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
const C = t => `var(--${t})`;
const now = () => performance.now();
const tick = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));  // let the UI paint between steps
const kb = n => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const MAX = 20 * 1024 * 1024;
const SAMPLES = ['Solar Program Q3 Review.pptx', 'Solar Sites Directory.xlsx'];
const EXAMPLES = ['Which sites in Spain are offline?', 'How much did the carport sites generate in August?',
  'What should the presenter stress about the risks?', 'Which panel vendor has the longest warranty?', 'What are the Q4 priorities?'];
const PROVIDERS = {
  ollama: { label: 'Ollama', base: 'http://localhost:11434/v1', key: false, hint: 'Allow this page in Ollama: <code>OLLAMA_ORIGINS=' + location.origin + ' ollama serve</code>' },
  openai: { label: 'OpenAI', base: 'https://api.openai.com/v1', key: true, hint: 'The key is sent only to api.openai.com.' },
  anthropic: { label: 'Anthropic', base: 'https://api.anthropic.com/v1', key: true, hint: 'The key is sent only to api.anthropic.com.' },
  groq: { label: 'Groq', base: 'https://api.groq.com/openai/v1', key: true, hint: 'Free keys at console.groq.com. Sent only to api.groq.com.' },
  openrouter: { label: 'OpenRouter', base: 'https://openrouter.ai/api/v1', key: true, hint: 'One key for many models. Sent only to openrouter.ai.' },
};

const state = { docs: {}, chunks: [], vecs: [], view: null, mode: 'structure', tab: 'inspect', hl: null, tfilter: null, runs: [], answers: [] };
let py, embedder, cfg = loadCfg();

function toast(msg) { const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; document.body.append(t); setTimeout(() => t.remove(), 4800); }
const lastLine = e => String(e?.message || e).split('\n').filter(Boolean).pop();

// ---------- pipeline runs ----------
function startRun(kind, title) {
  const run = { id: crypto.randomUUID(), kind, title, t0: now(), steps: [], status: 'run' };
  state.runs.unshift(run); renderRuns(run.id); return run;
}
function step(run, name, detail = '') { const s = { name, detail, status: 'run', t0: now() }; run.steps.push(s); renderRuns(run.id); return s; }
function finish(run, s, status = 'done', detail) { s.status = status; s.ms = now() - s.t0; if (detail !== undefined) s.detail = detail; renderRuns(run.id); }
function endRun(run, status = 'done') { run.status = status; run.ms = now() - run.t0; renderRuns(run.id); }
const fmtMs = ms => ms == null ? '' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
const ICON = { done: '✓', fail: '✕', skip: '–', run: '' };

function renderRuns(focusId) {
  if (!state.runs.length) { $('runs').innerHTML = '<div class="empty">Every ingestion and every question runs as a pipeline. Its steps, timings and decisions appear here live.</div>'; return; }
  const open = new Set([...document.querySelectorAll('.run[open]')].map(d => d.dataset.id));
  $('runs').innerHTML = state.runs.map((r, i) => `<details class="run${r.id === focusId ? ' focus' : ''}" data-id="${r.id}" ${i === 0 || open.has(r.id) ? 'open' : ''}>
    <summary><span class="stt ${r.status}"></span><span class="k${r.kind === 'query' ? ' q' : ''}">${r.kind}</span><span class="t" title="${esc(r.title)}">${esc(r.title)}</span><span class="ms">${fmtMs(r.ms)}</span></summary>
    <ol class="steps">${r.steps.map(s => `<li class="st ${s.status}"><span class="ic">${ICON[s.status]}</span><span class="nm">${esc(s.name)}</span><span class="ms">${s.status === 'run' ? '' : fmtMs(s.ms)}</span>${s.detail ? `<div class="dt2">${s.detail}</div>` : ''}</li>`).join('')}</ol></details>`).join('');
  if (focusId) setTimeout(() => document.querySelector('.run.focus')?.classList.remove('focus'), 1200);
}

// ---------- python bridge ----------
window.llmPost = async (url, headers, body) => {
  const r = await fetch(url, { method: 'POST', headers: JSON.parse(headers), body });
  const text = await r.text();
  if (!r.ok) throw new Error(`${r.status} from ${new URL(url).host}: ${text.slice(0, 160)}`);
  return text;
};
const BRIDGE = `
import json, io, zipfile, base64, sys
sys.path.insert(0, '/pkg')
import deckray
from deckray import llm, chunks as ch
from js import llmPost
DOCS, BYTES, ZIPS = {}, {}, {}
async def post(url, headers, body):
    return json.loads(await llmPost(url, json.dumps(headers), json.dumps(body)))
def validate_json(name, data):
    b = bytes(data.to_py()); z, info = deckray.validate(name, b); BYTES[name], ZIPS[name] = b, z
    return json.dumps(info)
def extract_json(name):
    DOCS[name] = {'name': name, 'extraction': deckray.extract(name, BYTES[name], ZIPS[name])}
    return json.dumps(DOCS[name]['extraction'])
def chunk_json(name):
    d = DOCS[name]; d['chunks'] = ch.chunk(name, d['extraction']); d['naive'] = ch.naive_chunks(name, d['extraction'])
    return json.dumps({'chunks': d['chunks'], 'naive': d['naive']})
def remove(name):
    for m in (DOCS, BYTES, ZIPS): m.pop(name, None)
async def describe_images(name, cfg_json):
    cfg, ex, z, n, failed = json.loads(cfg_json), DOCS[name]['extraction'], ZIPS[name], 0, []
    for s in ex.get('slides', []):
        for img in s['images']:
            part = img.get('part')
            ext = (part or '').rsplit('.', 1)[-1].lower()
            if not part or part not in z.namelist() or ext not in ('png', 'jpg', 'jpeg', 'gif', 'webp'): continue
            try:
                img['description'] = await llm.describe_image(cfg, post, base64.b64encode(z.read(part)).decode(), 'image/jpeg' if ext == 'jpg' else 'image/' + ext); n += 1
            except Exception as e:
                failed.append(str(e)[:120])
    return json.dumps({'described': n, 'failed': failed, 'extraction': ex})
async def plan_json(cfg_json, q):
    src = ch.sources({n: d['extraction'] for n, d in DOCS.items()})
    return json.dumps(await llm.plan(json.loads(cfg_json), post, q, src))
def select_json(chunks_json, scores_json, plan_json):
    stats = {}
    ctx = ch.select_context(json.loads(chunks_json), json.loads(scores_json), json.loads(plan_json), stats=stats)
    return json.dumps({'context': ctx, 'stats': stats})
async def answer_json(cfg_json, q, ctx_json):
    return await llm.answer(json.loads(cfg_json), post, q, json.loads(ctx_json))
`;
const pycall = (fn, ...a) => py.globals.get(fn)(...a);

async function boot() {
  const run = startRun('boot', 'Start workspace');
  let s = step(run, 'Load Python runtime', 'Pyodide 0.29.5 (CPython compiled to WebAssembly)');
  try {
    py = await loadPyodide();
    finish(run, s, 'done', `Python ${py.runPython('import sys; sys.version.split()[0]')} in WebAssembly`);
    s = step(run, 'Install parser dependencies', '<code>openpyxl</code>, <code>defusedxml</code> from PyPI');
    await py.loadPackage('micropip'); await py.pyimport('micropip').install(['openpyxl', 'defusedxml']);
    finish(run, s);
    s = step(run, 'Load deckray package');
    py.FS.mkdirTree('/pkg/deckray');
    const files = ['__init__.py', 'pptx.py', 'xlsx.py', 'chunks.py', 'llm.py'];
    await Promise.all(files.map(async f => py.FS.writeFile('/pkg/deckray/' + f, await (await fetch('deckray/' + f)).text())));
    py.runPython(BRIDGE);
    finish(run, s, 'done', files.map(f => `<code>${f}</code>`).join(' '));
    endRun(run);
    $('sb-py-dot').className = 'dot ok'; $('sb-py').textContent = `Python ${py.runPython('import sys; sys.version.split()[0]')} · Pyodide`;
    $('addBtn').disabled = false; document.querySelectorAll('[data-act="samples"],[data-act="browse"]').forEach(b => b.disabled = false);
  } catch (e) {
    console.error(e); finish(run, s, 'fail', esc(lastLine(e))); endRun(run, 'fail');
    $('sb-py-dot').className = 'dot bad'; $('sb-py').textContent = 'Python failed to load';
  }
}

// ---------- ingestion pipeline ----------
function parseSummary(ex) {
  if (ex.kind === 'pptx') {
    const S = ex.slides, cnt = f => S.reduce((n, s) => n + f(s), 0);
    const pts = cnt(s => s.charts.reduce((n, c) => n + c.series.reduce((m, x) => m + x.points.length, 0), 0));
    const rows = cnt(s => s.tables.reduce((n, t) => n + t.rows.length, 0));
    return `${plural(S.length, 'slide')} in presentation order<ul>${[
      `${cnt(s => s.lines.length)} text lines with list levels`,
      cnt(s => s.tables.length) && `${plural(cnt(s => s.tables.length), 'table')} · ${rows} rows, merged cells resolved`,
      cnt(s => s.charts.length) && `${plural(cnt(s => s.charts.length), 'chart')} · ${pts} data points`,
      cnt(s => s.smartart.length) && `${cnt(s => s.smartart.length)} SmartArt tree(s)`,
      cnt(s => s.notes ? 1 : 0) && `speaker notes on ${plural(cnt(s => s.notes ? 1 : 0), 'slide')}`,
      cnt(s => s.links.length) && `${plural(cnt(s => s.links.length), 'hyperlink')}`,
      cnt(s => s.images.length) && `${plural(cnt(s => s.images.length), 'image')} · ${cnt(s => s.images.filter(i => i.alt).length)} with alt text`,
      cnt(s => s.hidden ? 1 : 0) && `${plural(cnt(s => s.hidden ? 1 : 0), 'hidden slide')} (still indexed)`,
    ].filter(Boolean).map(x => `<li>${x}</li>`).join('')}</ul>`;
  }
  return `${plural(ex.sheets.length, 'sheet')}<ul>${ex.sheets.map(s => {
    const unnamed = s.headers.filter(h => /^Column [A-Z]+$/.test(h)).length;
    return `<li><b>${esc(s.name)}</b>${s.hidden ? ' (hidden)' : ''}: header row ${s.header_row}, ${plural(s.rows.length, 'row')}${s.merged.length ? `, ${plural(s.merged.length, 'merged range')} unmerged` : ''}${unnamed ? `, ${unnamed} unnamed column kept in place` : ''}${s.description ? ', title rows → description' : ''}${s.truncated ? ', capped at 5,000×60' : ''}</li>`;
  }).join('')}</ul>`;
}
function chunkSummary(chunks, naive) {
  const by = {}; chunks.forEach(c => by[c.type] = (by[c.type] || 0) + 1);
  const tokens = chunks.reduce((n, c) => n + c.tokens, 0);
  return `${plural(chunks.length, 'structure-aware chunk')} (~${tokens.toLocaleString()} tokens): ${Object.entries(by).map(([t, n]) => `<code style="color:${C(t)}">${t} ${n}</code>`).join(' ')}<br>Naive baseline for comparison: ${plural(naive.length, 'fixed window')}`;
}

async function addFiles(files, label) {
  const run = startRun('ingest', label || files.map(f => f.name).join(', '));
  let ok = 0;
  for (const f of files) {
    const tag = `<code>${esc(f.name)}</code>`;
    let s = step(run, `Read ${f.name}`);
    if (!/\.(pptx|xlsx|xlsm)$/i.test(f.name)) { finish(run, s, 'fail', 'Only .pptx, .xlsx and .xlsm are supported (old .ppt/.xls are binary formats).'); continue; }
    if (f.size > MAX) { finish(run, s, 'fail', `${kb(f.size)} is over the 20 MB limit.`); continue; }
    const bytes = new Uint8Array(await f.arrayBuffer());
    finish(run, s, 'done', `${kb(f.size)} read into memory (not uploaded)`);
    if (state.docs[f.name]) removeDoc(f.name, true);
    try {
      s = step(run, 'Validate archive'); await tick();
      const info = JSON.parse(pycall('validate_json', f.name, bytes));
      finish(run, s, 'done', `OOXML zip: ${info.entries} parts, ${kb(info.unzipped)} uncompressed (limits ${info.limits.entries.toLocaleString()} parts / ${kb(info.limits.unzipped)}) · XML parsed with <code>defusedxml</code>`);
      s = step(run, info.kind === 'pptx' ? 'Parse slides' : 'Parse sheets'); await tick();
      let ex = JSON.parse(pycall('extract_json', f.name));
      finish(run, s, 'done', parseSummary(ex));
      const imgs = ex.kind === 'pptx' ? ex.slides.reduce((n, x) => n + x.images.length, 0) : 0;
      if (ex.kind === 'pptx' && imgs) {
        s = step(run, 'Describe images');
        if (cfg.provider && cfg.vision) {
          const r = JSON.parse(await pycall('describe_images', f.name, JSON.stringify(cfg)));
          ex = r.extraction;
          finish(run, s, r.described ? 'done' : 'fail', `${r.described}/${imgs} described by <code>${esc(cfg.model)}</code>${r.failed.length ? ` · ${esc(r.failed[0])}` : ''}`);
        } else finish(run, s, 'skip', cfg.provider ? 'Vision is off in model settings, using alt text' : 'No model set, using alt text');
      }
      s = step(run, 'Chunk'); await tick();
      const { chunks, naive } = JSON.parse(pycall('chunk_json', f.name));
      finish(run, s, 'done', chunkSummary(chunks, naive));
      state.docs[f.name] = { name: f.name, info, extraction: ex, chunks, naive };
      if (!state.view) state.view = firstView(f.name);
      ok++;
    } catch (e) { console.error(e); finish(run, s, 'fail', esc(lastLine(e))); }
  }
  rebuild(); render();
  if (ok) await embedAll(run);
  endRun(run, ok === files.length ? 'done' : ok ? 'done' : 'fail');
}

async function embedAll(run) {
  const todo = state.vecs.map((v, i) => v ? -1 : i).filter(i => i >= 0);
  if (!todo.length) return;
  let s = step(run, 'Embed chunks', 'all-MiniLM-L6-v2 (quantized ONNX) in this tab');
  try {
    if (!embedder) {
      $('sb-emb-dot').className = 'dot run'; $('sb-emb').textContent = 'Embeddings: downloading model…';
      embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
      $('sb-emb-dot').className = 'dot ok'; $('sb-emb').textContent = 'Embeddings: all-MiniLM-L6-v2 · 384-d';
    }
    for (let i = 0; i < todo.length; i += 16) {
      const batch = todo.slice(i, i + 16), vecs = await embed(batch.map(j => state.chunks[j].text));
      batch.forEach((j, k) => state.vecs[j] = vecs[k]);
      const done = Math.min(i + 16, todo.length);
      s.detail = `all-MiniLM-L6-v2 · ${done}/${todo.length}<div class="bar"><i style="width:${done / todo.length * 100}%"></i></div>`; renderRuns();
    }
    finish(run, s, 'done', `${plural(todo.length, 'vector')} · 384 dimensions · L2-normalized`);
    s = step(run, 'Index ready'); finish(run, s, 'done', `${plural(state.chunks.length, 'chunk')} searchable across ${plural(Object.keys(state.docs).length, 'source')}`);
    $('sb-idx').textContent = `${state.chunks.length} chunks indexed`; renderTable();
  } catch (e) { console.error(e); finish(run, s, 'fail', esc(lastLine(e))); $('sb-emb-dot').className = 'dot bad'; $('sb-emb').textContent = 'Embeddings: failed'; }
}
async function embed(texts) {
  const out = await embedder(texts, { pooling: 'mean', normalize: true });
  return Array.from({ length: texts.length }, (_, i) => out.data.slice(i * 384, (i + 1) * 384));
}

async function loadSamples() {
  document.querySelectorAll('[data-act="samples"]').forEach(b => b.disabled = true);
  const files = await Promise.all(SAMPLES.map(async n => new File([await (await fetch('samples/' + encodeURIComponent(n))).blob()], n)));
  await addFiles(files, 'Sample deck + workbook');
  $('examples').innerHTML = EXAMPLES.map(x => `<button class="ex" type="button">${esc(x)}</button>`).join('');
}
function removeDoc(name, quiet) {
  delete state.docs[name]; pycall('remove', name);
  if (state.view?.doc === name) state.view = Object.keys(state.docs).length ? firstView(Object.keys(state.docs)[0]) : null;
  if (!quiet) { rebuild(); render(); $('sb-idx').textContent = `${state.chunks.length} chunks indexed`; }
}
const firstView = doc => state.docs[doc].extraction.kind === 'pptx' ? { doc, slide: 1 } : { doc, sheet: state.docs[doc].extraction.sheets[0].name };
function rebuild() {
  const old = new Map(state.chunks.map((c, i) => [c.id + '\0' + c.text, state.vecs[i]]));
  state.chunks = Object.values(state.docs).flatMap(d => d.chunks);
  state.vecs = state.chunks.map(c => old.get(c.id + '\0' + c.text) || null);
  $('chunkCount').textContent = state.chunks.length;
  $('q').disabled = $('askBtn').disabled = !state.chunks.length;
}

// ---------- render: sources, inspect, chunks table ----------
function render() { renderTree(); renderCanvas(); renderInspector(); renderTable(); }

function renderTree() {
  const names = Object.keys(state.docs);
  $('tree').innerHTML = names.length ? names.map(n => {
    const d = state.docs[n], ex = d.extraction, pp = ex.kind === 'pptx';
    const items = pp
      ? ex.slides.map(s => `<button class="titem${state.view?.doc === n && state.view.slide === s.index ? ' on' : ''}" data-doc="${esc(n)}" data-slide="${s.index}"><span class="n">${s.index}</span><span class="lb">${esc(s.title || 'Untitled slide')}</span>${s.hidden ? '<span class="pill warn">hidden</span>' : ''}</button>`)
      : ex.sheets.map(s => `<button class="titem${state.view?.doc === n && state.view.sheet === s.name ? ' on' : ''}" data-doc="${esc(n)}" data-sheet="${esc(s.name)}"><span class="n">▦</span><span class="lb">${esc(s.name)}</span>${s.hidden ? '<span class="pill warn">hidden</span>' : `<span class="pill">${s.rows.length}</span>`}</button>`);
    return `<div class="tdoc"><div class="tdoc-h"><span class="ft" style="background:${pp ? '#d24726' : '#107c41'}">${pp ? 'PPT' : 'XLS'}</span><span class="nm" title="${esc(n)}">${esc(n)}</span><span class="pill">${d.chunks.length}</span><button class="icon-btn" title="Remove" data-remove="${esc(n)}">✕</button></div>${items.join('')}</div>`;
  }).join('') : '<div class="empty">No sources yet.</div>';
}

const el = (type, id, label, inner) => `<div class="el" style="--c:${C(type)}" data-chunk="${esc(id)}"><span class="tag">${label}</span>${inner}</div>`;
function slideHtml(doc, s) {
  const id = x => `${doc}#s${s.index}-${x}`, parts = [];
  if (s.lines.length) parts.push(el('text', id('text'), 'text', `<p class="lines">${esc(s.lines.join('\n'))}</p>`));
  s.tables.forEach((t, i) => parts.push(el('table', id('table' + (i + 1)), `table · ${t.rows.length} rows`,
    `<table class="xt"><tr>${t.headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr>${t.rows.map(r => `<tr>${t.headers.map(h => `<td>${esc(r[h])}</td>`).join('')}</tr>`).join('')}</table>`)));
  const hues = ['var(--chart)', 'var(--text)', 'var(--table)', 'var(--row)'];
  s.charts.forEach((c, i) => {
    const cats = [...new Set(c.series.flatMap(x => x.points.map(p => p.label)))], max = Math.max(...c.series.flatMap(x => x.points.map(p => p.value)), 1);
    const groups = cats.map(cat => `<div class="g"><div class="bs">${c.series.map((x, k) => { const p = x.points.find(p => p.label === cat); return p ? `<div class="b" style="height:${Math.max(2, p.value / max * 100)}%;background:${hues[k % 4]}"><span>${p.value}</span></div>` : ''; }).join('')}</div><label>${esc(cat)}</label></div>`).join('');
    parts.push(el('chart', id('chart' + (i + 1)), `${c.kind} chart · data extracted from XML`, `<b>${esc(c.title)}</b><div class="bars">${groups}</div><div class="legend">${c.series.map((x, k) => `<span><i style="background:${hues[k % 4]}"></i>${esc(x.name)}</span>`).join('')}</div>`));
  });
  const tree = ns => `<ul class="tree-ul">${ns.map(n => `<li>${esc(n.text)}${n.children.length ? tree(n.children) : ''}</li>`).join('')}</ul>`;
  s.smartart.forEach((t, i) => parts.push(el('smartart', id('smartart' + (i + 1)), 'smartart', tree(t))));
  if (s.images.length) parts.push(el('images', id('images'), plural(s.images.length, 'image'), s.images.map(im =>
    `<div class="alt"><b>img</b><span>${esc(im.description || im.alt || 'No alt text. Pick a vision model in settings to describe it.')} <em style="color:var(--faint)">${im.description ? '(described by model)' : im.alt ? '(alt text)' : ''}</em></span></div>`).join('')));
  if (s.links.length) parts.push(el('links', id('links'), 'links', s.links.map(l => `<div>↗ ${esc(l.text || 'link')} <span class="mono" style="color:var(--faint);font-size:11.5px">${esc(l.url)}</span></div>`).join('')));
  if (s.notes) parts.push(el('notes', id('notes'), 'speaker notes', `<div class="sticky">${esc(s.notes)}</div>`));
  return `<div class="slide">${s.title ? `<h2>${esc(s.title)}</h2>` : ''}${parts.join('') || '<p class="empty">Nothing extractable on this slide.</p>'}</div>`;
}
function sheetHtml(doc, s) {
  const letters = n => { let r = ''; for (n++; n; n = Math.floor((n - 1) / 26)) r = String.fromCharCode(65 + (n - 1) % 26) + r; return r; };
  const col = t => [...t].reduce((a, c) => a * 26 + c.charCodeAt(0) - 64, 0) - 1, merged = new Set();
  s.merged.forEach(r => { const m = r.match(/^([A-Z]+)(\d+):([A-Z]+)(\d+)$/); if (m) for (let a = +m[2]; a <= +m[4]; a++) for (let b = col(m[1]); b <= col(m[3]); b++) if (a !== +m[2] || b !== col(m[1])) merged.add(`${a},${b}`); });
  const width = Math.max(...s.preview.map(r => r.length), 1), data = new Set(s.rows.map(r => r.row));
  const body = s.preview.map((row, i) => {
    const rn = i + 1, cls = rn === s.header_row ? 'hdr' : rn < s.header_row ? 'desc' : data.has(rn) ? 'data' : '';
    return `<tr class="${cls}"${cls === 'data' ? ` data-chunk="${esc(`${doc}#${s.name}!${rn}`)}"` : ''}><td class="rn">${rn}</td>${Array.from({ length: width }, (_, c) => `<td class="${merged.has(`${rn},${c}`) ? 'm' : ''}" title="${esc(row[c])}">${esc(row[c])}</td>`).join('')}</tr>`;
  }).join('');
  return `<div class="gridwrap"><table class="grid"><thead><tr><th></th>${Array.from({ length: width }, (_, c) => `<th>${letters(c)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
    <div class="key"><span><i style="background:var(--acc-soft)"></i>detected header</span><span><i></i><em>title rows → description</em></span><span><i style="background-image:repeating-linear-gradient(135deg,transparent 0 3px,color-mix(in oklab,var(--table) 30%,transparent) 3px 5px)"></i>filled from a merged cell</span><span>Click a row to see its chunk${s.preview.length < s.rows.length + s.header_row ? ` · preview shows the first ${s.preview.length} rows` : ''}</span></div>`;
}
function renderCanvas() {
  const v = state.view;
  if (!v || !state.docs[v.doc]) {
    $('canvas').innerHTML = `<div class="welcome"><span class="badge">🔒 Runs entirely in your browser</span><h1>See inside your slides and spreadsheets</h1>
      <p>Deckray turns PowerPoint and Excel files into structure-aware chunks for RAG, keeping titles, bullet levels, table headers, merged cells, chart data and speaker notes, and shows every step of the pipeline as it runs.</p>
      <button class="btn primary" data-act="samples" ${py ? '' : 'disabled'}>Load sample deck + workbook</button> <button class="btn" data-act="browse" ${py ? '' : 'disabled'}>Open your own files…</button>
      <div class="flowline">${[['validate', 'naive'], ['parse', 'table'], ['chunk', 'text'], ['embed', 'chart'], ['plan', 'smartart'], ['retrieve', 'row'], ['answer', 'links']].map(([n, c]) => `<span style="--c:${C(c)}"><i></i>${n}</span>`).join('<em>→</em>')}</div></div>`;
    return;
  }
  const ex = state.docs[v.doc].extraction;
  if (ex.kind === 'pptx') {
    const s = ex.slides.find(x => x.index === v.slide) || ex.slides[0], n = state.docs[v.doc].chunks.filter(c => c.where.slide === s.index).length;
    $('canvas').innerHTML = `<div class="canvas-h"><b>Slide ${s.index}</b><span>${esc(v.doc)}</span><span class="chip">${plural(n, 'chunk')}</span>${s.hidden ? '<span class="chip warn">hidden slide, still indexed</span>' : ''}<span class="spacer"></span><span>Click an element to find its chunk</span></div>` + slideHtml(v.doc, s);
  } else {
    const s = ex.sheets.find(x => x.name === v.sheet) || ex.sheets[0], unnamed = s.headers.filter(h => /^Column [A-Z]+$/.test(h)).length;
    $('canvas').innerHTML = `<div class="canvas-h"><b>${esc(s.name)}</b><span>${esc(v.doc)}</span><span class="chip">header row ${s.header_row}</span><span class="chip">${plural(s.rows.length, 'row chunk')}</span>${s.merged.length ? `<span class="chip">${plural(s.merged.length, 'merged range')} unmerged</span>` : ''}${unnamed ? `<span class="chip">${unnamed} unnamed column kept in place</span>` : ''}${s.hidden ? '<span class="chip warn">hidden sheet, still indexed</span>' : ''}</div>` + sheetHtml(v.doc, s);
  }
  if (state.hl) highlight(state.hl, false);
}
function renderInspector() {
  const v = state.view, d = v && state.docs[v.doc];
  if (!d) { $('inspTitle').textContent = 'Chunks'; $('insp').innerHTML = '<div class="empty">Select a slide or sheet.</div>'; return; }
  const list = state.mode === 'naive' ? d.naive : d.chunks.filter(c => v.slide ? c.where.slide === v.slide : c.where.sheet === v.sheet);
  $('inspTitle').textContent = state.mode === 'naive' ? `Naive · ${list.length}` : `Chunks · ${list.length}`;
  $('insp').innerHTML = (state.mode === 'naive' ? `<div class="note">What a typical pipeline produces from this file: all text flattened into fixed windows. No slide titles or column names, and merged cells mostly empty. Compare with the structured chunks.</div>` : '')
    + `<div class="clist">${list.map(c => `<div class="ck" style="--c:${C(c.type)}" data-chunk="${esc(c.id)}"><header><b>${esc(c.type)}</b><span>~${c.tokens} tok</span></header><pre>${esc(c.text)}</pre></div>`).join('') || '<div class="empty">No chunks here.</div>'}</div>`;
}
function renderTable() {
  const types = {}; state.chunks.forEach(c => types[c.type] = (types[c.type] || 0) + 1);
  $('tfilters').innerHTML = Object.entries(types).map(([t, n]) => `<button class="fchip${state.tfilter === t ? ' on' : ''}" style="--c:${C(t)}" data-t="${t}"><i></i>${t} ${n}</button>`).join('');
  const q = $('tq').value.trim().toLowerCase();
  const rows = state.chunks.map((c, i) => [c, i]).filter(([c]) => (!state.tfilter || c.type === state.tfilter) && (!q || c.text.toLowerCase().includes(q)));
  $('tinfo').textContent = `${rows.length} of ${state.chunks.length}${rows.length > 400 ? ' · showing 400' : ''}`;
  $('tbody').innerHTML = rows.slice(0, 400).map(([c, i]) => `<tr data-i="${i}"><td><span class="ty" style="--c:${C(c.type)}"><i></i>${c.type}</span></td><td class="mono" style="font-size:11.5px;white-space:nowrap">${c.where.slide ? 'slide ' + c.where.slide : esc(c.where.sheet) + ' · row ' + c.where.row}</td><td style="color:var(--mute)">${esc(c.doc)}</td><td class="mono">${c.tokens}</td><td>${state.vecs[i] ? '<span style="color:var(--ok)">✓</span>' : '<span style="color:var(--faint)">…</span>'}</td><td class="pv">${esc(c.text.replace(/^\[[^\]]*\]\s*/, '').replace(/\n/g, ' · '))}</td></tr>`).join('')
    || `<tr><td colspan="6" class="empty">${state.chunks.length ? 'No chunks match.' : 'No sources yet.'}</td></tr>`;
}
function highlight(id, scroll = true) {
  state.hl = id;
  document.querySelectorAll('.hl').forEach(e => e.classList.remove('hl'));
  document.querySelectorAll(`[data-chunk="${CSS.escape(id)}"]`).forEach(e => { e.classList.add('hl'); if (scroll) e.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
}
function setTab(t) {
  state.tab = t;
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  ['inspect', 'chunks', 'ask'].forEach(x => $('tab-' + x).hidden = x !== t);
}
function go(c) {
  state.view = c.where.slide ? { doc: c.doc, slide: c.where.slide } : { doc: c.doc, sheet: c.where.sheet };
  state.mode = 'structure'; document.querySelectorAll('#seg button').forEach(b => b.classList.toggle('on', b.dataset.mode === 'structure'));
  setTab('inspect'); render(); highlight(c.id);
}

// ---------- query pipeline ----------
async function ask(q) {
  q = q.trim(); if (!q || !state.chunks.length) return;
  $('askBtn').disabled = true; $('q').value = ''; $('askEmpty')?.remove();
  const run = startRun('query', q);
  const turn = { q, run: run.id }; state.answers.push(turn);
  $('thread').insertAdjacentHTML('beforeend', `<div class="msg-q">${esc(q)}</div><div class="msg-a" id="a-${run.id}"><div class="body dim">Running pipeline…</div></div>`);
  $('thread').scrollTop = $('thread').scrollHeight;
  const box = $('a-' + run.id), chips = [];
  let s;
  try {
    let plan = { route: 'both', sheet: null, filters: {}, query: q, source: 'none' };
    s = step(run, 'Plan', cfg.provider ? `asking <code>${esc(cfg.model)}</code> to route, pick a sheet and extract filters` : '');
    if (cfg.provider) {
      plan = JSON.parse(await pycall('plan_json', JSON.stringify(cfg), q));
      const f = Object.entries(plan.filters || {});
      finish(run, s, plan.source === 'llm' ? 'done' : 'fail', plan.source === 'llm'
        ? `route <code>${plan.route === 'none' ? 'none → searching everything' : plan.route}</code>${plan.sheet ? ` · sheet <code>${esc(plan.sheet)}</code>` : ''}${f.length ? ` · filters ${f.map(([k, v]) => `<code>${esc(k)} = ${esc(v)}</code>`).join(' ')}` : ''}<br>validated against the real sheets, columns and values${plan.query && plan.query !== q ? `<br>search query: “${esc(plan.query)}”` : ''}`
        : `model call failed, searching everything: ${esc(plan.error || '')}`);
    } else finish(run, s, 'skip', 'No model set: searching all chunks. Add a model for routing and row filters.');
    chips.push(`<span class="chip">route ${esc(plan.route)}</span>`, ...(plan.sheet ? [`<span class="chip">sheet ${esc(plan.sheet)}</span>`] : []), ...Object.entries(plan.filters || {}).map(([k, v]) => `<span class="chip">${esc(k)} = ${esc(v)}</span>`));

    if (state.vecs.some(v => !v)) await embedAll(run);
    s = step(run, 'Embed query'); const t = now();
    const [qv] = await embed([plan.query || q]);
    finish(run, s, 'done', `“${esc(plan.query || q)}” → 384-d vector in ${fmtMs(now() - t)}`);

    s = step(run, 'Filter + rank'); await tick();
    const scores = state.vecs.map(v => v ? v.reduce((a, x, i) => a + x * qv[i], 0) : 0);
    const { context, stats } = JSON.parse(pycall('select_json', JSON.stringify(state.chunks), JSON.stringify(scores), JSON.stringify(plan)));
    finish(run, s, context.length ? 'done' : 'fail', `${stats.eligible} of ${stats.total} chunks eligible after route/filters · top ${stats.hits} by cosine:<ul>${(stats.top || []).map(x => `<li><code>${x.score.toFixed(3)}</code> ${esc(x.id.split('#')[1])}</li>`).join('')}</ul>`);
    s = step(run, 'Expand context'); finish(run, s, stats.siblings ? 'done' : 'skip', stats.siblings ? `+${plural(stats.siblings, 'sibling chunk')} from the same slides, so the answer sees whole slides · ${plural(context.length, 'source')} total` : `top hits already cover their slides · ${plural(context.length, 'source')}`);

    const srcs = `<div class="srcs">${context.map((c, i) => `<button class="src" style="--c:${C(c.type)}" data-run="${run.id}" data-src="${i}"><b>[${i + 1}]</b>${esc(c.id.split('#')[1])}</button>`).join('')}</div>`;
    turn.context = context;
    s = step(run, 'Answer');
    if (!cfg.provider) {
      finish(run, s, 'skip', 'No model set: showing the most relevant sources instead of a written answer');
      box.innerHTML = `<div class="meta">${chips.join('')}<span class="chip">search only</span></div><div class="body dim">Top matches are below. <button class="cite" data-act="settings">Add a model</button> for a written, cited answer.</div>${srcs}`;
    } else if (!context.length) {
      finish(run, s, 'skip', 'nothing retrieved'); box.innerHTML = `<div class="body">I couldn't find that in the loaded documents.</div>`;
    } else {
      s.detail = `<code>${esc(cfg.model)}</code> writing from ${plural(context.length, 'numbered source')}`; renderRuns();
      const text = await pycall('answer_json', JSON.stringify(cfg), q, JSON.stringify(context));
      const cites = new Set([...text.matchAll(/\[(\d+)\]/g)].map(m => +m[1]).filter(n => n >= 1 && n <= context.length));
      finish(run, s, 'done', `<code>${esc(cfg.model)}</code> · cites ${cites.size} of ${context.length} sources`);
      box.innerHTML = `<div class="meta">${chips.join('')}<span class="chip">${esc(cfg.model)}</span></div><div class="body">${esc(text).replace(/\[(\d+)\]/g, (m, n) => +n >= 1 && +n <= context.length ? `<button class="cite" data-run="${run.id}" data-src="${+n - 1}">${n}</button>` : m)}</div>${srcs}`;
    }
    box.insertAdjacentHTML('beforeend', `<button class="trace-link" data-trace="${run.id}">View pipeline trace (${fmtMs(now() - run.t0)}) →</button>`);
    endRun(run);
  } catch (e) {
    console.error(e); if (s && s.status === 'run') finish(run, s, 'fail', esc(lastLine(e)));
    box.innerHTML = `<div class="body dim">The pipeline failed: ${esc(lastLine(e))}. Check the model settings and the Pipeline panel.</div>`; endRun(run, 'fail');
  }
  $('askBtn').disabled = false; $('thread').scrollTop = $('thread').scrollHeight;
}

// ---------- settings ----------
function loadCfg() { try { const c = JSON.parse(localStorage.getItem('deckray-cfg') || '{}'); c.key = localStorage.getItem('deckray-key') || sessionStorage.getItem('deckray-key') || ''; return c; } catch { return {}; } }
function saveCfg() {
  try {
    const { key, ...rest } = cfg; localStorage.setItem('deckray-cfg', JSON.stringify(rest));
    localStorage.removeItem('deckray-key'); sessionStorage.removeItem('deckray-key');
    if (key) (cfg.remember ? localStorage : sessionStorage).setItem('deckray-key', key);
  } catch {}
  const label = cfg.provider ? `${cfg.model} · ${PROVIDERS[cfg.provider].label}` : 'none';
  $('modelName').textContent = label; $('sb-model').textContent = 'Model: ' + (cfg.provider ? label : 'none (search only)');
}
function fillSettings() {
  $('prov').value = cfg.provider || ''; $('base').value = cfg.base || PROVIDERS[cfg.provider]?.base || '';
  $('key').value = cfg.key || ''; $('model').value = cfg.model || ''; $('vision').checked = !!cfg.vision; $('remember').checked = !!cfg.remember; syncProv();
}
function syncProv() {
  const p = PROVIDERS[$('prov').value]; $('provFields').hidden = !p; if (!p) return;
  if (!$('base').value) $('base').value = p.base;
  $('keyWrap').hidden = !p.key; $('provHint').innerHTML = p.hint;
  if ($('prov').value === 'ollama' && !$('model').value) $('model').value = 'qwen2.5:14b';
}
async function fetchModels() {
  const prov = $('prov').value, base = $('base').value.replace(/\/$/, ''), key = $('key').value;
  const headers = prov === 'anthropic' ? { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' } : key ? { authorization: 'Bearer ' + key } : {};
  try {
    const r = await fetch(base + '/models', { headers }); if (!r.ok) throw new Error(r.status + ' ' + (await r.text()).slice(0, 120));
    const ids = ((await r.json()).data || []).map(m => m.id).sort();
    $('models').innerHTML = ids.map(i => `<option value="${esc(i)}">`).join(''); toast(`${ids.length} models available: click the model box to pick one.`);
  } catch (e) { toast('Could not list models: ' + e.message); }
}

// ---------- events ----------
document.addEventListener('click', e => {
  const a = e.target.closest('[data-act]')?.dataset.act;
  if (a === 'samples') return loadSamples();
  if (a === 'browse') return $('file').click();
  if (a === 'settings') { fillSettings(); return $('settings').showModal(); }
  const src = e.target.closest('[data-src]');
  if (src) { const turn = state.answers.find(t => t.run === src.dataset.run); return turn && go(turn.context[+src.dataset.src]); }
  const tr = e.target.closest('[data-trace]');
  if (tr) { $('activity').classList.add('show'); renderRuns(tr.dataset.trace); document.querySelector(`.run[data-id="${tr.dataset.trace}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
});
$('browse').onclick = () => $('file').click();
$('addBtn').onclick = () => $('file').click();
$('file').onchange = e => { const f = [...e.target.files]; e.target.value = ''; if (f.length) addFiles(f); };
let dragDepth = 0;
addEventListener('dragenter', e => { if (e.dataTransfer?.types.includes('Files')) { dragDepth++; $('overlay').hidden = false; } });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('overlay').hidden = true; } });
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; $('overlay').hidden = true; if (!py) return toast('Still starting the Python runtime…'); const f = [...e.dataTransfer.files]; if (f.length) addFiles(f); });
$('tree').onclick = e => {
  const rm = e.target.closest('[data-remove]'); if (rm) return removeDoc(rm.dataset.remove);
  const it = e.target.closest('.titem'); if (!it) return;
  state.view = it.dataset.slide ? { doc: it.dataset.doc, slide: +it.dataset.slide } : { doc: it.dataset.doc, sheet: it.dataset.sheet };
  state.hl = null; setTab('inspect'); render();
};
document.querySelector('.tabs').onclick = e => { const t = e.target.closest('.tab'); if (t) setTab(t.dataset.tab); };
$('seg').onclick = e => { const b = e.target.closest('button'); if (!b) return; state.mode = b.dataset.mode; document.querySelectorAll('#seg button').forEach(x => x.classList.toggle('on', x === b)); renderInspector(); };
$('canvas').onclick = e => { const t = e.target.closest('[data-chunk]'); if (t) highlight(t.dataset.chunk); };
$('insp').onclick = e => { const ck = e.target.closest('.ck'); if (!ck) return; ck.classList.toggle('open'); highlight(ck.dataset.chunk); };
$('tq').oninput = renderTable;
$('tfilters').onclick = e => { const b = e.target.closest('[data-t]'); if (!b) return; state.tfilter = state.tfilter === b.dataset.t ? null : b.dataset.t; renderTable(); };
$('tbody').onclick = e => { const r = e.target.closest('[data-i]'); if (r) go(state.chunks[+r.dataset.i]); };
$('askForm').onsubmit = e => { e.preventDefault(); ask($('q').value); };
$('examples').onclick = e => { if (e.target.classList.contains('ex')) { setTab('ask'); ask(e.target.textContent); } };
$('clearRuns').onclick = () => { state.runs = state.runs.filter(r => r.status === 'run'); renderRuns(); };
$('togglePipe').onclick = () => $('activity').classList.toggle('show');
$('openSettings').onclick = () => { fillSettings(); $('settings').showModal(); };
$('prov').onchange = () => { $('base').value = PROVIDERS[$('prov').value]?.base || ''; $('model').value = ''; $('models').innerHTML = ''; syncProv(); };
$('fetchModels').onclick = fetchModels;
$('setForm').onsubmit = e => {
  if (e.submitter?.value !== 'save') return;
  const provider = $('prov').value;
  if (provider && !$('model').value.trim()) { e.preventDefault(); return toast('Pick or type a model id.'); }
  if (provider && PROVIDERS[provider].key && !$('key').value.trim()) { e.preventDefault(); return toast('This provider needs an API key.'); }
  cfg = provider ? { provider, base: $('base').value.trim(), key: $('key').value.trim(), model: $('model').value.trim(), vision: $('vision').checked, remember: $('remember').checked } : {};
  saveCfg();
};

saveCfg(); render(); boot();
