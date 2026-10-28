// Office RAG in the browser: the officerag Python package runs in Pyodide, embeddings run in transformers.js,
// and LLM calls (optional) go straight from this page to the provider the visitor picked.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
env.allowLocalModels = false;

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
const MAX = 20 * 1024 * 1024;
const SAMPLES = ['Solar Program Q3 Review.pptx', 'Solar Sites Directory.xlsx'];
const EXAMPLES = ['Which sites in Spain are offline?', 'How much did the carport sites generate in August?',
  'What should the presenter stress about the risks?', 'Which panel vendor has the longest warranty?', 'What are the Q4 priorities?'];
const COLOR = t => `var(--${t})`;
const PROVIDERS = {
  ollama: { base: 'http://localhost:11434/v1', key: false, hint: 'Start Ollama so this page may call it: <code>OLLAMA_ORIGINS=' + location.origin + ' ollama serve</code>' },
  openai: { base: 'https://api.openai.com/v1', key: true, hint: 'Create a key at platform.openai.com. It is sent only to api.openai.com.' },
  anthropic: { base: 'https://api.anthropic.com/v1', key: true, hint: 'Create a key at console.anthropic.com. It is sent only to api.anthropic.com.' },
  groq: { base: 'https://api.groq.com/openai/v1', key: true, hint: 'Free keys at console.groq.com. It is sent only to api.groq.com.' },
  openrouter: { base: 'https://openrouter.ai/api/v1', key: true, hint: 'One key for many models at openrouter.ai. It is sent only to openrouter.ai.' },
};

const state = { docs: {}, chunks: [], vecs: [], view: null, mode: 'structure', hl: null };
let py, embedder, cfg = loadCfg();

// ---------- boot ----------
function step(id, cls, text) { const el = $('st-' + id); el.className = 'step ' + (cls || ''); if (text) el.textContent = text; }
function toast(msg) { const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; document.body.append(t); setTimeout(() => t.remove(), 4500); }

window.llmPost = async (url, headers, body) => {
  const r = await fetch(url, { method: 'POST', headers: JSON.parse(headers), body });
  const text = await r.text();
  if (!r.ok) throw new Error(`${r.status} from ${new URL(url).host}: ${text.slice(0, 160)}`);
  return text;
};

const BRIDGE = `
import json, io, zipfile, base64, sys
sys.path.insert(0, '/pkg')
import officerag
from officerag import llm, chunks as ch
from js import llmPost
DOCS, BYTES = {}, {}
async def post(url, headers, body):
    return json.loads(await llmPost(url, json.dumps(headers), json.dumps(body)))
def ingest_json(name, data):
    b = bytes(data.to_py()); r = officerag.ingest(name, b); DOCS[name], BYTES[name] = r, b
    return json.dumps(r)
def remove(name):
    DOCS.pop(name, None); BYTES.pop(name, None)
async def describe_images(name, cfg_json):
    cfg, r = json.loads(cfg_json), DOCS[name]
    z = zipfile.ZipFile(io.BytesIO(BYTES[name])); n = 0
    for s in r['extraction'].get('slides', []):
        for img in s['images']:
            part = img.get('part')
            if not part or part not in z.namelist() or img.get('description'): continue
            ext = part.rsplit('.', 1)[-1].lower()
            if ext not in ('png', 'jpg', 'jpeg', 'gif', 'webp'): continue
            mime = 'image/jpeg' if ext == 'jpg' else 'image/' + ext
            try:
                img['description'] = await llm.describe_image(cfg, post, base64.b64encode(z.read(part)).decode(), mime); n += 1
            except Exception as e:
                img['description_error'] = str(e)[:160]
    r['chunks'] = ch.chunk(name, r['extraction'])
    return json.dumps({'described': n, 'doc': r})
async def plan_json(cfg_json, q):
    src = ch.sources({n: d['extraction'] for n, d in DOCS.items()})
    return json.dumps(await llm.plan(json.loads(cfg_json), post, q, src))
def select_json(chunks_json, scores_json, plan_json):
    return json.dumps(ch.select_context(json.loads(chunks_json), json.loads(scores_json), json.loads(plan_json)))
async def answer_json(cfg_json, q, ctx_json):
    return await llm.answer(json.loads(cfg_json), post, q, json.loads(ctx_json))
`;

async function boot() {
  step('py', 'run', 'python runtime…');
  try {
    py = await loadPyodide();
    await py.loadPackage('micropip');
    await py.pyimport('micropip').install(['openpyxl', 'defusedxml']);
    py.FS.mkdirTree('/pkg/officerag');
    await Promise.all(['__init__.py', 'pptx.py', 'xlsx.py', 'chunks.py', 'llm.py'].map(async f =>
      py.FS.writeFile('/pkg/officerag/' + f, await (await fetch('officerag/' + f)).text())));
    py.runPython(BRIDGE);
    step('py', 'ok', 'python ready');
    $('loadSamples').disabled = $('pick').disabled = false;
  } catch (e) { console.error(e); step('py', 'err', 'python failed to load'); toast('Could not start the Python runtime: ' + e.message); }
}

// ---------- ingest ----------
async function addFiles(files) {
  for (const f of files) {
    if (!/\.(pptx|xlsx|xlsm)$/i.test(f.name)) { toast(`${f.name}: only .pptx, .xlsx and .xlsm are supported.`); continue; }
    if (f.size > MAX) { toast(`${f.name} is over 20 MB.`); continue; }
    if (state.docs[f.name]) removeDoc(f.name);
    step('parse', 'run', `parsing ${f.name}…`);
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      let doc = JSON.parse(py.globals.get('ingest_json')(f.name, bytes));
      if (cfg.provider && cfg.vision && doc.extraction.kind === 'pptx' && doc.extraction.slides.some(s => s.images.length)) {
        step('img', 'run', 'describing images…');
        const r = JSON.parse(await py.globals.get('describe_images')(f.name, JSON.stringify(cfg)));
        doc = r.doc; step('img', 'ok', `${r.described} image${r.described === 1 ? '' : 's'} described`);
      }
      state.docs[f.name] = doc;
      step('parse', 'ok', `parsed ${Object.keys(state.docs).length} file${Object.keys(state.docs).length > 1 ? 's' : ''}`);
      if (!state.view) state.view = firstView(f.name);
    } catch (e) {
      console.error(e); step('parse', 'err', 'parse failed');
      toast(String(e.message || e).split('\n').filter(Boolean).pop());
    }
  }
  rebuild(); render(); await embedAll();
}

async function loadSamples() {
  $('loadSamples').disabled = true;
  const files = await Promise.all(SAMPLES.map(async n => new File([await (await fetch('samples/' + encodeURIComponent(n))).blob()], n)));
  await addFiles(files);
  $('examples').innerHTML = EXAMPLES.map(x => `<button class="chip" type="button">${esc(x)}</button>`).join('');
}

function removeDoc(name) {
  delete state.docs[name]; py.globals.get('remove')(name);
  if (state.view?.doc === name) state.view = Object.keys(state.docs).length ? firstView(Object.keys(state.docs)[0]) : null;
  rebuild(); render();
}

function firstView(doc) { return state.docs[doc].extraction.kind === 'pptx' ? { doc, slide: 1 } : { doc, sheet: state.docs[doc].extraction.sheets[0].name }; }

function rebuild() {
  const old = new Map(state.chunks.map((c, i) => [c.id + '\u0000' + c.text, state.vecs[i]]));
  state.chunks = Object.values(state.docs).flatMap(d => d.chunks);
  state.vecs = state.chunks.map(c => old.get(c.id + '\u0000' + c.text) || null);
  const ready = state.chunks.length > 0;
  $('q').disabled = $('askBtn').disabled = !ready;
  $('q').placeholder = ready ? 'Ask a question about the loaded documents…' : 'Load documents first, then ask a question…';
}

// ---------- embeddings (all-MiniLM-L6-v2, in the browser) ----------
async function embed(texts) {
  embedder ??= await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
  const out = await embedder(texts, { pooling: 'mean', normalize: true });
  return Array.from({ length: texts.length }, (_, i) => out.data.slice(i * 384, (i + 1) * 384));
}
async function embedAll() {
  const todo = state.vecs.map((v, i) => v ? -1 : i).filter(i => i >= 0);
  if (!todo.length) return;
  step('embed', 'run', 'loading embedding model…');
  try {
    for (let i = 0; i < todo.length; i += 16) {
      const batch = todo.slice(i, i + 16), vecs = await embed(batch.map(j => state.chunks[j].text));
      batch.forEach((j, k) => state.vecs[j] = vecs[k]);
      step('embed', 'run', `embedding ${Math.min(i + 16, todo.length)}/${todo.length}`);
    }
    step('embed', 'ok', `${state.chunks.length} chunks embedded`);
  } catch (e) { console.error(e); step('embed', 'err', 'embedding failed'); toast('Embedding model failed to load: ' + e.message); }
}

// ---------- render ----------
function render() { renderLib(); renderView(); renderChunks(); }

function renderLib() {
  const names = Object.keys(state.docs);
  if (!names.length) { $('lib').innerHTML = '<div class="empty">No documents yet.</div>'; return; }
  $('lib').innerHTML = names.map(n => {
    const ex = state.docs[n].extraction;
    const items = ex.kind === 'pptx'
      ? ex.slides.map(s => `<button class="item${state.view?.doc === n && state.view.slide === s.index ? ' on' : ''}" data-doc="${esc(n)}" data-slide="${s.index}"><small>${s.index}</small>${esc(s.title || 'Untitled')}${s.hidden ? '<span class="badge">hidden</span>' : ''}</button>`)
      : ex.sheets.map(s => `<button class="item${state.view?.doc === n && state.view.sheet === s.name ? ' on' : ''}" data-doc="${esc(n)}" data-sheet="${esc(s.name)}"><small>▦</small>${esc(s.name)}${s.hidden ? '<span class="badge">hidden</span>' : `<span class="badge">${s.rows.length} rows</span>`}</button>`);
    return `<div class="doc"><div class="dn"><span>${ex.kind === 'pptx' ? '📊' : '📗'} ${esc(n)}</span><button title="Remove" data-remove="${esc(n)}">×</button></div>${items.join('')}</div>`;
  }).join('');
}

const el = (type, id, label, inner) => `<div class="el" style="--c:${COLOR(type)}" data-chunk="${esc(id)}"><span class="tag">${label}</span>${inner}</div>`;

function renderSlide(doc, s) {
  const id = sfx => `${doc}#s${s.index}-${sfx}`, parts = [];
  if (s.lines.length) parts.push(el('text', id('text'), 'text', `<p class="lines">${esc(s.lines.join('\n'))}</p>`));
  s.tables.forEach((t, i) => parts.push(el('table', id('table' + (i + 1)), `table · ${t.rows.length} rows`,
    `<table class="xt"><tr>${t.headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr>${t.rows.map(r => `<tr>${t.headers.map(h => `<td>${esc(r[h])}</td>`).join('')}</tr>`).join('')}</table>`)));
  s.charts.forEach((c, i) => {
    const cats = [...new Set(c.series.flatMap(x => x.points.map(p => p.label)))], max = Math.max(...c.series.flatMap(x => x.points.map(p => p.value)), 1);
    const hues = ['var(--chart)', 'var(--text)', 'var(--table)', 'var(--row)'];
    const groups = cats.map(cat => `<div class="g"><div class="bs">${c.series.map((x, k) => { const p = x.points.find(p => p.label === cat); return p ? `<div class="b" style="height:${Math.max(2, p.value / max * 100)}%;background:${hues[k % 4]}"><span>${p.value}</span></div>` : ''; }).join('')}</div><label>${esc(cat)}</label></div>`).join('');
    parts.push(el('chart', id('chart' + (i + 1)), `${c.kind} chart · data extracted`, `<b>${esc(c.title)}</b><div class="bars">${groups}</div><div class="legend">${c.series.map((x, k) => `<span><i style="background:${hues[k % 4]}"></i>${esc(x.name)}</span>`).join('')}</div>`));
  });
  const tree = ns => `<ul class="tree">${ns.map(n => `<li>${esc(n.text)}${n.children.length ? tree(n.children) : ''}</li>`).join('')}</ul>`;
  s.smartart.forEach((t, i) => parts.push(el('smartart', id('smartart' + (i + 1)), 'smartart', tree(t))));
  if (s.images.length) parts.push(el('images', id('images'), `images · ${s.images.length}`, s.images.map(im =>
    `<div class="alt"><b>img</b><span>${esc(im.description || im.alt || 'No alt text: add a vision model in settings to describe it.')}${im.description ? ' <em style="color:var(--faint)">(described by model)</em>' : im.alt ? ' <em style="color:var(--faint)">(alt text)</em>' : ''}</span></div>`).join('')));
  if (s.links.length) parts.push(el('links', id('links'), 'links', s.links.map(l => `<div>↗ ${esc(l.text || 'link')} <span class="mono" style="color:var(--faint);font-size:12px">${esc(l.url)}</span></div>`).join('')));
  if (s.notes) parts.push(el('notes', id('notes'), 'speaker notes', `<div class="sticky">${esc(s.notes)}</div>`));
  return `<div class="slide">${s.title ? `<h2>${esc(s.title)}</h2>` : ''}${parts.join('') || '<p class="empty">Nothing extractable on this slide.</p>'}</div>`;
}

function renderSheet(doc, s) {
  const letters = n => { let s = ''; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; };
  const merged = new Set(), origins = new Set();
  s.merged.forEach(r => {
    const m = r.match(/^([A-Z]+)(\d+):([A-Z]+)(\d+)$/); if (!m) return;
    const col = t => [...t].reduce((a, c) => a * 26 + c.charCodeAt(0) - 64, 0) - 1;
    for (let rr = +m[2]; rr <= +m[4]; rr++) for (let cc = col(m[1]); cc <= col(m[3]); cc++) (rr === +m[2] && cc === col(m[1]) ? origins : merged).add(`${rr},${cc}`);
  });
  const width = Math.max(...s.preview.map(r => r.length), 1), dataRows = new Set(s.rows.map(r => r.row));
  const body = s.preview.map((row, i) => {
    const rn = i + 1, cls = rn === s.header_row ? 'hdr' : rn < s.header_row ? 'desc' : dataRows.has(rn) ? 'data' : '';
    const id = `${doc}#${s.name}!${rn}`;
    return `<tr class="${cls}"${cls === 'data' ? ` data-chunk="${esc(id)}"` : ''}><td class="rn">${rn}</td>${Array.from({ length: width }, (_, c) => `<td class="${merged.has(`${rn},${c}`) ? 'm' : ''}" title="${esc(row[c])}">${esc(row[c])}</td>`).join('')}</tr>`;
  }).join('');
  const info = [`header row ${s.header_row}`, `${s.rows.length} row chunks`, s.merged.length ? `${s.merged.length} merged range${s.merged.length > 1 ? 's' : ''} unmerged` : '', s.headers.filter(h => h.startsWith('Column ')).length ? `${s.headers.filter(h => h.startsWith('Column ')).length} unnamed column kept in place` : ''].filter(Boolean);
  return `<div class="sheetbar">${s.hidden ? '<span class="warn">hidden sheet: still indexed</span>' : ''}${info.map(x => `<span>${esc(x)}</span>`).join('')}${s.truncated ? '<span class="warn">large sheet: capped at 5,000 rows × 60 columns</span>' : ''}</div>
    <div class="gridwrap"><table class="grid"><thead><tr><th></th>${Array.from({ length: width }, (_, c) => `<th>${letters(c)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
    <div class="gridnote"><span><i style="background:color-mix(in oklab,var(--acc) 16%,#fff)"></i>detected header</span><span><i style="background:#fff;font-style:italic"></i><em>title rows → sheet description</em></span><span><i style="background-image:repeating-linear-gradient(135deg,transparent 0 3px,color-mix(in oklab,var(--table) 30%,transparent) 3px 5px)"></i>filled from a merged cell</span><span>Click a row to see its chunk.</span></div>`;
}

function renderView() {
  const v = state.view;
  if (!v || !state.docs[v.doc]) { $('xray').innerHTML = '<div class="empty">Load the samples or drop a file to see its structure: every element is outlined in the color of the chunk it becomes.</div>'; $('viewTitle').textContent = 'X-ray'; $('viewMeta').textContent = ''; return; }
  const ex = state.docs[v.doc].extraction;
  if (ex.kind === 'pptx') {
    const s = ex.slides.find(x => x.index === v.slide) || ex.slides[0];
    $('viewTitle').textContent = `X-ray · slide ${s.index}`; $('viewMeta').textContent = v.doc;
    $('xray').innerHTML = renderSlide(v.doc, s);
  } else {
    const s = ex.sheets.find(x => x.name === v.sheet) || ex.sheets[0];
    $('viewTitle').textContent = `X-ray · ${s.name}`; $('viewMeta').textContent = v.doc;
    $('xray').innerHTML = renderSheet(v.doc, s);
  }
  if (state.hl) highlight(state.hl, false);
}

function viewChunks() {
  const v = state.view; if (!v || !state.docs[v.doc]) return [];
  const d = state.docs[v.doc];
  if (state.mode === 'naive') return d.naive;
  return d.chunks.filter(c => v.slide ? c.where.slide === v.slide : c.where.sheet === v.sheet);
}

function renderChunks() {
  const list = viewChunks(), d = state.view && state.docs[state.view.doc];
  $('chunkTitle').textContent = `Chunks · ${list.length}`;
  const note = state.mode === 'naive' && d
    ? `<div class="note">The naive baseline: every value in the file flattened and cut into ${list[0]?.text.length || 700}-character windows, the way many pipelines start. No slide titles, no column names, merged cells mostly empty. Compare with the structure-aware chunks.</div>` : '';
  $('clist').innerHTML = note + (list.length ? list.map(c => `<div class="ck" style="--c:${COLOR(c.type)}" data-chunk="${esc(c.id)}"><header><b>${esc(c.type)}</b><span>~${c.tokens} tokens</span></header><pre>${esc(c.text)}</pre>${c.text.length > 360 ? '<button class="more" type="button">show all</button>' : ''}</div>`).join('') : '<div class="empty">No chunks here.</div>');
}

function highlight(id, scroll = true) {
  state.hl = id;
  document.querySelectorAll('.hl').forEach(e => e.classList.remove('hl'));
  document.querySelectorAll(`[data-chunk="${CSS.escape(id)}"]`).forEach(e => { e.classList.add('hl'); if (scroll) e.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
}

function go(chunk) {
  state.view = chunk.where.slide ? { doc: chunk.doc, slide: chunk.where.slide } : { doc: chunk.doc, sheet: chunk.where.sheet };
  state.mode = 'structure'; document.querySelectorAll('.seg button').forEach(b => b.classList.toggle('on', b.dataset.mode === 'structure'));
  render(); highlight(chunk.id);
  document.querySelector('.lab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- ask ----------
let lastContext = [];
async function ask(q) {
  q = q.trim(); if (!q || !state.chunks.length) return;
  $('askBtn').disabled = true; $('flow').hidden = false; $('result').hidden = false;
  $('answer').className = 'answer dim'; $('answer').textContent = 'Working…'; $('srcs').innerHTML = '';
  const t0 = performance.now(), pills = [];
  const flow = () => $('flow').innerHTML = pills.join('<span class="arrow">→</span>');
  try {
    let plan = { route: 'both', sheet: null, filters: {}, query: q, source: 'none' };
    if (cfg.provider) {
      pills.push('<span class="pill">planning…</span>'); flow();
      plan = JSON.parse(await py.globals.get('plan_json')(JSON.stringify(cfg), q));
      pills.pop();
    }
    const f = Object.entries(plan.filters || {});
    pills.push(`<span class="pill">plan · <b>${plan.source === 'llm' ? 'model' : plan.source === 'fallback' ? 'fallback (model error)' : 'no model'}</b></span>`,
      `<span class="pill">route <b>${esc(plan.route)}</b>${plan.sheet ? ` · sheet <b>${esc(plan.sheet)}</b>` : ''}</span>`);
    if (f.length) pills.push(`<span class="pill">filters ${f.map(([k, v]) => `<b>${esc(k)} = ${esc(v)}</b>`).join(', ')}</span>`);
    flow();
    if (state.vecs.some(v => !v)) await embedAll();
    const [qv] = await embed([plan.query || q]);
    const scores = state.vecs.map(v => v ? v.reduce((s, x, i) => s + x * qv[i], 0) : 0);
    lastContext = JSON.parse(py.globals.get('select_json')(JSON.stringify(state.chunks), JSON.stringify(scores), JSON.stringify(plan)));
    pills.push(`<span class="pill">retrieved <b>${lastContext.length}</b> chunk${lastContext.length === 1 ? '' : 's'}</span>`); flow();
    $('srcs').innerHTML = lastContext.map((c, i) => `<button class="src" style="--c:${COLOR(c.type)}" data-src="${i}"><b>[${i + 1}]</b><span><small>${esc(c.id)}</small>${esc(c.text.replace(/^\[[^\]]*\]\s*/, ''))}</span></button>`).join('')
      || '<div class="note">No matching chunks. If the plan applied filters, try rephrasing.</div>';
    if (!cfg.provider) {
      $('answer').className = 'answer dim';
      $('answer').innerHTML = 'Search only: these are the most relevant chunks. <button class="cite" type="button" id="openSettings2">Add a model</button> to get a written, cited answer.';
    } else if (!lastContext.length) {
      $('answer').className = 'answer'; $('answer').textContent = "I couldn't find that in the loaded documents.";
    } else {
      pills.push('<span class="pill">answering…</span>'); flow();
      const text = await py.globals.get('answer_json')(JSON.stringify(cfg), q, JSON.stringify(lastContext));
      pills.pop(); pills.push(`<span class="pill">answer · <b>${esc(cfg.model)}</b></span>`); flow();
      $('answer').className = 'answer';
      $('answer').innerHTML = esc(text).replace(/\[(\d+)\]/g, (m, n) => +n >= 1 && +n <= lastContext.length ? `<button class="cite" type="button" data-src="${+n - 1}">${n}</button>` : m);
    }
    $('askMeta').textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
  } catch (e) {
    console.error(e); $('answer').className = 'answer dim';
    $('answer').textContent = 'The model call failed: ' + String(e.message || e).split('\n').filter(Boolean).pop() + ' Check the model settings.';
  }
  $('askBtn').disabled = false;
}

// ---------- settings ----------
function loadCfg() {
  try { const c = JSON.parse(localStorage.getItem('officerag-cfg') || '{}'); c.key = localStorage.getItem('officerag-key') || sessionStorage.getItem('officerag-key') || ''; return c; } catch { return {}; }
}
function saveCfg() {
  try {
    const { key, ...rest } = cfg;
    localStorage.setItem('officerag-cfg', JSON.stringify(rest));
    localStorage.removeItem('officerag-key'); sessionStorage.removeItem('officerag-key');
    if (key) (cfg.remember ? localStorage : sessionStorage).setItem('officerag-key', key);
  } catch {}
  $('modelName').textContent = cfg.provider ? `${cfg.model || '?'} · ${cfg.provider}` : 'none (search only)';
}
function fillSettings() {
  $('prov').value = cfg.provider || ''; $('base').value = cfg.base || PROVIDERS[cfg.provider]?.base || '';
  $('key').value = cfg.key || ''; $('model').value = cfg.model || ''; $('vision').checked = !!cfg.vision; $('remember').checked = !!cfg.remember;
  syncProv();
}
function syncProv() {
  const p = PROVIDERS[$('prov').value];
  $('provFields').hidden = !p;
  if (!p) return;
  if (!Object.values(PROVIDERS).some(x => x.base === $('base').value) && $('base').value) { /* keep custom base */ } else $('base').value = p.base;
  $('keyWrap').hidden = !p.key; $('provHint').innerHTML = p.hint;
  if ($('prov').value === 'ollama' && !$('model').value) $('model').value = 'qwen2.5:14b';
}
async function fetchModels() {
  const prov = $('prov').value, base = $('base').value.replace(/\/$/, ''), key = $('key').value;
  const headers = prov === 'anthropic' ? { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' } : key ? { authorization: 'Bearer ' + key } : {};
  try {
    const r = await fetch(base + '/models', { headers }); if (!r.ok) throw new Error(r.status + ' ' + (await r.text()).slice(0, 120));
    const ids = ((await r.json()).data || []).map(m => m.id).sort();
    $('models').innerHTML = ids.map(i => `<option value="${esc(i)}">`).join('');
    toast(`${ids.length} models available. Click the model box to pick one.`);
  } catch (e) { toast('Could not list models: ' + e.message); }
}

// ---------- events ----------
$('loadSamples').onclick = loadSamples;
$('pick').onclick = () => $('file').click();
$('file').onchange = e => { addFiles([...e.target.files]); e.target.value = ''; };
const drop = $('drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (py) addFiles([...e.dataTransfer.files]); else toast('Still starting the Python runtime…'); });
$('lib').onclick = e => {
  const rm = e.target.closest('[data-remove]'); if (rm) return removeDoc(rm.dataset.remove);
  const it = e.target.closest('.item'); if (!it) return;
  state.view = it.dataset.slide ? { doc: it.dataset.doc, slide: +it.dataset.slide } : { doc: it.dataset.doc, sheet: it.dataset.sheet };
  state.hl = null; render();
};
document.querySelector('.seg').onclick = e => { const b = e.target.closest('button'); if (!b) return; state.mode = b.dataset.mode; document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b)); renderChunks(); };
$('xray').onclick = e => { const t = e.target.closest('[data-chunk]'); if (t) highlight(t.dataset.chunk); };
$('clist').onclick = e => {
  if (e.target.classList.contains('more')) { const ck = e.target.closest('.ck'); ck.classList.toggle('open'); e.target.textContent = ck.classList.contains('open') ? 'show less' : 'show all'; return; }
  const t = e.target.closest('[data-chunk]'); if (t) highlight(t.dataset.chunk);
};
$('askForm').onsubmit = e => { e.preventDefault(); ask($('q').value); };
$('examples').onclick = e => { if (e.target.classList.contains('chip')) { $('q').value = e.target.textContent; ask($('q').value); } };
document.querySelector('.ask').onclick = e => {
  if (e.target.id === 'openSettings2') { fillSettings(); $('settings').showModal(); return; }
  const s = e.target.closest('[data-src]'); if (s) go(lastContext[+s.dataset.src]);
};
$('openSettings').onclick = () => { fillSettings(); $('settings').showModal(); };
$('prov').onchange = () => { $('base').value = PROVIDERS[$('prov').value]?.base || ''; $('model').value = ''; $('models').innerHTML = ''; syncProv(); };
$('fetchModels').onclick = fetchModels;
$('setForm').onsubmit = e => {
  if (e.submitter?.value !== 'save') return;
  const provider = $('prov').value;
  if (provider && !$('model').value.trim()) { e.preventDefault(); toast('Pick or type a model id.'); return; }
  if (provider && PROVIDERS[provider].key && !$('key').value.trim()) { e.preventDefault(); toast('This provider needs an API key.'); return; }
  cfg = provider ? { provider, base: $('base').value.trim(), key: $('key').value.trim(), model: $('model').value.trim(), vision: $('vision').checked, remember: $('remember').checked } : {};
  saveCfg();
};

saveCfg();
boot();
