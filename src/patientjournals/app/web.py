from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from patientjournals.app.catalog import list_google_model_options
from patientjournals.app.models import SubmitJobDraft
from patientjournals.app.settings_store import load_app_settings
from patientjournals.app.task_runner import TaskRunner
from patientjournals.app.workflows import WorkflowService, serializable


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PatientJournals</title>
  <style>
    :root { --bg:#FFFFFF; --accent:#00B2CA; --ink:#1E1E24; --line:#DDE7EA; --muted:#667276; --soft:#F5F8F9; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Arial, sans-serif; color:var(--ink); background:var(--bg); }
    .app { display:grid; grid-template-columns: 232px minmax(0,1fr); min-height:100vh; }
    aside { background:var(--ink); color:white; padding:22px 18px; }
    .brand { font-size:23px; font-weight:800; line-height:1.05; margin-bottom:26px; }
    nav button { width:100%; border:0; background:transparent; color:white; text-align:left; padding:14px 14px; margin:4px 0; font-size:16px; font-weight:700; cursor:pointer; }
    nav button.active { background:var(--accent); color:var(--ink); }
    main { padding:26px 30px; overflow:auto; }
    h1 { margin:0 0 6px; font-size:31px; letter-spacing:0; }
    .sub { color:var(--muted); margin-bottom:20px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:14px 0; }
    button, select, input, textarea { font:inherit; }
    .btn { border:0; background:var(--accent); color:var(--ink); padding:13px 18px; font-weight:800; cursor:pointer; min-height:46px; }
    .btn.secondary { background:var(--soft); border:1px solid var(--line); }
    .btn:disabled { opacity:.45; cursor:not-allowed; }
    .grid { display:grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap:12px; margin:14px 0 20px; }
    .metric { border:1px solid var(--line); padding:16px; background:white; }
    .metric strong { display:block; font-size:28px; margin-bottom:4px; }
    table { width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); }
    th, td { padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { background:var(--soft); font-size:13px; text-transform:uppercase; letter-spacing:0; }
    tr.selected { outline:3px solid var(--accent); outline-offset:-3px; }
    .split { display:grid; grid-template-columns: minmax(0,1.1fr) minmax(320px,.9fr); gap:16px; align-items:start; }
    .panel { border:1px solid var(--line); padding:16px; background:white; }
    .panel h2 { margin:0 0 12px; font-size:18px; }
    label { display:block; font-weight:700; margin:10px 0 6px; }
    input, select, textarea { width:100%; border:1px solid var(--line); padding:12px; min-height:44px; background:white; color:var(--ink); }
    textarea { min-height:88px; resize:vertical; }
    input[type="checkbox"], input[type="radio"] { width:auto; min-height:auto; }
    .select-cell { width:44px; text-align:center; }
    .clickable { cursor:pointer; }
    .muted { color:var(--muted); }
    details { border:1px solid var(--line); padding:12px; margin-top:12px; background:var(--soft); }
    summary { cursor:pointer; font-weight:800; }
    .inline-control { display:flex; gap:8px; align-items:center; }
    .inline-control select, .inline-control input { width:auto; }
    .notice { border:1px solid var(--line); background:var(--soft); padding:12px; margin:12px 0; }
    .pill { display:inline-flex; align-items:center; border:1px solid var(--line); background:var(--soft); padding:4px 8px; margin-right:6px; font-size:12px; font-weight:800; }
    .race { display:grid; gap:12px; }
    .race-row { display:grid; grid-template-columns: 132px minmax(0,1fr) 132px; gap:12px; align-items:center; }
    .race-name { min-width:0; }
    .race-name strong { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .race-name span, .race-score span { color:var(--muted); font-size:12px; }
    .race-lane { position:relative; height:42px; border:1px solid var(--line); background:var(--soft); overflow:hidden; }
    .race-fill { position:absolute; inset:0 auto 0 0; width:var(--pct); background:linear-gradient(90deg, rgba(0,178,202,.22), rgba(0,178,202,.72)); }
    .race-line { position:absolute; inset:0 18px 0 auto; width:3px; background:repeating-linear-gradient(to bottom, var(--ink) 0 6px, transparent 6px 12px); opacity:.25; }
    .race-car { position:absolute; left:calc(var(--pct) - 30px); top:8px; width:38px; height:20px; background:var(--ink); border:2px solid var(--accent); }
    .race-car::before { content:""; position:absolute; left:7px; top:-7px; width:18px; height:7px; background:var(--accent); }
    .race-wheel { position:absolute; bottom:-7px; width:9px; height:9px; border-radius:50%; background:var(--ink); border:2px solid var(--bg); }
    .race-wheel.left { left:5px; }
    .race-wheel.right { right:5px; }
    .race-score { text-align:right; }
    .race-row.demo { opacity:.72; }
    .race-row.demo .race-fill { background:repeating-linear-gradient(45deg, rgba(0,178,202,.18) 0 8px, rgba(0,178,202,.42) 8px 16px); }
    .race-row.demo .race-car { background:white; }
    .small-note { font-size:12px; color:var(--muted); margin:6px 0 12px; }
    .status { color:var(--muted); margin:10px 0; min-height:22px; white-space:pre-wrap; }
    .bad { color:#9A3412; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
    .validator { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; min-height:calc(100vh - 96px); }
    .image-stage { border:1px solid var(--line); background:var(--soft); overflow:auto; display:flex; align-items:flex-start; justify-content:center; padding:12px; min-height:520px; }
    .image-stage img { max-width:100%; height:auto; transform-origin:top center; background:white; box-shadow:0 1px 6px rgba(30,30,36,.12); }
    .decision-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .decision-grid .btn { width:100%; }
    .btn.accept { background:#16A34A; color:white; }
    .btn.partial { background:#84CC16; color:var(--ink); }
    .btn.reject { background:#DC2626; color:white; }
    .btn.unsure { background:#F59E0B; color:var(--ink); }
    .table-scroll { width:100%; overflow:auto; border:1px solid var(--line); }
    .table-scroll table { border:0; min-width:760px; }
    .schema-layout { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; align-items:start; }
    .schema-fields { display:grid; gap:8px; }
    .schema-field { display:grid; grid-template-columns:72px minmax(150px,1fr) 150px 100px 44px; gap:8px; align-items:start; padding:10px; border:1px solid var(--line); border-left:4px solid transparent; }
    .schema-field textarea { grid-column:2 / -1; min-height:64px; }
    .schema-field.diff-added { background:#ECFDF3; border-color:#86D7A0; border-left-color:#1A7F37; }
    .schema-field.diff-changed { background:#FFF8C5; border-color:#E7C75A; border-left-color:#9A6700; }
    .schema-change-label { display:inline-flex; align-items:center; justify-content:center; min-height:30px; padding:5px 7px; border:1px solid var(--line); background:var(--soft); color:var(--muted); font-size:11px; font-weight:800; text-transform:uppercase; }
    .schema-change-label.added { color:#116329; background:#DAFBE1; border-color:#86D7A0; }
    .schema-change-label.changed { color:#6F4B00; background:#FAE17D; border-color:#D4A72C; }
    .schema-change-label.removed { color:#A40E26; background:#FFEBE9; border-color:#FF8182; }
    .schema-diff-summary { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:10px 12px; margin:0 0 12px; border:1px solid var(--line); background:var(--soft); }
    .schema-diff-summary .pill { margin:0; }
    .schema-diff-summary .added { color:#116329; background:#DAFBE1; border-color:#86D7A0; }
    .schema-diff-summary .changed { color:#6F4B00; background:#FAE17D; border-color:#D4A72C; }
    .schema-diff-summary .removed { color:#A40E26; background:#FFEBE9; border-color:#FF8182; }
    .schema-original-row.diff-added td, tr.diff-added td { background:#ECFDF3; }
    .schema-original-row.diff-changed td, tr.diff-changed td { background:#FFF8C5; }
    .schema-original-row.diff-removed td, tr.diff-removed td { background:#FFEBE9; }
    .schema-original-row.diff-changed td:first-child { border-left:4px solid #9A6700; }
    .schema-original-row.diff-removed td:first-child { border-left:4px solid #CF222E; }
    .schema-original-row.diff-removed td:not(:last-child) { text-decoration:line-through; text-decoration-color:#CF222E; text-decoration-thickness:2px; }
    .schema-row-action { min-height:34px; padding:6px 10px; border:1px solid currentColor; background:white; color:var(--ink); font-weight:800; cursor:pointer; }
    .star-btn { width:44px; min-width:44px; padding:8px; font-size:22px; background:transparent; border:1px solid var(--line); cursor:pointer; }
    .star-btn.active { color:#00B2CA; background:#1E1E24; }
    .preview-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }
    .preview-item { border:1px solid var(--line); background:var(--soft); padding:8px; }
    .preview-item img { display:block; width:100%; height:190px; object-fit:contain; background:white; }
    .preview-item strong { display:block; margin-top:7px; overflow-wrap:anywhere; }
    .image-overlay { position:fixed; inset:0; z-index:20; background:rgba(30,30,36,.92); padding:24px; display:grid; grid-template-rows:auto minmax(0,1fr); }
    .image-overlay .toolbar { justify-content:flex-end; margin:0 0 12px; }
    .image-overlay img { width:100%; height:100%; object-fit:contain; }
    .diff-added { background:#ECFDF3; }
    .diff-removed { background:#FFEBE9; }
    .diff-changed { background:#FFF8C5; }
    @media (max-width: 900px) { .app { grid-template-columns:1fr; } aside { position:static; } .grid,.split,.validator,.schema-layout { grid-template-columns:1fr; } .schema-field { grid-template-columns:64px minmax(0,1fr) 44px; } .schema-field > .schema-change-label { grid-column:1; grid-row:1; } .schema-field > input { grid-column:2 / -1; grid-row:1; min-width:0; } .schema-field > select { grid-column:2; grid-row:2; min-width:0; } .schema-field > .inline-control { grid-column:2; grid-row:3; } .schema-field > .star-btn { grid-column:3; grid-row:2; } .schema-field textarea { grid-column:2 / -1; grid-row:4; } }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <div class="brand">Patient<br>Journals</div>
    <nav>
      <button data-tab="dashboard" class="active">Dashboard</button>
      <button data-tab="validate">Validate</button>
      <button data-tab="jobs">Jobs</button>
      <button data-tab="datasets">Datasets</button>
      <button data-tab="schemas">Schemas</button>
      <button data-tab="submit">Submit</button>
      <button data-tab="cloud">Cloud</button>
      <button data-tab="tasks">Tasks</button>
    </nav>
  </aside>
  <main id="main"></main>
</div>
<script>
const state = {
  tab: 'dashboard',
  jobs: [],
  datasets: [],
  datasetItems: [],
  localInputs: [],
  cloudInputs: [],
  selectedDataset: '',
  selectedDatasetKeys: new Set(),
  datasetsIncludeCloud: false,
  selectedJobIds: new Set(),
  selectedCloudPrefixes: new Set(),
  selectedLocalPath: '',
  selectedValidationCloudPrefixes: new Set(),
  validatorIdentity: null,
  validationSession: null,
  validationSample: null,
  validationZoom: 1,
  schemaVersions: [],
  selectedSchemaIds: new Set(),
  schemaEditorFields: [],
  schemaEditorParent: '',
  schemaEditorName: '',
  schemaEditorMakeActive: false,
  datasetInspect: null
};
const $ = (sel) => document.querySelector(sel);
async function api(path, opts={}) {
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function displayValue(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'object') {
    try { return JSON.stringify(v); } catch (_) { return String(v); }
  }
  return String(v);
}
function esc(v) { return displayValue(v).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function setStatus(text, bad=false) { const el = $('#status'); if (el) { el.textContent = text || ''; el.className = bad ? 'status bad' : 'status'; } }
function activate(tab) {
  state.tab = tab;
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  ({dashboard, validate, jobs, datasets, schemas, submit, cloud, tasks}[tab])();
}
document.querySelectorAll('nav button').forEach(b => b.onclick = () => activate(b.dataset.tab));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeDatasetImage();
});
function metric(label, value) { return `<div class="metric"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`; }
function table(headers, rows) {
  const body = rows.length ? rows.map(r=>`<tr>${r.map(c=>`<td>${esc(c)}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${headers.length}" class="muted">No rows.</td></tr>`;
  return `<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
}
function rawTable(headers, rowsHtml) {
  return `<table><thead><tr>${headers.join('')}</tr></thead><tbody>${rowsHtml || `<tr><td colspan="${headers.length}" class="muted">No rows.</td></tr>`}</tbody></table>`;
}
function selectedJobs() {
  return state.jobs.filter(j => state.selectedJobIds.has(j.job_id) && j.run_dir);
}
function shortText(value, max=34) {
  const text = displayValue(value);
  if (text.length <= max) return text;
  const head = Math.max(8, Math.floor((max - 3) * 0.55));
  const tail = Math.max(6, max - 3 - head);
  return `${text.slice(0, head)}...${text.slice(text.length - tail)}`;
}
function modelScore(value) {
  return value == null ? '-' : Number(value).toFixed(1) + '%';
}
function validationLeaderboard(entries) {
  const realRacers = (entries || []).map(item => ({...item, demo:false}));
  const demoRacers = [
    {rank:'Demo', validator_id:'demo_validator_100', validator_account:'Visual placeholder', decisions:100, runs:1, datasets:1, accuracy:null, demo:true},
    {rank:'Demo', validator_id:'demo_validator_50', validator_account:'Visual placeholder', decisions:50, runs:1, datasets:1, accuracy:null, demo:true}
  ];
  const racers = realRacers.length < 3 ? [...realRacers, ...demoRacers] : realRacers;
  if (!racers.length) return '<div class="muted">No synced validation columns yet.</div>';
  const maxDecisions = Math.max(1, ...racers.map(item => Number(item.decisions || 0)));
  racers.sort((a, b) => Number(b.decisions || 0) - Number(a.decisions || 0));
  return `<div class="race">${racers.slice(0, 8).map(item => {
    const decisions = Number(item.decisions || 0);
    const pct = Math.max(8, Math.min(100, decisions / maxDecisions * 100));
    const score = modelScore(item.accuracy);
    const rank = item.demo ? 'Demo' : `#${displayValue(item.rank || '')}`;
    const account = item.validator_account ? `<span>${esc(item.validator_account)}</span>` : `<span>${esc(item.runs || 0)} run(s), ${esc(item.datasets || 0)} dataset(s)</span>`;
    const scoreText = item.demo ? 'placeholder' : `model score ${score}`;
    return `<div class="race-row ${item.demo ? 'demo' : ''}">
      <div class="race-name"><span>${esc(rank)}</span><strong>${esc(item.validator_id || 'unknown')}</strong>${account}</div>
      <div class="race-lane" style="--pct:${pct}%"><div class="race-fill"></div><div class="race-line"></div><div class="race-car"><span class="race-wheel left"></span><span class="race-wheel right"></span></div></div>
      <div class="race-score"><strong>${esc(decisions)}</strong><span>validated columns &middot; ${esc(scoreText)}</span></div>
    </div>`;
  }).join('')}</div>`;
}
function validationRunsList(runs) {
  const rows = (runs || []).slice(0, 8).map(r => {
    const runId = displayValue(r.run_id || '');
    const dataset = displayValue(r.dataset_file || '');
    return `<tr>
      <td class="mono" title="${esc(runId)}">${esc(shortText(runId, 24))}</td>
      <td>${esc(shortText(r.validator_id || 'unknown', 22))}</td>
      <td title="${esc(dataset)}">${esc(shortText(dataset, 30))}</td>
      <td>${esc(r.model || 'Unknown')}</td>
      <td>${esc(modelScore(r.accuracy))}</td>
      <td>${esc(r.decisions || 0)}</td>
    </tr>`;
  }).join('');
  return rawTable(['<th>Run</th>','<th>Validator</th>','<th>Dataset</th>','<th>Model</th>','<th>Model score</th>','<th>Validated columns</th>'], rows);
}

async function dashboard() {
  $('#main').innerHTML = `<h1>Dashboard</h1><div class="sub">Research metrics, validation outcomes, and dataset inspection.</div><div id="status" class="status">Loading...</div><div id="dashboardBody"></div>`;
  try {
    const [summary, datasets, localInputs] = await Promise.all([api('/api/dashboard'), api('/api/datasets'), api('/api/local-inputs').catch(() => [])]);
    state.datasets = datasets.local || [];
    state.localInputs = localInputs || [];
    const options = state.datasets.map(d => `<option value="${esc(d.local_path || d.location)}">${esc(d.run_id || d.name)} - ${esc(d.name)} (${esc(d.row_count ?? '?')} rows)</option>`).join('');
    $('#dashboardBody').innerHTML = `
      <div class="grid">
        ${metric('Datasets', summary.dataset_count)}
        ${metric('Dataset rows', summary.dataset_rows)}
        ${metric('Synced validated columns', summary.shared_validation_count ?? 0)}
        ${metric('Processing records', summary.processing_record_count)}
      </div>
      <div class="split">
        <section class="panel">
          <h2>Analyze Dataset</h2>
          <select id="datasetSelect">${options}</select>
          <div class="toolbar"><button class="btn" onclick="analyzeSelected()">Analyze</button><button class="btn secondary" onclick="dashboard()">Refresh</button></div>
          <div id="analysis">${state.datasets.length ? '' : '<div class="muted">No local datasets found.</div>'}</div>
        </section>
        <section class="panel">
          <h2>Validation Leaderboard</h2>
          <div class="small-note">Cloud-synced validation runs only. Demo rows are visual placeholders when there are too few synced validators.</div>
          ${validationLeaderboard(summary.shared_validation_leaderboard || [])}
          <h2>Validation Runs</h2>
          ${validationRunsList(summary.shared_validation_runs || [])}
          <h2>Start Validation</h2>
          <div class="muted">Use the browser validator for local images or signed cloud image links.</div>
          <div class="toolbar"><button class="btn" onclick="openValidatorFromDashboard()">Open validator</button></div>
        </section>
      </div>`;
    setStatus('Dashboard loaded.');
    if (state.datasets.length) analyzeSelected();
  } catch (e) { setStatus(e.message, true); }
}
async function analyzeSelected() {
  const path = $('#datasetSelect')?.value || state.selectedDataset;
  if (!path) return;
  state.selectedDataset = path;
  $('#analysis').innerHTML = 'Analyzing...';
  try {
    const a = await api('/api/dataset/analyze?path=' + encodeURIComponent(path));
    $('#analysis').innerHTML = `
      <div class="grid">
        ${metric('Rows', a.row_count)}
        ${metric('Columns', a.column_count)}
        ${metric('Failed rows', a.failed_rows)}
        ${metric('Avg logprobs count', a.avg_logprobs?.count ?? 0)}
      </div>
      <h2>Least Complete Schema Fields</h2>
      ${table(['Field','Complete','Missing'], (a.schema_field_completeness || a.field_completeness || []).slice(0,12).map(f => [f.column, f.completeness.toFixed(1)+'%', f.missing]))}
      <h2>Metadata / Diagnostics Fields</h2>
      ${table(['Field','Complete','Missing'], (a.metadata_field_completeness || []).slice(0,12).map(f => [f.column, f.completeness.toFixed(1)+'%', f.missing]))}
      <h2>Failure Reasons</h2>
      ${table(['Reason','Rows'], Object.entries(a.failure_reasons || {}).map(([k,v]) => [k, v]))}
      <h2>Models and Schema Versions</h2>
      ${table(['Kind','Value','Rows'], [...Object.entries(a.model_counts || {}).map(([k,v]) => ['Model', k, v]), ...Object.entries(a.schema_version_counts || {}).map(([k,v]) => ['Schema version', k, v])])}
      <h2>Sample Rows</h2>
      ${table((a.columns || []).slice(0,8), (a.sample_rows || []).slice(0,8).map(r => (a.columns || []).slice(0,8).map(c => r[c] ?? '')))}
    `;
  } catch (e) { $('#analysis').innerHTML = `<div class="bad">${esc(e.message)}</div>`; }
}
function openValidatorFromDashboard() {
  const selected = $('#datasetSelect')?.value || state.selectedDataset;
  if (selected) state.selectedDataset = selected;
  activate('validate');
}

async function schemas() {
  $('#main').innerHTML = `<h1>Schemas</h1><div class="sub">Versioned extraction columns shared through the configured cloud bucket.</div><div id="status" class="status">Syncing schemas...</div>
    <div class="toolbar"><button class="btn" onclick="openNewSchema()">New schema</button><button id="viewSchemaButton" class="btn secondary" onclick="viewSelectedSchema()">View selected</button><button id="editSchemaButton" class="btn secondary" onclick="editSelectedSchema()">Edit selected</button><button id="compareSchemaButton" class="btn secondary" onclick="compareSelectedSchemas()">Compare two</button><button class="btn secondary" onclick="schemas()">Refresh</button></div>
    <div id="schemasBody"></div><div id="schemaWorkspace"></div>`;
  try {
    const data = await api('/api/schemas');
    state.schemaVersions = data.versions || [];
    state.selectedSchemaIds = new Set([...state.selectedSchemaIds].filter(id => state.schemaVersions.some(item => item.version_id === id)));
    renderSchemas();
    const sync = data.cloud_sync || {};
    setStatus(sync.error ? `Schemas loaded locally. Cloud sync failed: ${sync.error}` : `Loaded ${state.schemaVersions.length} version(s). Cloud: ${sync.status || 'not configured'}.`, Boolean(sync.error));
  } catch (e) { setStatus(e.message, true); }
}
function schemaById(id) { return state.schemaVersions.find(item => item.version_id === id); }
function renderSchemas() {
  const rows = state.schemaVersions.map((item, index) => {
    const checked = state.selectedSchemaIds.has(item.version_id) ? 'checked' : '';
    const selected = checked ? 'selected' : '';
    const activeTitle = item.is_active ? 'Active schema' : 'Mark this version active';
    return `<tr class="clickable ${selected}" onclick="toggleSchemaSelection(${index})">
      <td class="select-cell"><input type="checkbox" ${checked} onclick="event.stopPropagation(); toggleSchemaSelection(${index}, this.checked)"></td>
      <td class="select-cell"><button class="star-btn ${item.is_active ? 'active' : ''}" title="${esc(activeTitle)}" onclick="event.stopPropagation(); setActiveSchema('${esc(item.version_id)}')">${item.is_active ? '&#9733;' : '&#9734;'}</button></td>
      <td><strong>${esc(item.name)}</strong><div class="muted">Version ${esc(item.version_number)}</div></td>
      <td class="mono">${esc(item.version_id)}</td><td>${esc(item.field_count)}</td><td>${esc(item.created_at)}</td><td>${esc(item.created_by)}</td>
    </tr>`;
  }).join('');
  $('#schemasBody').innerHTML = `<div class="table-scroll">${rawTable(['<th class="select-cell"></th>','<th>Active</th>','<th>Schema</th>','<th>Version ID</th>','<th>Leaf columns</th>','<th>Created</th>','<th>Author</th>'], rows)}</div>`;
  const view = $('#viewSchemaButton');
  const edit = $('#editSchemaButton');
  const compare = $('#compareSchemaButton');
  if (view) view.disabled = state.selectedSchemaIds.size !== 1;
  if (edit) edit.disabled = state.selectedSchemaIds.size !== 1;
  if (compare) compare.disabled = state.selectedSchemaIds.size !== 2;
}
function toggleSchemaSelection(index, checked=null) {
  const item = state.schemaVersions[index];
  if (!item) return;
  const next = checked === null ? !state.selectedSchemaIds.has(item.version_id) : checked;
  if (next) {
    if (state.selectedSchemaIds.size >= 2) state.selectedSchemaIds.delete([...state.selectedSchemaIds][0]);
    state.selectedSchemaIds.add(item.version_id);
  } else state.selectedSchemaIds.delete(item.version_id);
  renderSchemas();
  setStatus(`${state.selectedSchemaIds.size} schema version(s) selected.`);
}
async function setActiveSchema(versionId) {
  try {
    const result = await api('/api/schemas/active', {method:'POST', body:JSON.stringify({version_id:versionId}), headers:{'Content-Type':'application/json'}});
    state.schemaVersions.forEach(item => item.is_active = item.version_id === versionId);
    renderSchemas();
    const sync = result.cloud_sync || {};
    setStatus(sync.error ? `Active locally. Cloud sync failed: ${sync.error}` : 'Active schema updated and synced.', Boolean(sync.error));
  } catch (e) { setStatus(e.message, true); }
}
function openNewSchema() {
  state.schemaEditorParent = '';
  state.schemaEditorName = 'NewSchema';
  state.schemaEditorMakeActive = false;
  state.schemaEditorFields = [{path:'new_column', type:'string', required:false, description:''}];
  renderSchemaEditor(null);
}
function editSelectedSchema() {
  const item = schemaById([...state.selectedSchemaIds][0]);
  if (!item) return setStatus('Select one schema version to edit.', true);
  state.schemaEditorParent = item.version_id;
  state.schemaEditorName = item.name;
  state.schemaEditorMakeActive = false;
  state.schemaEditorFields = (item.fields || []).map((field, index) => ({
    ...schemaFieldValue(field),
    _original: schemaFieldValue(field),
    _originalIndex: index
  }));
  renderSchemaEditor(item);
}
function viewSelectedSchema() {
  const item = schemaById([...state.selectedSchemaIds][0]);
  if (!item) return setStatus('Select one schema version to view.', true);
  $('#schemaWorkspace').innerHTML = `<section class="panel" style="margin-top:16px;"><h2>${esc(item.name)} version ${esc(item.version_number)}</h2>
    <div class="notice"><span class="pill">${item.is_active ? 'Active' : 'Version'}</span><span class="mono">${esc(item.version_id)}</span><div class="muted">Created ${esc(item.created_at)} by ${esc(item.created_by || 'unknown')}</div></div>
    <div class="table-scroll">${readonlySchemaFields(item.fields || [])}</div></section>`;
  $('#schemaWorkspace').scrollIntoView({behavior:'smooth', block:'start'});
}
function captureSchemaEditor() {
  const name = $('#schemaEditorName');
  const makeActive = $('#schemaMakeActive');
  if (name) state.schemaEditorName = name.value;
  if (makeActive) state.schemaEditorMakeActive = makeActive.checked;
  state.schemaEditorFields = state.schemaEditorFields.map((_field, index) => ({
    ..._field,
    path: $(`#schemaPath${index}`)?.value || '',
    type: $(`#schemaType${index}`)?.value || 'string',
    required: $(`#schemaRequired${index}`)?.checked || false,
    description: $(`#schemaDescription${index}`)?.value || ''
  }));
}
function schemaFieldValue(field) {
  return {
    path: field?.path || '',
    type: field?.type || 'string',
    required: Boolean(field?.required),
    description: field?.description || ''
  };
}
function schemaFieldsMatch(left, right) {
  const a = schemaFieldValue(left), b = schemaFieldValue(right);
  return a.path === b.path && a.type === b.type && a.required === b.required && a.description === b.description;
}
function schemaFieldChangeKind(field) {
  if (!field?._original) return 'added';
  return schemaFieldsMatch(field, field._original) ? '' : 'changed';
}
function schemaChangeLabel(kind) {
  return kind === 'added' ? '+ Added' : kind === 'changed' ? '~ Edited' : kind === 'removed' ? '- Deleted' : 'Same';
}
function schemaFieldEditor(field, index) {
  const types = ['string','date','integer','number','boolean','list[string]','list[integer]','list[number]','list[boolean]'];
  const kind = schemaFieldChangeKind(field);
  return `<div id="schemaField${index}" class="schema-field ${kind ? `diff-${kind}` : ''}">
    <span id="schemaFieldStatus${index}" class="schema-change-label ${kind}">${schemaChangeLabel(kind)}</span>
    <input id="schemaPath${index}" value="${esc(field.path)}" aria-label="Column path" oninput="updateSchemaField(${index})">
    <select id="schemaType${index}" aria-label="Column type" onchange="updateSchemaField(${index})">${types.map(type => `<option value="${type}" ${type === field.type ? 'selected' : ''}>${type}</option>`).join('')}</select>
    <label class="inline-control"><input id="schemaRequired${index}" type="checkbox" ${field.required ? 'checked' : ''} onchange="updateSchemaField(${index})"> Required</label>
    <button class="star-btn" title="Remove column" onclick="removeSchemaField(${index})">&times;</button>
    <textarea id="schemaDescription${index}" placeholder="What should the model transcribe for this column?" oninput="updateSchemaField(${index})">${esc(field.description)}</textarea>
  </div>`;
}
function originalSchemaFieldState(index) {
  const current = state.schemaEditorFields.find(field => field._originalIndex === index);
  return current ? schemaFieldChangeKind(current) : 'removed';
}
function readonlySchemaFields(fields, showEditorDiff=false) {
  if (!showEditorDiff) return table(['Column','Type','Required','Description'], (fields || []).map(field => [field.path, field.type, field.required ? 'Yes' : 'No', field.description || '']));
  const rows = (fields || []).map((field, index) => {
    const kind = originalSchemaFieldState(index);
    const action = kind === 'removed'
      ? `<button class="schema-row-action" onclick="undoSchemaField(${index})">Undo</button>`
      : kind === 'changed' ? `<button class="schema-row-action" onclick="resetSchemaField(${index})">Reset</button>` : '';
    return `<tr class="schema-original-row ${kind ? `diff-${kind}` : ''}"><td><span class="schema-change-label ${kind}">${schemaChangeLabel(kind)}</span></td><td class="mono">${esc(field.path)}</td><td>${esc(field.type)}</td><td>${field.required ? 'Yes' : 'No'}</td><td>${esc(field.description || '')}</td><td>${action}</td></tr>`;
  }).join('');
  return rawTable(['<th>Change</th>','<th>Column</th>','<th>Type</th>','<th>Required</th>','<th>Description</th>','<th></th>'], rows);
}
function schemaDiffCounts() {
  const added = state.schemaEditorFields.filter(field => schemaFieldChangeKind(field) === 'added').length;
  const changed = state.schemaEditorFields.filter(field => schemaFieldChangeKind(field) === 'changed').length;
  const parent = schemaById(state.schemaEditorParent);
  const retained = new Set(state.schemaEditorFields.map(field => field._originalIndex).filter(index => Number.isInteger(index)));
  const removed = (parent?.fields || []).filter((_field, index) => !retained.has(index)).length;
  return {added, changed, removed};
}
function schemaDiffSummary() {
  const counts = schemaDiffCounts();
  if (!counts.added && !counts.changed && !counts.removed) return '<span class="muted">No changes yet.</span>';
  return `<strong>Changes</strong><span class="pill added">+ ${counts.added} added</span><span class="pill changed">~ ${counts.changed} edited</span><span class="pill removed">- ${counts.removed} deleted</span>`;
}
function renderSchemaEditor(parent) {
  const original = parent ? readonlySchemaFields(parent.fields || [], true) : '<div class="muted">This is a new schema.</div>';
  const nameControl = parent ? `<input id="schemaEditorName" value="${esc(state.schemaEditorName)}" readonly>` : `<input id="schemaEditorName" value="${esc(state.schemaEditorName)}" placeholder="Schema name">`;
  $('#schemaWorkspace').innerHTML = `<section class="panel" style="margin-top:16px;"><h2>${parent ? `Create version ${Number(parent.version_number || 0) + 1}` : 'Create schema'}</h2>
    <div id="schemaDiffSummary" class="schema-diff-summary">${schemaDiffSummary()}</div>
    <div class="schema-layout"><div><h2>Original reference</h2><div id="schemaOriginalFields" class="table-scroll">${original}</div></div><div><h2>New version</h2><label>Name</label>${nameControl}<div class="schema-fields">${state.schemaEditorFields.map(schemaFieldEditor).join('')}</div>
      <div class="toolbar"><button class="btn secondary" onclick="addSchemaField()">Add column</button></div><label><input id="schemaMakeActive" type="checkbox" ${state.schemaEditorMakeActive ? 'checked' : ''}> Mark this version active</label></div></div>
    <div class="toolbar"><button class="btn" onclick="saveSchemaVersion()">Save new version</button><button class="btn secondary" onclick="$('#schemaWorkspace').innerHTML=''">Cancel</button></div></section>`;
  $('#schemaWorkspace').scrollIntoView({behavior:'smooth', block:'start'});
}
function updateSchemaField(_index) {
  captureSchemaEditor();
  state.schemaEditorFields.forEach((field, index) => {
    const kind = schemaFieldChangeKind(field);
    const row = $(`#schemaField${index}`);
    const status = $(`#schemaFieldStatus${index}`);
    if (row) row.className = `schema-field ${kind ? `diff-${kind}` : ''}`;
    if (status) {
      status.className = `schema-change-label ${kind}`;
      status.textContent = schemaChangeLabel(kind);
    }
  });
  const parent = schemaById(state.schemaEditorParent);
  const original = $('#schemaOriginalFields');
  if (original && parent) original.innerHTML = readonlySchemaFields(parent.fields || [], true);
  const summary = $('#schemaDiffSummary');
  if (summary) summary.innerHTML = schemaDiffSummary();
}
function addSchemaField() {
  captureSchemaEditor();
  state.schemaEditorFields.push({path:'', type:'string', required:false, description:''});
  renderSchemaEditor(schemaById(state.schemaEditorParent));
}
function removeSchemaField(index) {
  captureSchemaEditor();
  state.schemaEditorFields.splice(index, 1);
  renderSchemaEditor(schemaById(state.schemaEditorParent));
}
function resetSchemaField(originalIndex) {
  captureSchemaEditor();
  const parent = schemaById(state.schemaEditorParent);
  const original = parent?.fields?.[originalIndex];
  const currentIndex = state.schemaEditorFields.findIndex(field => field._originalIndex === originalIndex);
  if (!original || currentIndex < 0) return;
  state.schemaEditorFields[currentIndex] = {...schemaFieldValue(original), _original:schemaFieldValue(original), _originalIndex:originalIndex};
  renderSchemaEditor(parent);
}
function undoSchemaField(originalIndex) {
  captureSchemaEditor();
  const parent = schemaById(state.schemaEditorParent);
  const original = parent?.fields?.[originalIndex];
  if (!original || state.schemaEditorFields.some(field => field._originalIndex === originalIndex)) return;
  const restored = {...schemaFieldValue(original), _original:schemaFieldValue(original), _originalIndex:originalIndex};
  const insertAt = state.schemaEditorFields.findIndex(field => !Number.isInteger(field._originalIndex) || field._originalIndex > originalIndex);
  if (insertAt < 0) state.schemaEditorFields.push(restored);
  else state.schemaEditorFields.splice(insertAt, 0, restored);
  renderSchemaEditor(parent);
}
async function saveSchemaVersion() {
  captureSchemaEditor();
  setStatus('Saving and syncing schema version...');
  try {
    const result = await api('/api/schemas/version', {method:'POST', body:JSON.stringify({
      name: state.schemaEditorName,
      parent_version_id: state.schemaEditorParent,
      fields: state.schemaEditorFields.map(schemaFieldValue),
      make_active: state.schemaEditorMakeActive
    }), headers:{'Content-Type':'application/json'}});
    state.selectedSchemaIds = new Set([result.version.version_id]);
    await schemas();
    const sync = result.cloud_sync || {};
    setStatus(sync.error ? `Version saved locally. Cloud sync failed: ${sync.error}` : `Saved ${result.version.name} version ${result.version.version_number} and synced it.`, Boolean(sync.error));
  } catch (e) { setStatus(e.message, true); }
}
function compareSelectedSchemas() {
  const items = [...state.selectedSchemaIds].map(schemaById).filter(Boolean);
  if (items.length !== 2) return setStatus('Select exactly two schema versions.', true);
  const [left, right] = items;
  const leftMap = new Map((left.fields || []).map(field => [field.path, field]));
  const rightMap = new Map((right.fields || []).map(field => [field.path, field]));
  const paths = [...new Set([...leftMap.keys(), ...rightMap.keys()])].sort();
  const rows = paths.map(path => {
    const a = leftMap.get(path), b = rightMap.get(path);
    const changed = a && b && (a.type !== b.type || a.required !== b.required || a.description !== b.description);
    const cls = !a ? 'diff-added' : !b ? 'diff-removed' : changed ? 'diff-changed' : '';
    const show = field => field ? `${field.type}${field.required ? ' (required)' : ''}${field.description ? ` - ${shortText(field.description, 70)}` : ''}` : '-';
    return `<tr class="${cls}"><td class="mono">${esc(path)}</td><td>${esc(show(a))}</td><td>${esc(show(b))}</td></tr>`;
  }).join('');
  $('#schemaWorkspace').innerHTML = `<section class="panel" style="margin-top:16px;"><h2>Schema comparison</h2><div class="small-note">Green: added in the right version. Red: removed. Yellow: changed.</div><div class="table-scroll">${rawTable(['<th>Column</th>',`<th>${esc(left.name)} v${esc(left.version_number)}</th>`,`<th>${esc(right.name)} v${esc(right.version_number)}</th>`], rows)}</div></section>`;
  $('#schemaWorkspace').scrollIntoView({behavior:'smooth', block:'start'});
}

async function validate() {
  $('#main').innerHTML = `<h1>Validate</h1><div class="sub">Browser validation with signed cloud image links and autosaved decisions.</div><div id="status" class="status">Loading...</div><div id="validateBody"></div>`;
  try {
    const [datasets, localInputs, identity] = await Promise.all([
      api('/api/datasets'),
      api('/api/local-inputs').catch(() => []),
      api('/api/validation/identity').catch(() => ({username:'unknown', account:'', source:''}))
    ]);
    state.datasets = datasets.local || [];
    state.localInputs = localInputs || [];
    state.validatorIdentity = identity || {username:'unknown', account:'', source:''};
    if (!state.selectedDataset && state.datasets.length) state.selectedDataset = state.datasets[0].local_path || state.datasets[0].location;
    if (!state.selectedLocalPath && state.localInputs.length) state.selectedLocalPath = state.localInputs[0].path;
    renderValidationSetup();
    setStatus('Validator ready.');
  } catch (e) { setStatus(e.message, true); }
}
function validationDatasetOptions() {
  return (state.datasets || []).map(d => {
    const value = d.gcs_uri || d.local_path || d.location;
    const selected = value === state.selectedDataset ? 'selected' : '';
    const provenance = [d.model || 'Unknown model', d.schema_name || '', d.schema_version_id ? shortText(d.schema_version_id, 18) : ''].filter(Boolean).join(' / ');
    return `<option value="${esc(value)}" ${selected}>${esc(d.source)} - ${esc(d.run_id || d.name)} (${esc(d.row_count ?? '?')} rows) - ${esc(provenance)}</option>`;
  }).join('');
}
function renderValidationSetup() {
  const identity = state.validatorIdentity || {username:'unknown', account:'', source:''};
  const account = identity.account || identity.username || 'unknown';
  const source = identity.source ? `Detected from ${identity.source}.` : 'Detected automatically.';
  $('#validateBody').innerHTML = `<section class="panel">
    <h2>Start Browser Validation</h2>
    <label>Dataset</label><select id="browserValidationDataset" onchange="state.selectedDataset=this.value">${validationDatasetOptions()}</select>
    <div class="toolbar"><button class="btn secondary" onclick="loadValidationSharedDatasets()">Load shared datasets</button><button class="btn secondary" onclick="validate()">Refresh local</button></div>
    <div class="notice"><span class="pill">Validator</span><div><strong>${esc(identity.username || 'unknown')}</strong></div><div class="muted mono">${esc(account)}</div><div class="muted">${esc(source)}</div></div>
    <label>Sampling mode</label><select id="browserSamplingMode"><option value="balanced_ucb">Balanced UCB</option><option value="random">True random</option></select>
    <label>Image source</label><select id="browserImageSource" onchange="renderValidationImageSource()"><option value="cloud">Cloud bucket, match by image_name</option><option value="local">Local folder</option></select>
    <div id="validationImageSourceBody"></div>
    <details><summary>Advanced</summary><label><input id="browserCorrections" type="checkbox" checked> Enable correction entry</label><label><input id="browserOfflineMode" type="checkbox"> Offline mode, do not upload validation run</label><label>Custom local image folder</label><input id="browserCustomLocalImages" placeholder="Only used when local folder is selected"></details>
    <div class="toolbar"><button class="btn" onclick="startBrowserValidation()">Start validation</button></div>
  </section>`;
  renderValidationImageSource();
}
async function loadValidationSharedDatasets() {
  setStatus('Loading shared datasets...');
  try {
    const data = await api('/api/datasets?cloud=1');
    state.datasets = [...(data.local || []), ...(data.cloud || [])];
    if (!state.selectedDataset && state.datasets.length) state.selectedDataset = state.datasets[0].gcs_uri || state.datasets[0].local_path || state.datasets[0].location;
    renderValidationSetup();
    setStatus(`Loaded ${(data.cloud || []).length} shared dataset(s).`);
  } catch (e) { setStatus(e.message, true); }
}
function renderValidationImageSource() {
  const source = $('#browserImageSource')?.value || 'cloud';
  if (source === 'local') {
    const rows = state.localInputs.map((item, i) => {
      const checked = state.selectedLocalPath === item.path ? 'checked' : '';
      return `<tr class="clickable ${checked ? 'selected' : ''}" onclick="selectValidationLocalInput(${i})">
        <td class="select-cell"><input type="radio" name="validationLocalInput" ${checked} onclick="event.stopPropagation(); selectValidationLocalInput(${i})"></td>
        <td>${esc(item.name)}</td><td>${esc(item.image_count)}</td><td>${esc(item.updated_at)}</td><td class="mono">${esc(item.path)}</td>
      </tr>`;
    }).join('');
    $('#validationImageSourceBody').innerHTML = `<label>Local image folder</label>${rawTable(['<th class="select-cell"></th>','<th>Name</th>','<th>Images</th>','<th>Updated</th>','<th>Path</th>'], rows)}`;
    return;
  }
  const body = $('#validationImageSourceBody');
  if (!body) return;
  body.innerHTML = `<label>Cloud images</label><div class="notice"><span class="pill">Automatic</span><div>Images are matched from the configured GCS bucket by each dataset row's <span class="mono">image_name</span>.</div><div class="muted">Only the active image, plus a small lookahead batch, is resolved while validating. Signed image links are not stored.</div></div>`;
}
function selectValidationLocalInput(index) {
  const item = state.localInputs[index];
  if (item) state.selectedLocalPath = item.path;
  renderValidationImageSource();
}
async function startBrowserValidation() {
  const results = $('#browserValidationDataset')?.value || state.selectedDataset;
  const imageSource = $('#browserImageSource')?.value || 'cloud';
  let images = '';
  let cloudPrefixes = [];
  if (!results) return setStatus('Select a dataset first.', true);
  if (imageSource === 'local') {
    images = ($('#browserCustomLocalImages')?.value || '').trim() || state.selectedLocalPath || '';
    if (!images) return setStatus('Select a local image folder.', true);
  } else {
    cloudPrefixes = [];
  }
  setStatus('Starting browser validation...');
  try {
    const sample = await api('/api/validation/session/start', {
      method:'POST',
      body: JSON.stringify({
        results,
        image_source: imageSource,
        images,
        cloud_prefixes: cloudPrefixes,
        corrections: $('#browserCorrections')?.checked ?? true,
        offline: $('#browserOfflineMode')?.checked || false,
        sampling_mode: $('#browserSamplingMode')?.value || 'balanced_ucb'
      }),
      headers:{'Content-Type':'application/json'}
    });
    state.validationSession = sample.session_id;
    renderValidationSample(sample);
  } catch (e) { setStatus(e.message, true); }
}
function renderValidationSample(sample) {
  state.validationSample = sample;
  state.validationSession = sample.session_id || state.validationSession;
  state.validationZoom = 1;
  const finishLabel = sample.offline_mode ? 'Save locally' : 'Save and sync';
  if (sample.status === 'complete') {
    const saveAction = sample.saved ? `<button class="btn" disabled>Saved</button>` : `<button class="btn" onclick="finishBrowserValidation()">${finishLabel}</button>`;
    $('#main').innerHTML = `<h1>Validate</h1><div class="sub">Validation complete.</div><div id="status" class="status"></div>
      <section class="panel">
        <div class="grid">${metric('Decisions', sample.decisions)}${metric('Total pairs', sample.total_pairs)}${metric('Remaining', sample.remaining_pairs)}</div>
        <div class="toolbar">${saveAction}<button class="btn secondary" onclick="validate()">New validation</button></div>
        <div class="mono">${esc(sample.run_id || '')}</div>
        <div class="mono">${esc(sample.csv_path || '')}</div>
      </section>`;
    setStatus('All available datapoints are complete.');
    return;
  }
  $('#main').innerHTML = `<h1>Validate</h1><div class="sub">${esc(sample.dataset_file)} - ${esc(sample.sampling_mode)} - ${esc(sample.decisions)} decisions</div><div id="status" class="status"></div>
    <div class="validator">
      <section>
        <div class="toolbar">
          <button class="btn secondary" onclick="setValidationZoom(.8)">-</button>
          <button class="btn secondary" onclick="setValidationZoom(1, true)">Fit</button>
          <button class="btn secondary" onclick="setValidationZoom(1.25)">+</button>
          <button class="btn secondary" onclick="refreshValidationSample()">Refresh image link</button>
        </div>
        <div class="image-stage"><img id="validationImage" src="${esc(sample.image_url)}" alt="${esc(sample.image_name)}"></div>
      </section>
      <section class="panel">
        <h2>${esc(sample.image_name)}</h2>
        <div class="notice"><span class="pill">Validator</span><strong>${esc(sample.validator_id || 'unknown')}</strong>${sample.offline_mode ? '<span class="pill">Offline</span>' : ''}<div class="muted mono">${esc(sample.validator_account || '')}</div></div>
        <div class="notice"><span class="pill">Model</span><strong>${esc(sample.model || 'Unknown')}</strong><div class="muted">${esc(sample.schema_name || '')} <span class="mono">${esc(sample.schema_version_id || '')}</span></div></div>
        <div class="muted mono">${esc(sample.image_source)} - ${esc(sample.image_uri)}</div>
        <div class="grid" style="grid-template-columns:1fr 1fr; margin:12px 0;">
          ${metric('Decisions', sample.decisions)}
          ${metric('Remaining', sample.remaining_pairs)}
        </div>
        <label>Field</label><input value="${esc(sample.field_name)}" readonly>
        <label>Model transcription</label><textarea readonly>${esc(sample.field_value)}</textarea>
        <label>Corrected value</label><textarea id="validationCorrection" ${sample.allow_corrections ? '' : 'readonly'} placeholder="Edit this value before saving a correction">${esc(sample.correction_value)}</textarea>
        <div class="decision-grid">
          <button class="btn accept" onclick="markValidation('accept')">Accept</button>
          <button class="btn partial" onclick="markValidation('somewhat_accept')">Somewhat</button>
          <button class="btn reject" onclick="markValidation('reject')">Reject</button>
          <button class="btn unsure" onclick="markValidation('unsure')">Unsure</button>
        </div>
        <div class="toolbar"><button class="btn" onclick="markValidation('corrected')" ${sample.allow_corrections ? '' : 'disabled'}>Save correction</button><button class="btn secondary" onclick="finishBrowserValidation()">Save and exit</button></div>
      </section>
    </div>`;
  setStatus('Signed URLs are short-lived and are regenerated per sample.');
}
function setValidationZoom(factor, absolute=false) {
  const img = $('#validationImage');
  if (!img) return;
  state.validationZoom = absolute ? 1 : Math.max(.2, Math.min(5, state.validationZoom * factor));
  img.style.transform = `scale(${state.validationZoom})`;
}
async function refreshValidationSample() {
  if (!state.validationSession) return;
  try {
    const sample = await api('/api/validation/session?session_id=' + encodeURIComponent(state.validationSession));
    renderValidationSample(sample);
  } catch (e) { setStatus(e.message, true); }
}
async function markValidation(label) {
  if (!state.validationSession) return setStatus('No active validation session.', true);
  try {
    const sample = await api('/api/validation/session/mark', {
      method:'POST',
      body: JSON.stringify({
        session_id: state.validationSession,
        label,
        corrected_field: $('#validationCorrection')?.value || ''
      }),
      headers:{'Content-Type':'application/json'}
    });
    renderValidationSample(sample);
  } catch (e) { setStatus(e.message, true); }
}
async function finishBrowserValidation() {
  if (!state.validationSession) return;
  try {
    const result = await api('/api/validation/session/finish', {
      method:'POST',
      body: JSON.stringify({ session_id: state.validationSession }),
      headers:{'Content-Type':'application/json'}
    });
    renderValidationSample(result);
    const upload = result.uploaded?.validation_csv_uri ? ` Synced: ${result.uploaded.validation_csv_uri}` : '';
    const skipped = result.upload_skipped_reason === 'offline_mode' ? ' Offline mode: not uploaded.' : '';
    const uploadError = result.upload_error ? ` Upload failed, but local files were saved: ${result.upload_error}` : '';
    setStatus(`Saved ${result.decisions} validation decision(s).${upload}${skipped}${uploadError}`, Boolean(result.upload_error));
  } catch (e) { setStatus(e.message, true); }
}

async function jobs() {
  $('#main').innerHTML = `<h1>Jobs</h1><div class="sub">SQLite-backed job state and grouped retrieval actions.</div><div id="status" class="status">Loading...</div>
  <div class="toolbar">
    <button class="btn" onclick="jobs()">Refresh</button>
    <button class="btn secondary" onclick="retrieveSelectedJobs()">Retrieve selected</button>
    <button class="btn secondary" onclick="jobAction('recover')">Recover API</button>
    <button class="btn secondary" onclick="jobAction('finalize')">Finalize Failed</button>
    <label class="inline-control"><input id="ignoreFailed" type="checkbox"> Ignore failed</label>
    <label class="inline-control">Duplicates <select id="duplicateStrategy"><option value="first_successful">First successful</option><option value="provide_all">Provide all</option></select></label>
  </div><div id="jobsBody"></div>`;
  try {
    state.jobs = await api('/api/jobs');
    state.selectedJobIds = new Set([...state.selectedJobIds].filter(id => state.jobs.some(j => j.job_id === id)));
    renderJobs();
    setStatus(`${state.jobs.length} job(s). ${state.selectedJobIds.size} selected.`);
  } catch (e) { setStatus(e.message, true); }
}
function renderJobs() {
  const allChecked = state.jobs.length > 0 && state.jobs.every(j => state.selectedJobIds.has(j.job_id));
  const rows = state.jobs.map((j, i) => {
    const checked = state.selectedJobIds.has(j.job_id) ? 'checked' : '';
    const selected = checked ? 'selected' : '';
    return `<tr class="clickable ${selected}" onclick="toggleJob(${i})">
      <td class="select-cell"><input type="checkbox" ${checked} onclick="event.stopPropagation(); toggleJob(${i}, this.checked)"></td>
      <td>${esc(j.created_at)}</td><td>${esc(j.model)}</td><td>${esc(j.schema_name || '')}<div class="muted mono" title="${esc(j.schema_version_id || '')}">${esc(shortText(j.schema_version_id || '', 24))}</div></td><td>${esc(j.input_location)}</td>
      <td>${esc(j.image_count)}</td><td>${esc(j.status)}</td><td>${esc(j.succeeded ?? '')}</td><td>${esc(j.failed ?? '')}</td>
    </tr>`;
  }).join('');
  $('#jobsBody').innerHTML = rawTable(
    [`<th class="select-cell"><input type="checkbox" ${allChecked ? 'checked' : ''} onchange="toggleAllJobs(this.checked)"></th>`, '<th>Created</th>', '<th>Model</th>', '<th>Schema version</th>', '<th>Input</th>', '<th>Images</th>', '<th>Status</th>', '<th>Success</th>', '<th>Missing</th>'],
    rows
  );
}
function toggleJob(index, checked=null) {
  const job = state.jobs[index];
  if (!job) return;
  const next = checked === null ? !state.selectedJobIds.has(job.job_id) : checked;
  if (next) state.selectedJobIds.add(job.job_id); else state.selectedJobIds.delete(job.job_id);
  renderJobs();
  setStatus(`${state.selectedJobIds.size} job(s) selected.`);
}
function toggleAllJobs(checked) {
  state.selectedJobIds = checked ? new Set(state.jobs.map(j => j.job_id)) : new Set();
  renderJobs();
  setStatus(`${state.selectedJobIds.size} job(s) selected.`);
}
async function retrieveSelectedJobs() {
  const rows = selectedJobs();
  if (!rows.length) return setStatus('Select at least one job.', true);
  const body = {
    run_dirs: rows.map(j => j.run_dir),
    ignore_failed: $('#ignoreFailed')?.checked || false,
    duplicate_strategy: $('#duplicateStrategy')?.value || 'first_successful'
  };
  try {
    const task = await api('/api/jobs/retrieve-many', { method:'POST', body: JSON.stringify(body), headers:{'Content-Type':'application/json'} });
    setStatus(`Started retrieve task ${task.task_id} for ${rows.length} job(s).`);
    activate('tasks');
  } catch (e) { setStatus(e.message, true); }
}
async function jobAction(action) {
  const rows = selectedJobs();
  if (!rows.length) return setStatus('Select at least one job.', true);
  const map = { recover:'/api/jobs/recover-api', finalize:'/api/jobs/finalize-failed' };
  try {
    const started = await Promise.all(rows.map(j => api(map[action], { method:'POST', body: JSON.stringify({ run_dir: j.run_dir }), headers:{'Content-Type':'application/json'} })));
    setStatus(`Started ${started.length} ${action} task(s).`);
    activate('tasks');
  } catch (e) { setStatus(e.message, true); }
}

async function datasets() {
  state.datasetsIncludeCloud = false;
  $('#main').innerHTML = `<h1>Datasets</h1><div class="sub">Combine local and shared datasets into a named research dataset.</div><div id="status" class="status">Loading...</div>
  <section class="panel">
    <div class="toolbar"><button class="btn" onclick="inspectSelectedDataset()">Inspect selected</button><button class="btn secondary" onclick="datasets()">Refresh Local</button><button class="btn secondary" onclick="loadCloudDatasets()">Load Shared</button></div>
    <label>New dataset name</label><input id="combinedDatasetName" placeholder="for example: journals_0618_complete">
    <label>Duplicates</label><select id="datasetDuplicateStrategy"><option value="first_successful">One row per image, prefer first successful</option><option value="provide_all">Include all rows</option></select>
    <div class="toolbar"><button class="btn" onclick="combineSelectedDatasets()">Combine selected</button></div>
  </section>
  <div id="datasetsBody"></div><div id="datasetInspector"></div>`;
  try {
    const data = await api('/api/datasets');
    state.datasetItems = data.local || [];
    state.datasets = state.datasetItems;
    renderDatasets(state.datasetItems);
    setStatus(`${state.datasets.length} local dataset(s).`);
  } catch (e) { setStatus(e.message, true); }
}
async function loadCloudDatasets() {
  try {
    state.datasetsIncludeCloud = true;
    const data = await api('/api/datasets?cloud=1');
    state.datasetItems = [...(data.local || []), ...(data.cloud || [])];
    state.datasets = state.datasetItems;
    renderDatasets(state.datasetItems);
    setStatus(`Loaded ${(data.cloud || []).length} shared dataset(s).`);
  } catch (e) { setStatus(e.message, true); }
}
function datasetKey(item) {
  return item.gcs_uri || item.local_path || item.location || item.name || '';
}
function selectedDatasetItems() {
  return state.datasetItems.filter(item => state.selectedDatasetKeys.has(datasetKey(item)));
}
function renderDatasets(items) {
  state.selectedDatasetKeys = new Set([...state.selectedDatasetKeys].filter(key => items.some(item => datasetKey(item) === key)));
  const allChecked = items.length > 0 && items.every(item => state.selectedDatasetKeys.has(datasetKey(item)));
  const rows = items.map((d, i) => {
    const key = datasetKey(d);
    const checked = state.selectedDatasetKeys.has(key) ? 'checked' : '';
    return `<tr class="clickable ${checked ? 'selected' : ''}" onclick="toggleDataset(${i})" ondblclick="inspectDataset(${i})">
      <td class="select-cell"><input type="checkbox" ${checked} onclick="event.stopPropagation(); toggleDataset(${i}, this.checked)"></td>
      <td>${esc(d.source)}</td><td>${esc(d.name)}</td><td>${esc(d.row_count ?? '')}</td><td><strong>${esc(d.model || 'Unknown')}</strong></td><td>${esc(d.schema_name || '')}<div class="muted mono">${esc(shortText(d.schema_version_id || '', 24))}</div></td><td>${esc(d.updated_at)}</td><td>${esc(d.run_id)}</td><td class="mono" title="${esc(d.location)}">${esc(shortText(d.location, 42))}</td>
    </tr>`;
  }).join('');
  $('#datasetsBody').innerHTML = rawTable(
    [`<th class="select-cell"><input type="checkbox" ${allChecked ? 'checked' : ''} onchange="toggleAllDatasets(this.checked)"></th>`, '<th>Source</th>', '<th>Name</th>', '<th>Rows</th>', '<th>Model</th>', '<th>Schema</th>', '<th>Updated</th>', '<th>Run</th>', '<th>Location</th>'],
    rows
  );
}
function inspectSelectedDataset() {
  const items = selectedDatasetItems();
  if (items.length !== 1) return setStatus('Select exactly one dataset to inspect.', true);
  inspectDataset(state.datasetItems.indexOf(items[0]));
}
async function inspectDataset(index, offset=0) {
  const item = state.datasetItems[index];
  if (!item) return;
  const location = datasetKey(item);
  const target = $('#datasetInspector');
  if (target) target.innerHTML = '<section class="panel" style="margin-top:16px;">Loading dataset rows...</section>';
  try {
    const page = await api(`/api/dataset/inspect?path=${encodeURIComponent(location)}&offset=${Number(offset || 0)}&limit=50`);
    page.itemIndex = index;
    state.datasetInspect = page;
    renderDatasetInspector();
    setStatus(`Inspecting rows ${page.offset + 1}-${page.offset + page.rows.length} of ${page.total_rows}. Double-click a row to open its page image.`);
  } catch (e) { if (target) target.innerHTML = `<div class="bad">${esc(e.message)}</div>`; setStatus(e.message, true); }
}
function renderDatasetInspector() {
  const page = state.datasetInspect;
  const target = $('#datasetInspector');
  if (!page || !target) return;
  const columns = page.columns || [];
  const rows = (page.rows || []).map((row, index) => `<tr class="clickable" ondblclick="openDatasetImageRow(${index})">${columns.map(column => `<td title="${esc(displayValue(row[column]))}">${esc(shortText(row[column], 70))}</td>`).join('')}</tr>`).join('');
  target.innerHTML = `<section class="panel" style="margin-top:16px;"><h2>${esc(page.dataset_label || 'Dataset inspection')}</h2><div class="small-note">Model and schema metadata are shown first when available. Double-click any row to open the corresponding scanned page.</div>
    <div class="table-scroll">${rawTable(columns.map(column => `<th>${esc(column)}</th>`), rows)}</div>
    <div class="toolbar"><button class="btn secondary" ${page.has_previous ? '' : 'disabled'} onclick="inspectDataset(${page.itemIndex}, ${Math.max(0, page.offset - page.limit)})">Previous</button><span class="muted">${page.offset + 1}-${page.offset + page.rows.length} of ${page.total_rows}</span><button class="btn secondary" ${page.has_next ? '' : 'disabled'} onclick="inspectDataset(${page.itemIndex}, ${page.offset + page.limit})">Next</button></div></section>`;
  target.scrollIntoView({behavior:'smooth', block:'start'});
}
async function openDatasetImageRow(index) {
  const row = state.datasetInspect?.rows?.[index];
  if (!row) return;
  const imageName = row.image_name || row.file_name || '';
  const hint = row.gcs_uri || row.source_path || row.key || row.file_name || '';
  if (!imageName) return setStatus('This row has no image_name or file_name.', true);
  setStatus(`Opening ${imageName}...`);
  try {
    const image = await api(`/api/dataset/image-link?name=${encodeURIComponent(imageName)}&hint=${encodeURIComponent(hint)}`);
    const overlay = document.createElement('div');
    overlay.id = 'datasetImageOverlay';
    overlay.className = 'image-overlay';
    overlay.innerHTML = `<div class="toolbar"><span style="color:white; margin-right:auto;"><strong>${esc(image.image_name)}</strong> <span class="mono">${esc(image.uri)}</span></span><button class="btn" onclick="closeDatasetImage()">Close</button></div><img src="${esc(image.url)}" alt="${esc(image.image_name)}">`;
    document.body.appendChild(overlay);
    setStatus(`Opened ${image.image_name}.`);
  } catch (e) { setStatus(e.message, true); }
}
function closeDatasetImage() { $('#datasetImageOverlay')?.remove(); }
function toggleDataset(index, checked=null) {
  const item = state.datasetItems[index];
  if (!item) return;
  const key = datasetKey(item);
  const next = checked === null ? !state.selectedDatasetKeys.has(key) : checked;
  if (next) state.selectedDatasetKeys.add(key); else state.selectedDatasetKeys.delete(key);
  renderDatasets(state.datasetItems);
  setStatus(`${state.selectedDatasetKeys.size} dataset(s) selected.`);
}
function toggleAllDatasets(checked) {
  state.selectedDatasetKeys = checked ? new Set(state.datasetItems.map(datasetKey).filter(Boolean)) : new Set();
  renderDatasets(state.datasetItems);
  setStatus(`${state.selectedDatasetKeys.size} dataset(s) selected.`);
}
async function refreshDatasetsAfterCombine(message) {
  const data = await api(state.datasetsIncludeCloud ? '/api/datasets?cloud=1' : '/api/datasets');
  state.datasetItems = state.datasetsIncludeCloud ? [...(data.local || []), ...(data.cloud || [])] : (data.local || []);
  state.datasets = state.datasetItems;
  renderDatasets(state.datasetItems);
  setStatus(message);
}
async function combineSelectedDatasets() {
  const items = selectedDatasetItems();
  const outputName = ($('#combinedDatasetName')?.value || '').trim();
  if (!items.length) return setStatus('Select at least one dataset to combine.', true);
  if (!outputName) return setStatus('Enter a name for the combined dataset.', true);
  setStatus('Combining datasets...');
  try {
    const result = await api('/api/datasets/combine', {
      method:'POST',
      body: JSON.stringify({
        datasets: items.map(item => ({
          source: item.source || '',
          name: item.name || '',
          location: item.location || '',
          local_path: item.local_path || '',
          gcs_uri: item.gcs_uri || '',
          run_id: item.run_id || ''
        })),
        output_name: outputName,
        duplicate_strategy: $('#datasetDuplicateStrategy')?.value || 'first_successful'
      }),
      headers:{'Content-Type':'application/json'}
    });
    state.selectedDatasetKeys = new Set();
    const syncText = result.cloud_uri ? ` Shared: ${result.cloud_uri}` : (result.cloud_sync_error ? ` Cloud sync failed: ${result.cloud_sync_error}` : ' Shared sync skipped.');
    await refreshDatasetsAfterCombine(`Created ${result.dataset_name} with ${result.row_count} row(s). Duplicates detected: ${result.duplicates_detected}. ${syncText}`);
  } catch (e) { setStatus(e.message, true); }
}

async function submit() {
  $('#main').innerHTML = `<h1>Submit</h1><div class="sub">Start a local API or cloud batch run from selectable inputs.</div><div id="status" class="status">Loading choices...</div><div id="submitBody"></div>`;
  try {
    const [opts, localInputs] = await Promise.all([api('/api/options'), api('/api/local-inputs').catch(() => [])]);
    state.localInputs = localInputs || [];
    if (!state.selectedLocalPath && state.localInputs.length) state.selectedLocalPath = state.localInputs[0].path;
    $('#submitBody').innerHTML = `<section class="panel">
      <label>Source</label><select id="source" onchange="renderInputChoices()"><option value="local">Local</option><option value="cloud">Cloud</option></select>
      <label>Run mode</label><select id="mode"><option value="local_api">Local API</option><option value="cloud_batch">Cloud batch</option></select>
      <div id="inputChoices"></div>
      <div class="toolbar"><button class="btn secondary" onclick="previewSubmission()">Preview random pages</button></div><div id="submitPreview"></div>
      <label>Schema version</label><select id="schema">${(opts.schemas || []).map(s=>`<option value="${esc(s.version_id)}" data-name="${esc(s.name)}" ${s.is_active ? 'selected' : ''}>${esc(s.name)} v${esc(s.version_number)}${s.is_active ? ' - Active' : ''}</option>`).join('')}</select>
      <label>Model</label><select id="model">${(opts.models || []).map(m=>`<option>${esc(m.name)}</option>`).join('')}</select>
      <details><summary>Advanced</summary><label>Batch chunks</label><input id="chunks" type="number" min="1" placeholder="optional"><label class="inline-control"><input id="subagentUsage" type="checkbox"> Sub Agent Usage</label><div class="small-note">Runs one parallel request per top-level schema field, then joins and validates the page before dataset insertion.</div></details>
      <div class="toolbar"><button class="btn" onclick="submitRun()">Submit</button><button class="btn secondary" onclick="submit()">Refresh choices</button></div>
    </section>`;
    renderInputChoices();
    setStatus('Choices loaded.');
  } catch (e) { setStatus(e.message, true); }
}
function renderInputChoices() {
  const source = $('#source')?.value || 'local';
  const mode = $('#mode');
  if (mode) {
    const localOption = [...mode.options].find(option => option.value === 'local_api');
    if (localOption) localOption.disabled = source === 'cloud';
    if (source === 'cloud') mode.value = 'cloud_batch';
  }
  if (source === 'cloud') {
    if (!state.cloudInputs.length) return loadCloudInputs();
    return renderCloudInputs();
  }
  return renderLocalInputs();
}
function renderLocalInputs() {
  const rows = state.localInputs.map((item, i) => {
    const checked = state.selectedLocalPath === item.path ? 'checked' : '';
    return `<tr class="clickable ${checked ? 'selected' : ''}" onclick="selectLocalInput(${i})">
      <td class="select-cell"><input type="radio" name="localInput" ${checked} onclick="event.stopPropagation(); selectLocalInput(${i})"></td>
      <td>${esc(item.name)}</td><td>${esc(item.image_count)}</td><td>${esc(item.updated_at)}</td><td>${esc(item.path)}</td>
    </tr>`;
  }).join('');
  $('#inputChoices').innerHTML = `<label>Local folder</label>${rawTable(['<th class="select-cell"></th>','<th>Name</th>','<th>Images</th>','<th>Updated</th>','<th>Path</th>'], rows)}
    <details><summary>Advanced</summary><label>Custom local folder</label><input id="customLocal" placeholder="Only use if the folder is not listed"></details>`;
}
function selectLocalInput(index) {
  const item = state.localInputs[index];
  if (item) state.selectedLocalPath = item.path;
  renderLocalInputs();
}
async function loadCloudInputs() {
  setStatus('Loading cloud folders...');
  try {
    state.cloudInputs = await api('/api/cloud-inputs');
    state.selectedCloudPrefixes = new Set([...state.selectedCloudPrefixes].filter(prefix => state.cloudInputs.some(c => c.prefix === prefix)));
    renderCloudInputs();
    setStatus(`Loaded ${state.cloudInputs.length} cloud folder(s).`);
  } catch (e) { setStatus(e.message, true); }
}
function renderCloudInputs() {
  if (!state.cloudInputs.length) {
    $('#inputChoices').innerHTML = `<label>Cloud folders</label><div class="toolbar"><button class="btn secondary" onclick="loadCloudInputs()">Load cloud folders</button></div><div class="muted">Load the bucket choices, then select one or more folders.</div>
      <details><summary>Advanced</summary><label>Custom cloud prefix</label><input id="customCloud" placeholder="Only use if the folder is not listed"></details>`;
    return;
  }
  const allChecked = state.cloudInputs.length > 0 && state.cloudInputs.every(c => state.selectedCloudPrefixes.has(c.prefix));
  const rows = state.cloudInputs.map((item, i) => {
    const checked = state.selectedCloudPrefixes.has(item.prefix) ? 'checked' : '';
    return `<tr class="clickable ${checked ? 'selected' : ''}" onclick="toggleCloudInput(${i})">
      <td class="select-cell"><input type="checkbox" ${checked} onclick="event.stopPropagation(); toggleCloudInput(${i}, this.checked)"></td>
      <td>${esc(item.prefix)}</td><td>${esc(item.image_count)}</td><td>${esc(item.updated_at)}</td>
    </tr>`;
  }).join('');
  $('#inputChoices').innerHTML = `<label>Cloud folders</label>${rawTable([`<th class="select-cell"><input type="checkbox" ${allChecked ? 'checked' : ''} onchange="toggleAllCloudInputs(this.checked)"></th>`, '<th>Prefix</th>', '<th>Images</th>', '<th>Updated</th>'], rows)}
    <details><summary>Advanced</summary><label>Custom cloud prefix</label><input id="customCloud" placeholder="Only use if the folder is not listed"></details>`;
}
function toggleCloudInput(index, checked=null) {
  const item = state.cloudInputs[index];
  if (!item) return;
  const next = checked === null ? !state.selectedCloudPrefixes.has(item.prefix) : checked;
  if (next) state.selectedCloudPrefixes.add(item.prefix); else state.selectedCloudPrefixes.delete(item.prefix);
  renderCloudInputs();
  setStatus(`${state.selectedCloudPrefixes.size} cloud folder(s) selected.`);
}
function toggleAllCloudInputs(checked) {
  state.selectedCloudPrefixes = checked ? new Set(state.cloudInputs.map(c => c.prefix)) : new Set();
  renderCloudInputs();
  setStatus(`${state.selectedCloudPrefixes.size} cloud folder(s) selected.`);
}
function selectedSubmissionInput() {
  const source = $('#source')?.value || 'local';
  let localPath = '';
  let cloudPrefixes = [];
  if (source === 'local') {
    localPath = state.selectedLocalPath || ($('#customLocal')?.value || '').trim();
  } else {
    cloudPrefixes = [...state.selectedCloudPrefixes];
    const custom = ($('#customCloud')?.value || '').trim();
    if (!cloudPrefixes.length && custom) cloudPrefixes = [custom];
  }
  return {source, localPath, cloudPrefixes};
}
async function previewSubmission() {
  const selection = selectedSubmissionInput();
  if (selection.source === 'local' && !selection.localPath) return setStatus('Select a local folder.', true);
  if (selection.source === 'cloud' && !selection.cloudPrefixes.length) return setStatus('Select one or more cloud folders.', true);
  const preview = $('#submitPreview');
  if (preview) preview.innerHTML = '<div class="muted">Drawing a random sample...</div>';
  try {
    const result = await api('/api/submit/preview', {method:'POST', body:JSON.stringify({
      dataset_source: selection.source,
      local_path: selection.localPath,
      cloud_prefixes: selection.cloudPrefixes,
      sample_size: 6
    }), headers:{'Content-Type':'application/json'}});
    if (preview) preview.innerHTML = `<div class="notice"><strong>${esc(result.selection_count)} selected image(s)</strong><div class="muted">Random sample from the exact folders currently selected.</div></div><div class="preview-grid">${(result.samples || []).map(item => `<div class="preview-item"><img src="${esc(item.url)}" alt="${esc(item.image_name)}"><strong>${esc(item.image_name)}</strong><div class="muted mono">${esc(shortText(item.location, 55))}</div></div>`).join('')}</div>`;
    setStatus(`Previewed ${(result.samples || []).length} of ${result.selection_count} selected image(s).`);
  } catch (e) { if (preview) preview.innerHTML = ''; setStatus(e.message, true); }
}
async function submitRun() {
  const selection = selectedSubmissionInput();
  const source = selection.source;
  const mode = $('#mode').value;
  const localPath = selection.localPath;
  const cloudPrefixes = selection.cloudPrefixes;
  if (source === 'local' && !localPath) return setStatus('Select a local folder.', true);
  if (source === 'cloud' && !cloudPrefixes.length) return setStatus('Select one or more cloud folders.', true);
  const schemaSelect = $('#schema');
  const schemaOption = schemaSelect?.selectedOptions?.[0];
  const body = {
    dataset_source: source,
    run_mode: mode,
    local_path: localPath,
    cloud_prefix: cloudPrefixes[0] || '',
    cloud_prefixes: cloudPrefixes,
    schema_name: schemaOption?.dataset?.name || '',
    schema_version_id: schemaSelect?.value || '',
    model_name: $('#model').value,
    output_format: 'jsonl',
    num_batches: $('#chunks')?.value ? Number($('#chunks').value) : null,
    subagents: Boolean($('#subagentUsage')?.checked)
  };
  try {
    const task = await api('/api/submit', { method:'POST', body: JSON.stringify(body), headers:{'Content-Type':'application/json'} });
    setStatus('Started submit task ' + task.task_id);
    activate('tasks');
  } catch (e) { setStatus(e.message, true); }
}

async function cloud() {
  $('#main').innerHTML = `<h1>Cloud</h1><div class="sub">Google Cloud connection, browser login, and access checks.</div><div id="status" class="status">Loading...</div><div id="cloudBody"></div>`;
  try {
    const settings = await api('/api/cloud/settings');
    $('#cloudBody').innerHTML = `<div class="split">
      <section class="panel">
        <h2>Connection</h2>
        <label>Auth mode</label><select id="cloudAuthMode"><option value="adc">Browser login</option><option value="service_account">Service account</option></select>
        <label>GCP project</label><input id="cloudProject" value="${esc(settings.gcp_project_id || '')}">
        <label>GCS bucket</label><input id="cloudBucket" value="${esc(settings.gcs_bucket_name || '')}">
        <label>GCP location</label><input id="cloudLocation" value="${esc(settings.gcp_location || '')}">
        <label>Vertex model location</label><input id="cloudVertexLocation" value="${esc(settings.vertex_model_location || '')}">
        <details><summary>Advanced</summary>
          <label>Service account JSON</label><input id="cloudServiceAccount" value="${esc(settings.service_account_file || '')}">
          <label>Pages prefix</label><input id="cloudPagesPrefix" value="${esc(settings.gcs_pages_prefix || '')}">
          <label>Batch requests prefix</label><input id="cloudRequestsPrefix" value="${esc(settings.batch_requests_gcs_prefix || '')}">
          <label>Batch outputs prefix</label><input id="cloudOutputsPrefix" value="${esc(settings.batch_outputs_gcs_prefix || '')}">
          <label>Datasets prefix</label><input id="cloudDatasetsPrefix" value="${esc(settings.datasets_gcs_prefix || '')}">
          <label>Validations prefix</label><input id="cloudValidationsPrefix" value="${esc(settings.validations_gcs_prefix || '')}">
          <label>Schemas prefix</label><input id="cloudSchemasPrefix" value="${esc(settings.schemas_gcs_prefix || '')}">
          <label><input id="cloudUploadValidations" type="checkbox" ${settings.upload_validation_to_gcs ? 'checked' : ''}> Upload validations to shared bucket</label>
        </details>
        <div class="toolbar">
          <button class="btn" onclick="connectCloud('adc')">Connect browser login</button>
          <button class="btn secondary" onclick="runCloudCheck()">Run access check</button>
          <button class="btn secondary" onclick="saveCloud()">Save</button>
          <button class="btn secondary" onclick="connectCloud('gcloud')">CLI login</button>
        </div>
      </section>
      <section class="panel">
        <h2>Access Checks</h2>
        <div id="cloudResults" class="muted">No check run yet.</div>
      </section>
    </div>`;
    $('#cloudAuthMode').value = settings.auth_mode || 'adc';
    setStatus('Cloud settings loaded.');
  } catch (e) { setStatus(e.message, true); }
}
function cloudPayload() {
  return {
    auth_mode: $('#cloudAuthMode')?.value || 'adc',
    service_account_file: $('#cloudServiceAccount')?.value || '',
    gcp_project_id: $('#cloudProject')?.value || '',
    gcp_location: $('#cloudLocation')?.value || '',
    vertex_model_location: $('#cloudVertexLocation')?.value || '',
    gcs_bucket_name: $('#cloudBucket')?.value || '',
    gcs_pages_prefix: $('#cloudPagesPrefix')?.value || '',
    batch_requests_gcs_prefix: $('#cloudRequestsPrefix')?.value || '',
    batch_outputs_gcs_prefix: $('#cloudOutputsPrefix')?.value || '',
    datasets_gcs_prefix: $('#cloudDatasetsPrefix')?.value || '',
    validations_gcs_prefix: $('#cloudValidationsPrefix')?.value || '',
    schemas_gcs_prefix: $('#cloudSchemasPrefix')?.value || '',
    upload_validation_to_gcs: $('#cloudUploadValidations')?.checked ?? true
  };
}
async function saveCloud() {
  try {
    await api('/api/cloud/settings', { method:'POST', body: JSON.stringify(cloudPayload()), headers:{'Content-Type':'application/json'} });
    setStatus('Cloud settings saved.');
  } catch (e) { setStatus(e.message, true); }
}
async function connectCloud(mode) {
  const payload = cloudPayload();
  if (mode === 'adc') {
    payload.auth_mode = 'adc';
    const auth = $('#cloudAuthMode');
    if (auth) auth.value = 'adc';
  }
  try {
    const result = await api('/api/cloud/login', { method:'POST', body: JSON.stringify({ mode, settings: payload }), headers:{'Content-Type':'application/json'} });
    setStatus(`Started ${result.command}. Complete the browser login, then run access check.`);
  } catch (e) { setStatus(e.message, true); }
}
async function runCloudCheck() {
  setStatus('Running access check...');
  $('#cloudResults').innerHTML = 'Checking...';
  try {
    const report = await api('/api/cloud/check', { method:'POST', body: JSON.stringify(cloudPayload()), headers:{'Content-Type':'application/json'} });
    renderCloudReport(report);
    setStatus(report.ready ? `Access ready. ${report.warnings} warning(s).` : `${report.failed} failed, ${report.warnings} warning(s).`, !report.ready);
  } catch (e) {
    $('#cloudResults').innerHTML = `<div class="bad">${esc(e.message)}</div>`;
    setStatus(e.message, true);
  }
}
function renderCloudReport(report) {
  const rows = (report.results || []).map(r => [
    String(r.status || '').toUpperCase(),
    r.name || '',
    r.detail || '',
    r.fix || ''
  ]);
  $('#cloudResults').innerHTML = table(['Status','Check','Detail','Fix'], rows);
}

async function tasks() {
  $('#main').innerHTML = `<h1>Tasks</h1><div class="sub">Background submit, retrieve, recovery, and finalization work.</div><div id="status" class="status">Loading...</div><div class="toolbar"><button class="btn" onclick="tasks()">Refresh</button></div><div id="tasksBody"></div>`;
  try {
    const rows = await api('/api/tasks');
    $('#tasksBody').innerHTML = table(['Task','Kind','Status','Updated','Error'], rows.map(t => [t.task_id, t.kind, t.status, t.updated_at, t.error ? t.error.slice(0,160) : '']));
    setStatus(`${rows.length} task(s).`);
  } catch (e) { setStatus(e.message, true); }
}
dashboard();
</script>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    service: WorkflowService
    runner: TaskRunner

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(serializable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = APP_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, *, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            return {}
        raw = self.rfile.read(size).decode("utf-8")
        payload = json.loads(raw or "{}")
        return payload if isinstance(payload, dict) else {}

    def _task(self, kind: str, func, metadata: dict | None = None) -> None:
        task_id = self.runner.submit(kind, func, metadata=metadata or {})
        self._send_json({"task_id": task_id, "status": "pending"})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_html()
            elif parsed.path == "/api/options":
                schema_data = self.service.list_schemas()
                self._send_json(
                    {
                        "schemas": [
                            {
                                "name": item.get("name", ""),
                                "version_id": item.get("version_id", ""),
                                "version_number": item.get("version_number", 1),
                                "field_count": item.get("field_count", 0),
                                "is_active": item.get("is_active", False),
                            }
                            for item in schema_data.get("versions", [])
                        ],
                        "models": serializable(list_google_model_options()),
                    }
                )
            elif parsed.path == "/api/schemas":
                self._send_json(self.service.list_schemas())
            elif parsed.path == "/api/jobs":
                self._send_json(self.service.list_jobs())
            elif parsed.path == "/api/tasks":
                self._send_json(self.runner.list_tasks())
            elif parsed.path == "/api/datasets":
                self._send_json(
                    self.service.list_datasets(
                        include_cloud=query.get("cloud", ["0"])[0] == "1"
                    )
                )
            elif parsed.path == "/api/cloud/settings":
                self._send_json(self.service.cloud_settings())
            elif parsed.path == "/api/local-inputs":
                self._send_json(self.service.local_input_choices())
            elif parsed.path == "/api/cloud-inputs":
                self._send_json(self.service.cloud_input_choices())
            elif parsed.path == "/api/dashboard":
                self._send_json(self.service.dashboard())
            elif parsed.path == "/api/dataset/analyze":
                path = query.get("path", [""])[0]
                if not path:
                    raise ValueError("Missing dataset path.")
                self._send_json(self.service.analyze_dataset(path))
            elif parsed.path == "/api/dataset/inspect":
                location = query.get("path", [""])[0]
                if not location:
                    raise ValueError("Missing dataset path.")
                self._send_json(
                    self.service.inspect_dataset(
                        location,
                        offset=int(query.get("offset", ["0"])[0] or 0),
                        limit=int(query.get("limit", ["50"])[0] or 50),
                    )
                )
            elif parsed.path == "/api/dataset/image-link":
                self._send_json(
                    self.service.dataset_image_link(
                        image_name=query.get("name", [""])[0],
                        object_hint=query.get("hint", [""])[0],
                    )
                )
            elif parsed.path == "/api/images/local":
                body, content_type = self.service.local_image_bytes(
                    query.get("token", [""])[0]
                )
                self._send_bytes(body, content_type=content_type)
            elif parsed.path == "/api/validation/identity":
                self._send_json(self.service.validation_identity())
            elif parsed.path == "/api/validation/session":
                session_id = query.get("session_id", [""])[0]
                if not session_id:
                    raise ValueError("Missing validation session_id.")
                self._send_json(self.service.browser_validation_current(session_id))
            elif parsed.path == "/api/validation/session/image":
                session_id = query.get("session_id", [""])[0]
                if not session_id:
                    raise ValueError("Missing validation session_id.")
                body, content_type = self.service.browser_validation_image(session_id)
                self._send_bytes(body, content_type=content_type)
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/submit":
                raw_prefixes = payload.get("cloud_prefixes") or ()
                if isinstance(raw_prefixes, str):
                    raw_prefixes = (raw_prefixes,)
                num_batches = payload.get("num_batches")
                draft = SubmitJobDraft(
                    dataset_source=payload.get("dataset_source", "local"),
                    run_mode=payload.get("run_mode", "local_api"),
                    schema_name=str(payload.get("schema_name") or ""),
                    model_name=str(payload.get("model_name") or ""),
                    schema_version_id=str(payload.get("schema_version_id") or ""),
                    output_format=str(payload.get("output_format") or "jsonl"),
                    local_path=str(payload.get("local_path") or ""),
                    cloud_prefix=str(payload.get("cloud_prefix") or ""),
                    cloud_prefixes=tuple(str(item) for item in raw_prefixes if item),
                    continue_dataset=str(payload.get("continue_dataset") or ""),
                    num_batches=int(num_batches) if num_batches not in {None, ""} else None,
                    subagents=bool(payload.get("subagents", False)),
                )
                self._task("submit", lambda: self.service.submit_batch(draft), payload)
            elif parsed.path == "/api/cloud/settings":
                self._send_json(self.service.save_cloud_settings(payload))
            elif parsed.path == "/api/cloud/check":
                self._send_json(self.service.cloud_access_report(payload))
            elif parsed.path == "/api/cloud/login":
                settings_payload = payload.get("settings")
                self._send_json(
                    self.service.start_cloud_browser_login(
                        mode=str(payload.get("mode") or "adc"),
                        payload=settings_payload if isinstance(settings_payload, dict) else {},
                    )
                )
            elif parsed.path == "/api/schemas/version":
                self._send_json(self.service.create_schema_version(payload))
            elif parsed.path == "/api/schemas/active":
                self._send_json(
                    self.service.set_active_schema(
                        str(payload.get("version_id") or "")
                    )
                )
            elif parsed.path == "/api/submit/preview":
                self._send_json(self.service.preview_submission(payload))
            elif parsed.path == "/api/jobs/retrieve":
                run_dir = str(payload.get("run_dir") or "")
                self._task(
                    "retrieve",
                    lambda: self.service.retrieve_results(
                        run_dir,
                        ignore_failed=bool(payload.get("ignore_failed")),
                        duplicate_strategy=str(payload.get("duplicate_strategy") or ""),
                        force=bool(payload.get("force")),
                    ),
                    payload,
                )
            elif parsed.path == "/api/jobs/retrieve-many":
                run_dirs = payload.get("run_dirs") or ()
                if isinstance(run_dirs, str):
                    run_dirs = (run_dirs,)
                self._task(
                    "retrieve_many",
                    lambda: self.service.retrieve_many(
                        [str(item) for item in run_dirs if item],
                        ignore_failed=bool(payload.get("ignore_failed")),
                        duplicate_strategy=str(payload.get("duplicate_strategy") or ""),
                        force=bool(payload.get("force")),
                    ),
                    payload,
                )
            elif parsed.path == "/api/jobs/finalize-failed":
                run_dir = str(payload.get("run_dir") or "")
                self._task("finalize_failed", lambda: self.service.finalize_failed_rows(run_dir), payload)
            elif parsed.path == "/api/jobs/recover-api":
                run_dir = str(payload.get("run_dir") or "")
                self._task("recover_api", lambda: self.service.recover_missing_with_api(run_dir), payload)
            elif parsed.path == "/api/jobs/resubmit-failed":
                run_dir = str(payload.get("run_dir") or "")
                count = int(payload.get("num_batches") or 1)
                self._task("resubmit_failed", lambda: self.service.resubmit_failed(run_dir, num_batches=count), payload)
            elif parsed.path == "/api/datasets/combine":
                raw_items = payload.get("datasets") or ()
                if not isinstance(raw_items, list):
                    raise ValueError("datasets must be a list.")
                self._send_json(
                    self.service.combine_datasets(
                        [item for item in raw_items if isinstance(item, dict)],
                        output_name=str(payload.get("output_name") or ""),
                        duplicate_strategy=str(
                            payload.get("duplicate_strategy") or "first_successful"
                        ),
                    )
                )
            elif parsed.path == "/api/validation/session/start":
                raw_prefixes = payload.get("cloud_prefixes") or ()
                if isinstance(raw_prefixes, str):
                    raw_prefixes = (raw_prefixes,)
                self._send_json(
                    self.service.start_browser_validation(
                        results=str(payload.get("results") or ""),
                        image_source=str(payload.get("image_source") or "cloud"),
                        images=str(payload.get("images") or ""),
                        cloud_prefixes=tuple(str(item) for item in raw_prefixes if item),
                        corrections=bool(payload.get("corrections", True)),
                        sampling_mode=str(payload.get("sampling_mode") or "balanced_ucb"),
                        offline=bool(payload.get("offline")),
                    )
                )
            elif parsed.path == "/api/validation/session/mark":
                self._send_json(
                    self.service.mark_browser_validation(
                        session_id=str(payload.get("session_id") or ""),
                        label=str(payload.get("label") or ""),
                        corrected_text=str(payload.get("corrected_field") or ""),
                    )
                )
            elif parsed.path == "/api/validation/session/finish":
                self._send_json(
                    self.service.finish_browser_validation(
                        str(payload.get("session_id") or "")
                    )
                )
            elif parsed.path == "/api/validation/start":
                self._task(
                    "validation",
                    lambda: self.service.start_validation(
                        results=str(payload.get("results") or ""),
                        images=str(payload.get("images") or ""),
                        username=str(payload.get("username") or "researcher"),
                        corrections=bool(payload.get("corrections", True)),
                        sampling_mode=str(payload.get("sampling_mode") or "balanced_ucb"),
                    ),
                    payload,
                )
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def run_server(*, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    settings = load_app_settings()
    service = WorkflowService(settings)
    runner = TaskRunner(service.store)
    AppHandler.service = service
    AppHandler.runner = runner
    server = ThreadingHTTPServer((host, port), AppHandler)
    url = f"http://{host}:{port}"
    print(f"PatientJournals web app running at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        server.serve_forever()
    finally:
        runner.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PatientJournals web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab.")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
