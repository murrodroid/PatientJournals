from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from patientjournals.app.catalog import list_google_model_options, list_schema_options
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
    .status { color:var(--muted); margin:10px 0; min-height:22px; white-space:pre-wrap; }
    .bad { color:#9A3412; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
    .validator { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; min-height:calc(100vh - 96px); }
    .image-stage { border:1px solid var(--line); background:var(--soft); overflow:auto; display:flex; align-items:flex-start; justify-content:center; padding:12px; min-height:520px; }
    .image-stage img { max-width:none; transform-origin:top center; background:white; box-shadow:0 1px 6px rgba(30,30,36,.12); }
    .decision-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .decision-grid .btn { width:100%; }
    .btn.accept { background:#16A34A; color:white; }
    .btn.partial { background:#84CC16; color:var(--ink); }
    .btn.reject { background:#DC2626; color:white; }
    .btn.unsure { background:#F59E0B; color:var(--ink); }
    @media (max-width: 900px) { .app { grid-template-columns:1fr; } aside { position:static; } .grid,.split,.validator { grid-template-columns:1fr; } }
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
  validationSession: null,
  validationSample: null,
  validationZoom: 1
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
  ({dashboard, validate, jobs, datasets, submit, cloud, tasks}[tab])();
}
document.querySelectorAll('nav button').forEach(b => b.onclick = () => activate(b.dataset.tab));
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

async function dashboard() {
  $('#main').innerHTML = `<h1>Dashboard</h1><div class="sub">Research metrics, validation outcomes, and dataset inspection.</div><div id="status" class="status">Loading...</div><div id="dashboardBody"></div>`;
  try {
    const [summary, datasets, localInputs] = await Promise.all([api('/api/dashboard'), api('/api/datasets'), api('/api/local-inputs').catch(() => [])]);
    state.datasets = datasets.local || [];
    state.localInputs = localInputs || [];
    const options = state.datasets.map(d => `<option value="${esc(d.local_path || d.location)}">${esc(d.run_id || d.name)} - ${esc(d.name)} (${esc(d.row_count ?? '?')} rows)</option>`).join('');
    const imageOptions = state.localInputs.map(d => `<option value="${esc(d.path)}">${esc(d.name)} (${esc(d.image_count)} images)</option>`).join('');
    $('#dashboardBody').innerHTML = `
      <div class="grid">
        ${metric('Datasets', summary.dataset_count)}
        ${metric('Dataset rows', summary.dataset_rows)}
        ${metric('Validation decisions', summary.validation_count)}
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
          <h2>Validation Runs</h2>
          ${table(['Run','Validator','Dataset','Accuracy','Decisions'], (summary.validation_runs || []).slice(0,12).map(r => [r.run_id, r.validator_id, r.dataset_file, r.accuracy == null ? '-' : r.accuracy.toFixed(1)+'%', r.decisions]))}
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

async function validate() {
  $('#main').innerHTML = `<h1>Validate</h1><div class="sub">Browser validation with signed cloud image links and autosaved decisions.</div><div id="status" class="status">Loading...</div><div id="validateBody"></div>`;
  try {
    const [datasets, localInputs] = await Promise.all([api('/api/datasets'), api('/api/local-inputs').catch(() => [])]);
    state.datasets = datasets.local || [];
    state.localInputs = localInputs || [];
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
    return `<option value="${esc(value)}" ${selected}>${esc(d.source)} - ${esc(d.run_id || d.name)} (${esc(d.row_count ?? '?')} rows)</option>`;
  }).join('');
}
function renderValidationSetup() {
  $('#validateBody').innerHTML = `<section class="panel">
    <h2>Start Browser Validation</h2>
    <label>Dataset</label><select id="browserValidationDataset" onchange="state.selectedDataset=this.value">${validationDatasetOptions()}</select>
    <div class="toolbar"><button class="btn secondary" onclick="loadValidationSharedDatasets()">Load shared datasets</button><button class="btn secondary" onclick="validate()">Refresh local</button></div>
    <label>Validator</label><input id="browserValidatorName" value="researcher">
    <label>Sampling mode</label><select id="browserSamplingMode"><option value="balanced_ucb">Balanced UCB</option><option value="random">True random</option></select>
    <label>Image source</label><select id="browserImageSource" onchange="renderValidationImageSource()"><option value="cloud">Cloud images, signed URLs</option><option value="local">Local folder</option></select>
    <div id="validationImageSourceBody"></div>
    <details><summary>Advanced</summary><label><input id="browserCorrections" type="checkbox" checked> Enable correction entry</label></details>
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
    $('#validationImageSourceBody').innerHTML = `<label>Local image folder</label>${rawTable(['<th class="select-cell"></th>','<th>Name</th>','<th>Images</th>','<th>Updated</th>','<th>Path</th>'], rows)}
      <details><summary>Advanced</summary><label>Custom local folder</label><input id="browserCustomLocalImages" placeholder="Only use if the folder is not listed"></details>`;
    return;
  }
  renderValidationCloudInputs();
}
function selectValidationLocalInput(index) {
  const item = state.localInputs[index];
  if (item) state.selectedLocalPath = item.path;
  renderValidationImageSource();
}
async function loadValidationCloudInputs() {
  setStatus('Loading cloud image folders...');
  try {
    state.cloudInputs = await api('/api/cloud-inputs');
    state.selectedValidationCloudPrefixes = new Set([...state.selectedValidationCloudPrefixes].filter(prefix => state.cloudInputs.some(c => c.prefix === prefix)));
    renderValidationCloudInputs();
    setStatus(`Loaded ${state.cloudInputs.length} cloud folder(s).`);
  } catch (e) { setStatus(e.message, true); }
}
function renderValidationCloudInputs() {
  const body = $('#validationImageSourceBody');
  if (!body) return;
  if (!state.cloudInputs.length) {
    body.innerHTML = `<label>Cloud image folders</label><div class="toolbar"><button class="btn secondary" onclick="loadValidationCloudInputs()">Load cloud folders</button></div><div class="muted">Cloud images are displayed with short-lived signed URLs. Links are not stored in validation outputs.</div>
      <details><summary>Advanced</summary><label>Custom cloud prefix</label><input id="browserCustomCloudPrefix" placeholder="Only use if the folder is not listed"></details>`;
    return;
  }
  const allChecked = state.cloudInputs.length > 0 && state.cloudInputs.every(c => state.selectedValidationCloudPrefixes.has(c.prefix));
  const rows = state.cloudInputs.map((item, i) => {
    const checked = state.selectedValidationCloudPrefixes.has(item.prefix) ? 'checked' : '';
    return `<tr class="clickable ${checked ? 'selected' : ''}" onclick="toggleValidationCloudInput(${i})">
      <td class="select-cell"><input type="checkbox" ${checked} onclick="event.stopPropagation(); toggleValidationCloudInput(${i}, this.checked)"></td>
      <td>${esc(item.prefix)}</td><td>${esc(item.image_count)}</td><td>${esc(item.updated_at)}</td>
    </tr>`;
  }).join('');
  body.innerHTML = `<label>Cloud image folders</label>${rawTable([`<th class="select-cell"><input type="checkbox" ${allChecked ? 'checked' : ''} onchange="toggleAllValidationCloudInputs(this.checked)"></th>`, '<th>Prefix</th>', '<th>Images</th>', '<th>Updated</th>'], rows)}
    <details><summary>Advanced</summary><label>Custom cloud prefix</label><input id="browserCustomCloudPrefix" placeholder="Only use if the folder is not listed"></details>`;
}
function toggleValidationCloudInput(index, checked=null) {
  const item = state.cloudInputs[index];
  if (!item) return;
  const next = checked === null ? !state.selectedValidationCloudPrefixes.has(item.prefix) : checked;
  if (next) state.selectedValidationCloudPrefixes.add(item.prefix); else state.selectedValidationCloudPrefixes.delete(item.prefix);
  renderValidationCloudInputs();
  setStatus(`${state.selectedValidationCloudPrefixes.size} cloud folder(s) selected.`);
}
function toggleAllValidationCloudInputs(checked) {
  state.selectedValidationCloudPrefixes = checked ? new Set(state.cloudInputs.map(c => c.prefix)) : new Set();
  renderValidationCloudInputs();
  setStatus(`${state.selectedValidationCloudPrefixes.size} cloud folder(s) selected.`);
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
    cloudPrefixes = [...state.selectedValidationCloudPrefixes];
    const custom = ($('#browserCustomCloudPrefix')?.value || '').trim();
    if (!cloudPrefixes.length && custom) cloudPrefixes = [custom];
    if (!cloudPrefixes.length) return setStatus('Load and select one or more cloud image folders.', true);
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
        username: $('#browserValidatorName')?.value || 'researcher',
        corrections: $('#browserCorrections')?.checked ?? true,
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
  if (sample.status === 'complete') {
    $('#main').innerHTML = `<h1>Validate</h1><div class="sub">Validation complete.</div><div id="status" class="status"></div>
      <section class="panel">
        <div class="grid">${metric('Decisions', sample.decisions)}${metric('Total pairs', sample.total_pairs)}${metric('Remaining', sample.remaining_pairs)}</div>
        <div class="toolbar"><button class="btn" onclick="finishBrowserValidation()">Save and sync</button><button class="btn secondary" onclick="validate()">New validation</button></div>
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
        <div class="muted mono">${esc(sample.image_source)} - ${esc(sample.image_uri)}</div>
        <div class="grid" style="grid-template-columns:1fr 1fr; margin:12px 0;">
          ${metric('Decisions', sample.decisions)}
          ${metric('Remaining', sample.remaining_pairs)}
        </div>
        <label>Field</label><input value="${esc(sample.field_name)}" readonly>
        <label>Extracted value</label><textarea readonly>${esc(sample.field_value)}</textarea>
        <label>Correction</label><textarea id="validationCorrection" ${sample.allow_corrections ? '' : 'readonly'}>${esc(sample.correction_value)}</textarea>
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
    const upload = result.uploaded?.validation_csv_uri ? ` Synced: ${result.uploaded.validation_csv_uri}` : '';
    setStatus(`Saved ${result.decisions} validation decision(s).${upload}`);
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
      <td>${esc(j.created_at)}</td><td>${esc(j.model)}</td><td>${esc(j.input_location)}</td>
      <td>${esc(j.image_count)}</td><td>${esc(j.status)}</td><td>${esc(j.succeeded ?? '')}</td><td>${esc(j.failed ?? '')}</td>
    </tr>`;
  }).join('');
  $('#jobsBody').innerHTML = rawTable(
    [`<th class="select-cell"><input type="checkbox" ${allChecked ? 'checked' : ''} onchange="toggleAllJobs(this.checked)"></th>`, '<th>Created</th>', '<th>Model</th>', '<th>Input</th>', '<th>Images</th>', '<th>Status</th>', '<th>Success</th>', '<th>Missing</th>'],
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
    <div class="toolbar"><button class="btn" onclick="datasets()">Refresh Local</button><button class="btn secondary" onclick="loadCloudDatasets()">Load Shared</button></div>
    <label>New dataset name</label><input id="combinedDatasetName" placeholder="for example: journals_0618_complete">
    <label>Duplicates</label><select id="datasetDuplicateStrategy"><option value="first_successful">One row per image, prefer first successful</option><option value="provide_all">Include all rows</option></select>
    <div class="toolbar"><button class="btn" onclick="combineSelectedDatasets()">Combine selected</button></div>
  </section>
  <div id="datasetsBody"></div>`;
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
    return `<tr class="clickable ${checked ? 'selected' : ''}" onclick="toggleDataset(${i})">
      <td class="select-cell"><input type="checkbox" ${checked} onclick="event.stopPropagation(); toggleDataset(${i}, this.checked)"></td>
      <td>${esc(d.source)}</td><td>${esc(d.name)}</td><td>${esc(d.row_count ?? '')}</td><td>${esc(d.updated_at)}</td><td>${esc(d.run_id)}</td><td class="mono">${esc(d.location)}</td>
    </tr>`;
  }).join('');
  $('#datasetsBody').innerHTML = rawTable(
    [`<th class="select-cell"><input type="checkbox" ${allChecked ? 'checked' : ''} onchange="toggleAllDatasets(this.checked)"></th>`, '<th>Source</th>', '<th>Name</th>', '<th>Rows</th>', '<th>Updated</th>', '<th>Run</th>', '<th>Location</th>'],
    rows
  );
}
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
      <label>Schema</label><select id="schema">${(opts.schemas || []).map(s=>`<option>${esc(s.name)}</option>`).join('')}</select>
      <label>Model</label><select id="model">${(opts.models || []).map(m=>`<option>${esc(m.name)}</option>`).join('')}</select>
      <details><summary>Advanced</summary><label>Batch chunks</label><input id="chunks" type="number" min="1" placeholder="optional"></details>
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
  if (source === 'cloud') return renderCloudInputs();
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
async function submitRun() {
  const source = $('#source').value;
  const mode = $('#mode').value;
  let localPath = '';
  let cloudPrefixes = [];
  if (source === 'local') {
    localPath = state.selectedLocalPath || ($('#customLocal')?.value || '').trim();
    if (!localPath) return setStatus('Select a local folder.', true);
  } else {
    cloudPrefixes = [...state.selectedCloudPrefixes];
    const custom = ($('#customCloud')?.value || '').trim();
    if (!cloudPrefixes.length && custom) cloudPrefixes = [custom];
    if (!cloudPrefixes.length) return setStatus('Select one or more cloud folders.', true);
  }
  const body = {
    dataset_source: source,
    run_mode: mode,
    local_path: localPath,
    cloud_prefix: cloudPrefixes[0] || '',
    cloud_prefixes: cloudPrefixes,
    schema_name: $('#schema').value,
    model_name: $('#model').value,
    output_format: 'jsonl',
    num_batches: $('#chunks')?.value ? Number($('#chunks').value) : null
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
                self._send_json(
                    {
                        "schemas": serializable(list_schema_options()),
                        "models": serializable(list_google_model_options()),
                    }
                )
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
                    output_format=str(payload.get("output_format") or "jsonl"),
                    local_path=str(payload.get("local_path") or ""),
                    cloud_prefix=str(payload.get("cloud_prefix") or ""),
                    cloud_prefixes=tuple(str(item) for item in raw_prefixes if item),
                    continue_dataset=str(payload.get("continue_dataset") or ""),
                    num_batches=int(num_batches) if num_batches not in {None, ""} else None,
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
                        username=str(payload.get("username") or "researcher"),
                        corrections=bool(payload.get("corrections", True)),
                        sampling_mode=str(payload.get("sampling_mode") or "balanced_ucb"),
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
