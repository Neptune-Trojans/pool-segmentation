// Add videos page (docs/labeling.md): pick frames from new videos for the training data, and let a trained model
// label their water. The server does the work in the background (POST api/add), one job at a time; this page shows
// its progress and links to the new frames on the Labels page.

import { $, el, hideError, requestJSON, showError } from "./dom.js";

const STORAGE = {
  folder: "pool-segmentation.add.folder",
  every: "pool-segmentation.add.every",
  max: "pool-segmentation.add.max",
};

const state = {
  options: null, // GET api/add: start folder, trained models, defaults, the job
  listing: null, // GET api/add/folder: folder, parent, subfolders, videos with their frames in the dataset
  selected: new Set(), // paths of the ticked videos
  rows: new Map(), // video path -> its row
  job: null, // the job running or last run
  polling: null,
  request: 0, // folder listings asked for: only the newest one is shown
};

// Remembered in this browser only (last folder, sample rate); the page works without it (private window)
const store = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* storage blocked */ }
  },
};

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const running = () => state.job?.state === "running";

// ---------------------------------------------------------------------------------------------------------------
// Start

async function main() {
  const options = await requestJSON("api/add");
  state.options = options;
  if (options.home) {
    // Shown inside another app (e.g. swim-analytics): a link back to it
    Object.assign($("home"), { textContent: options.home.text, href: options.home.url, hidden: false });
  }
  $("dataset-dir").textContent = options.dataset;
  $("every").value = store.get(STORAGE.every) ?? options.defaults.every_seconds;
  $("max-per-video").value = store.get(STORAGE.max) ?? options.defaults.max_per_video;
  buildModels(options);
  setupControls();
  window.addEventListener("popstate", () => attempt(loadFolder(new URLSearchParams(location.search).get("folder"))));

  const fromUrl = new URLSearchParams(location.search).get("folder");
  const remembered = store.get(STORAGE.folder);
  try {
    await loadFolder(fromUrl ?? remembered, "replace");
  } catch (error) {
    if (!fromUrl && !remembered) throw error;
    // The remembered folder was moved or deleted: open the default one, keep the message
    await loadFolder(null, "replace");
    showError(error.message);
  }
  showJob(options.job);
  if (running()) startPolling();
}

// Errors of actions the user started (a folder that doesn't exist, a job the server refused) go to the banner
function attempt(promise) {
  promise.then(hideError, (error) => showError(error.message));
}

// The sample rate typed in, or null if it isn't a valid one
function sampling() {
  const every = $("every");
  const max = $("max-per-video");
  if (!every.checkValidity() || !max.checkValidity()) return null;
  return { every_seconds: Number(every.value), max_per_video: Number(max.value) };
}

// ---------------------------------------------------------------------------------------------------------------
// Folder: path field, parent folder, subfolders, videos

// Load a folder (null = the server's videos folder). `history`: "push" (the user moved to another folder: Back
// returns), "replace" (first load), "none" (Back / Forward, a new sample rate, a job done)
async function loadFolder(folder, history = "none") {
  const { every_seconds, max_per_video } = sampling() ?? state.options.defaults;
  const query = new URLSearchParams({ every_seconds, max_per_video });
  if (folder) query.set("path", folder);
  const request = ++state.request;
  const listing = await requestJSON(`api/add/folder?${query}`);
  if (request !== state.request) return; // a newer listing is on its way
  const moved = listing.folder !== state.listing?.folder;
  state.listing = listing;
  store.set(STORAGE.folder, listing.folder);
  const url = `?folder=${encodeURIComponent(listing.folder)}`;
  if (history === "push" && moved) window.history.pushState(null, "", url);
  if (history === "replace") window.history.replaceState(null, "", url);
  if (moved) state.selected.clear();
  renderFolder();
  renderVideos();
}

function renderFolder() {
  const { folder, parent, folders } = state.listing;
  $("folder-path").value = folder;
  $("folder-up").disabled = !parent;
  const chips = folders.map((f) => {
    const button = el("button", "folder-chip", f.name);
    button.type = "button";
    button.addEventListener("click", () => attempt(loadFolder(f.path, "push")));
    return button;
  });
  $("subfolders").replaceChildren(...chips);
  $("subfolders").hidden = !chips.length;
}

function renderVideos() {
  state.rows.clear();
  const rows = state.listing.videos.map(videoRow);
  const empty = el("li", "videos-empty", "No videos in this folder (.mov, .mp4, .m4v, .avi, .mkv).");
  $("videos").replaceChildren(...(rows.length ? rows : [empty]));
  updateToolbar();
}

// One row per video, all of it a label for its checkbox: a click anywhere ticks it
function videoRow(v) {
  const li = el("li", "video-row");
  const label = el("label", "video-label");
  const checkbox = el("input");
  checkbox.type = "checkbox";
  checkbox.checked = state.selected.has(v.path);
  checkbox.disabled = !v.info;
  checkbox.setAttribute("aria-label", `Select ${v.file}`);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) state.selected.add(v.path);
    else state.selected.delete(v.path);
    updateToolbar();
  });

  const thumbBox = el("div", "thumb-box");
  if (v.info) {
    const img = el("img", "thumb");
    img.loading = "lazy";
    img.alt = "";
    img.src = `api/add/thumbnail?path=${encodeURIComponent(v.path)}&v=${v.modified}`;
    img.addEventListener("error", () => img.remove());
    thumbBox.append(img);
  }

  const title = el("div", "video-title");
  title.append(el("span", "video-file", v.file));
  title.append(v.in_dataset
    ? el("span", "pill", `${plural(v.in_dataset, "frame")} in the dataset`)
    : el("span", "pill muted", "not in the dataset"));
  const main = el("div", "video-main");
  main.append(title, el("div", "video-meta", videoMeta(v)));

  let picks;
  if (!v.info) picks = el("span", "new-frames none", "can't read this video");
  else if (v.new_frames) picks = el("span", "new-frames", `+${plural(v.new_frames, "new frame")}`);
  else picks = el("span", "new-frames none", "nothing new");

  label.append(checkbox, thumbBox, main, picks);
  li.append(label);
  li.classList.toggle("selected", checkbox.checked);
  li.classList.toggle("unreadable", !v.info);
  state.rows.set(v.path, li);
  return li;
}

function videoMeta(v) {
  const parts = [];
  if (v.info) {
    parts.push(duration(v.info.frame_count / v.info.fps), `${v.info.frame_count} frames`,
      `${v.info.width}×${v.info.height}`, `${Math.round(v.info.fps)} fps`);
  }
  parts.push(v.size >= 1e9 ? `${(v.size / 1e9).toFixed(1)} GB` : `${Math.round(v.size / 1e6)} MB`);
  return parts.join(" · ");
}

function duration(seconds) {
  if (seconds < 60) return `${Math.max(Math.round(seconds), 1)} s`;
  return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")} min`;
}

// ---------------------------------------------------------------------------------------------------------------
// Settings: sample rate (remembered in this browser), the model that labels the new frames

// Training runs are named by the date and time they started: 20261116-224516 -> 2026-11-16 22:45
function runDate(run) {
  const m = run.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : run;
}

function buildModels(options) {
  const option = (value, text) => Object.assign(el("option", "", text), { value });
  const items = options.models.map((run, k) => option(run, k ? `Model of ${runDate(run)}` : `Newest model (${runDate(run)})`));
  items.push(option("", "None: draw them by hand"));
  $("model").replaceChildren(...items);
  $("model").title = `Training runs in ${options.models_dir}`;
  if (!options.models.length) {
    $("model-note").textContent = `No trained model in ${options.models_dir} yet: the frames are added without labels, `
      + "to draw on the Labels page.";
    $("model-note").hidden = false;
  }
}

let reloadTimer = null;

function onSamplingChange() {
  updateToolbar();
  const picked = sampling();
  if (!picked) return;
  store.set(STORAGE.every, picked.every_seconds);
  store.set(STORAGE.max, picked.max_per_video);
  // The new frames of each video change with the sample rate: ask the server again once typing stops
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(() => attempt(loadFolder(state.listing?.folder)), 300);
}

// ---------------------------------------------------------------------------------------------------------------
// Toolbar: select all, what will be added, Add

const selectable = () => state.listing?.videos.filter((v) => v.info) ?? [];
const selectedVideos = () => selectable().filter((v) => state.selected.has(v.path));

function setupControls() {
  $("folder-form").addEventListener("submit", (event) => {
    event.preventDefault();
    attempt(loadFolder($("folder-path").value.trim(), "push"));
  });
  $("folder-up").addEventListener("click", () => attempt(loadFolder(state.listing.parent, "push")));
  $("every").addEventListener("input", onSamplingChange);
  $("max-per-video").addEventListener("input", onSamplingChange);
  $("select-all").addEventListener("change", (event) => {
    for (const v of selectable()) {
      if (event.target.checked) state.selected.add(v.path);
      else state.selected.delete(v.path);
      state.rows.get(v.path).querySelector("input").checked = event.target.checked;
    }
    updateToolbar();
  });
  $("add").addEventListener("click", () => attempt(addVideos()));
  $("cancel").addEventListener("click", () => attempt(cancelJob()));
}

function updateToolbar() {
  const all = selectable();
  const chosen = selectedVideos();
  const count = state.listing?.videos.length ?? 0;
  Object.assign($("select-all"), {
    checked: all.length > 0 && chosen.length === all.length,
    indeterminate: chosen.length > 0 && chosen.length < all.length,
    disabled: !all.length,
  });
  $("video-count").textContent = chosen.length ? `${chosen.length} of ${plural(count, "video")} selected` : plural(count, "video");
  for (const [path, li] of state.rows) li.classList.toggle("selected", state.selected.has(path));

  const picked = sampling();
  const frames = chosen.reduce((sum, v) => sum + v.new_frames, 0);
  let summary = "Select the videos to add";
  if (!picked) summary = "Type a sample rate: seconds above 0, frames from 1";
  else if (chosen.length && frames) summary = `${plural(frames, "new frame")} from ${plural(chosen.length, "video")}`;
  else if (chosen.length) summary = "Nothing new: their frames are in the dataset already (a shorter step picks more)";
  $("add-summary").textContent = summary;
  $("add").disabled = !picked || !frames || running();
}

async function addVideos() {
  const picked = sampling();
  const videos = selectedVideos().map((v) => v.path);
  if (!picked || !videos.length) return;
  const { job } = await requestJSON("api/add", { videos, ...picked, model: $("model").value || null });
  state.selected.clear();
  for (const li of state.rows.values()) li.querySelector("input").checked = false;
  showJob(job);
  startPolling();
}

async function cancelJob() {
  const { job } = await requestJSON("api/add/job/cancel", {});
  showJob(job);
}

// ---------------------------------------------------------------------------------------------------------------
// The job: progress while it runs, what it added when done

function showJob(job) {
  state.job = job;
  $("job").hidden = !job;
  updateToolbar();
  if (!job) return;
  const active = job.state === "running";
  $("job").className = `job ${job.state}`;
  $("job-text").textContent = jobText(job);
  Object.assign($("cancel"), { hidden: !active, disabled: job.cancel_requested });
  const bar = $("job-progress");
  bar.hidden = !active;
  if (active && job.step !== "loading" && job.total) bar.value = job.done / job.total;
  else bar.removeAttribute("value"); // moving to and fro: no progress to show (loading the model)
  // The new frames are not checked yet: the Labels page opens the first, showing only frames not checked
  $("check").hidden = active || !job.first_added;
  if (job.first_added) $("check").href = `./?filter=todo&frame=${encodeURIComponent(job.first_added)}`;
}

function jobText(job) {
  const left = job.seconds_left != null ? ` · about ${duration(job.seconds_left)} left` : "";
  if (job.state === "running") {
    if (job.cancel_requested) return "Cancelling…";
    if (job.step === "picking") return `Picking frames from ${job.video} (video ${job.videos_done + 1} of ${job.videos_total})${left}`;
    if (job.step === "loading") return `Loading the model ${runDate(job.model)}…`;
    return `The model labels the water: ${job.done} of ${plural(job.total, "frame")}${left}`;
  }
  let added = job.added ? `${plural(job.added, "frame")} added` : "no new frames";
  if (job.added && !job.model) added += ", without labels: draw them on the Labels page";
  else if (job.added && job.labeled === job.added) added += `, their water labeled by the model of ${runDate(job.model)}`;
  else if (job.added) added += `, ${job.labeled} of them labeled by the model of ${runDate(job.model)}`;
  if (job.state === "done") return `Done: ${added}.`;
  if (job.state === "cancelled") return `Cancelled: ${added}.`;
  return `Adding failed: ${job.error}. So far: ${added}.`;
}

function startPolling() {
  if (!state.polling) state.polling = setTimeout(poll, 500);
}

async function poll() {
  state.polling = null;
  let job;
  try {
    ({ job } = await requestJSON("api/add/job"));
  } catch (error) {
    showError(error.message); // e.g. the server restarted: try again
    state.polling = setTimeout(poll, 3000);
    return;
  }
  showJob(job);
  if (running()) state.polling = setTimeout(poll, 1000);
  else attempt(loadFolder(state.listing.folder)); // the videos' frames in the dataset changed
}

// For the browser test
window.__add = { listing: () => state.listing, job: () => state.job };

main().catch((error) => showError(error.message));
