const form = document.getElementById('buildForm');
const initiateBtn = document.getElementById('initiateBtn');
const errorBanner = document.getElementById('errorBanner');
const logEl = document.getElementById('log');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const baton = document.getElementById('baton');
const filesStrip = document.getElementById('filesStrip');
const filesEmpty = document.getElementById('filesEmpty');
const trackCaption = document.getElementById('trackCaption');
const outputActions = document.getElementById('outputActions');
const downloadZipBtn = document.getElementById('downloadZipBtn');
const previewBtn = document.getElementById('previewBtn');
const previewOverlay = document.getElementById('previewOverlay');
const previewIframe = document.getElementById('previewIframe');
const previewNewTab = document.getElementById('previewNewTab');
const closePreview = document.getElementById('closePreview');

let generatedFiles = [];

let currentJobId = null;

function setStatus(text, mode) {
  statusText.textContent = text;
  statusDot.className = 'status-dot' + (mode ? ' ' + mode : '');
}

function clearLog() {
  logEl.innerHTML = '';
}

function appendLog(message, cls) {
  const empty = logEl.querySelector('.terminal-empty');
  if (empty) empty.remove();

  const line = document.createElement('div');
  line.className = 'line' + (cls ? ' ' + cls : '');

  const ts = document.createElement('span');
  ts.className = 'ts';
  ts.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });

  line.appendChild(ts);
  line.appendChild(document.createTextNode(message));
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function setNodeState(provider, state) {
  const node = document.getElementById('node-' + provider);
  if (!node) return;
  node.className = 'node' + (state ? ' ' + state : '');
  const stateText = node.querySelector('text.state');
  if (stateText) stateText.textContent = state ? state.replace('_', ' ') : 'idle';
}

function resetNodes() {
  ['claude', 'openai', 'gemini'].forEach((p) => setNodeState(p, ''));
}

function addFilePill(jobId, filename) {
  filesEmpty.style.display = 'none';
  generatedFiles.push(filename);
  const a = document.createElement('a');
  a.className = 'file-pill';
  a.href = `/api/download/${jobId}/${encodeURIComponent(filename)}`;
  a.textContent = filename;
  a.setAttribute('download', filename);
  filesStrip.appendChild(a);
}

function resetFiles() {
  filesStrip.querySelectorAll('.file-pill').forEach((el) => el.remove());
  filesEmpty.style.display = 'inline';
  generatedFiles = [];
  outputActions.style.display = 'none';
  previewBtn.style.display = 'none';
}

function findEntryHtml() {
  if (generatedFiles.includes('index.html')) return 'index.html';
  return generatedFiles.find((f) => f.toLowerCase().endsWith('.html')) || null;
}

function revealOutputActions(jobId) {
  outputActions.style.display = 'flex';
  downloadZipBtn.onclick = () => {
    window.location.href = `/api/download-zip/${jobId}`;
  };

  const entry = findEntryHtml();
  if (entry) {
    previewBtn.style.display = 'inline-block';
    const previewUrl = `/api/preview/${jobId}/${entry}`;
    previewBtn.onclick = () => openPreview(previewUrl);
  } else {
    previewBtn.style.display = 'none';
  }
}

function openPreview(url) {
  previewIframe.src = url;
  previewNewTab.href = url;
  previewOverlay.classList.add('open');
}

closePreview.addEventListener('click', () => {
  previewOverlay.classList.remove('open');
  previewIframe.src = 'about:blank';
});

previewOverlay.addEventListener('click', (e) => {
  if (e.target === previewOverlay) {
    previewOverlay.classList.remove('open');
    previewIframe.src = 'about:blank';
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorBanner.style.display = 'none';

  const payload = {
    prompt: document.getElementById('prompt').value.trim(),
    claude_key: document.getElementById('claudeKey').value.trim(),
    openai_key: document.getElementById('openaiKey').value.trim(),
    gemini_key: document.getElementById('geminiKey').value.trim(),
  };

  initiateBtn.disabled = true;
  initiateBtn.textContent = 'Relay running...';
  clearLog();
  resetNodes();
  resetFiles();
  baton.classList.remove('running');
  setStatus('launching', 'live');
  trackCaption.textContent = 'Runners connecting...';

  let res;
  try {
    res = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    showError('Server tak nahi pahunch paaya. Backend chal raha hai?');
    resetButton();
    return;
  }

  const data = await res.json();
  if (!res.ok) {
    showError(data.error || 'Kuch galat ho gaya.');
    resetButton();
    return;
  }

  currentJobId = data.job_id;
  baton.classList.add('running');
  streamJob(currentJobId);
});

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.style.display = 'block';
  setStatus('error', 'danger');
}

function resetButton() {
  initiateBtn.disabled = false;
  initiateBtn.textContent = 'Initiate Relay';
}

function streamJob(jobId) {
  const source = new EventSource(`/api/stream/${jobId}`);

  source.onmessage = (evt) => {
    const event = JSON.parse(evt.data);

    switch (event.type) {
      case 'log':
        appendLog(event.message, classify(event.message));
        break;

      case 'provider_status':
        setNodeState(event.provider, event.status);
        if (event.status === 'active') {
          trackCaption.textContent = `${capitalize(event.provider)} has the baton.`;
        } else if (event.status === 'quota_exceeded') {
          trackCaption.textContent = `${capitalize(event.provider)} tapped out — passing the baton.`;
        }
        break;

      case 'file_written':
        addFilePill(jobId, event.filename);
        break;

      case 'done':
        baton.classList.remove('running');
        setStatus('complete', '');
        trackCaption.textContent = 'Relay complete. Files are ready below.';
        resetButton();
        revealOutputActions(jobId);
        break;

      case 'paused':
        baton.classList.remove('running');
        setStatus('paused — quota exhausted', 'danger');
        trackCaption.textContent = 'Sab providers ruk gaye. Keys top-up karke dobara Initiate karo.';
        resetButton();
        break;

      case 'error':
        baton.classList.remove('running');
        appendLog(event.message, 'error');
        setStatus('error', 'danger');
        resetButton();
        break;

      case 'stream_closed':
        source.close();
        break;
    }
  };

  source.onerror = () => {
    source.close();
    if (statusText.textContent === 'launching' || statusText.textContent.includes('runner')) {
      setStatus('disconnected', 'danger');
    }
    resetButton();
  };
}

function classify(message) {
  if (message.includes('⚠️')) return 'quota_exceeded';
  if (message.includes('🎉') || message.includes('✓')) return 'success';
  if (message.includes('🏗️') || message.includes('📝') || message.includes('📂')) return 'system';
  return '';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
