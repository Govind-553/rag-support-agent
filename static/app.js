/* Aster & Row Support Agent — Frontend Logic */

const API_BASE = window.location.origin;
let sessionId = generateSessionId();

function generateSessionId() {
  return 'sess-' + Math.random().toString(36).slice(2, 10);
}

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const newChatBtn = document.getElementById('newChat');

// Auto-resize textarea
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

// Send on Enter (Shift+Enter = newline)
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

newChatBtn.addEventListener('click', () => {
  sessionId = generateSessionId();
  messagesEl.innerHTML = '';
  appendMessage('assistant', {
    answer: "Started a new conversation. How can I help you?",
    sources: [],
    handoff: false,
    tool_used: false,
  });
});

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  // Render user message
  const userEl = document.createElement('div');
  userEl.className = 'message user';
  userEl.innerHTML = `<div class="bubble"><p>${escapeHtml(text)}</p></div>`;
  messagesEl.appendChild(userEl);
  scrollDown();

  // Show typing indicator
  const typingEl = document.createElement('div');
  typingEl.className = 'message assistant typing';
  typingEl.innerHTML = `<div class="bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  messagesEl.appendChild(typingEl);
  scrollDown();

  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    typingEl.remove();
    appendMessage('assistant', data);
  } catch (err) {
    typingEl.remove();
    appendMessage('assistant', {
      answer: `Sorry, I encountered an error: ${err.message}. Please try again.`,
      sources: [],
      handoff: false,
      tool_used: false,
    });
  }

  sendBtn.disabled = false;
  inputEl.focus();
  scrollDown();
}

function appendMessage(role, data) {
  const el = document.createElement('div');
  el.className = `message ${role}`;

  // Format answer with basic markdown (newlines → paragraphs)
  const paragraphs = data.answer
    .split(/\n\n+/)
    .map(p => `<p>${escapeHtml(p.trim()).replace(/\n/g, '<br/>')}</p>`)
    .join('');

  let metaHtml = '';

  if (data.tool_used) {
    metaHtml += `<span class="tag-tool">🔍 Order lookup</span>`;
  }

  if (data.sources && data.sources.length > 0) {
    data.sources.forEach(src => {
      metaHtml += `<span class="tag-source">📄 ${escapeHtml(src.filename)}</span>`;
    });
  }

  if (data.handoff) {
    metaHtml += `<span class="tag-handoff">⚑ Human support recommended</span>`;
  }

  el.innerHTML = `
    <div class="bubble">${paragraphs}</div>
    ${metaHtml ? `<div class="meta">${metaHtml}</div>` : ''}
  `;

  messagesEl.appendChild(el);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
