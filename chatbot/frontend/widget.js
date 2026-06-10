/**
 * Waste Management Chatbot — Embeddable Widget
 *
 * Drop this script on any page to add a floating chat widget:
 *   <script src="https://your-domain/widget.js" data-api-url="https://your-api"></script>
 *
 * The widget injects its own styles and HTML — fully self-contained.
 * Uses Shadow DOM for CSS isolation from the host page.
 */

(function () {
  'use strict';

  // ─── Read config from script tag ──────────────────────────────────
  const scriptTag = document.currentScript;
  const API_URL = (scriptTag && scriptTag.getAttribute('data-api-url')) || '';
  const POSITION = (scriptTag && scriptTag.getAttribute('data-position')) || 'bottom-right';
  const SESSION_KEY = 'wm_chatbot_widget_session';

  // ─── State ────────────────────────────────────────────────────────
  let isOpen = false;
  let isLoading = false;
  let sessionId = null;

  try { sessionId = localStorage.getItem(SESSION_KEY); } catch (e) { /* no-op */ }

  // ─── Create Widget Container ──────────────────────────────────────
  const widgetHost = document.createElement('div');
  widgetHost.id = 'wm-chatbot-widget';
  document.body.appendChild(widgetHost);

  const shadow = widgetHost.attachShadow({ mode: 'open' });

  // ─── Inject Styles ────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    :host {
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --bg-dark: #0f172a;
      --bg-card: #1e293b;
      --bg-input: #334155;
      --text: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --border: rgba(148,163,184,0.15);
      --radius: 1rem;
      --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── Floating Button ─────────────────── */
    .widget-fab {
      position: fixed;
      ${POSITION === 'bottom-left' ? 'left: 20px;' : 'right: 20px;'}
      bottom: 20px;
      width: 60px;
      height: 60px;
      border-radius: 50%;
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 20px rgba(59,130,246,0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.6rem;
      z-index: 999999;
      transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .widget-fab:hover {
      transform: scale(1.1);
      box-shadow: 0 6px 30px rgba(59,130,246,0.5);
    }
    .widget-fab--hidden { display: none; }

    /* ── Chat Panel ──────────────────────── */
    .widget-panel {
      position: fixed;
      ${POSITION === 'bottom-left' ? 'left: 20px;' : 'right: 20px;'}
      bottom: 20px;
      width: 400px;
      height: 600px;
      max-height: calc(100vh - 40px);
      max-width: calc(100vw - 40px);
      background: var(--bg-dark);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      z-index: 999999;
      box-shadow: 0 12px 48px rgba(0,0,0,0.5);
      animation: widget-slide-in 0.3s cubic-bezier(0.4,0,0.2,1);
      font-family: var(--font);
    }
    .widget-panel--hidden { display: none; }

    @keyframes widget-slide-in {
      from { opacity: 0; transform: translateY(20px) scale(0.95); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ── Panel Header ────────────────────── */
    .widget-header {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.03);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .widget-header__icon {
      width: 36px; height: 36px;
      border-radius: 8px;
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      display: flex; align-items: center; justify-content: center;
      font-size: 1.1rem;
    }
    .widget-header__title {
      flex: 1;
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--text);
    }
    .widget-header__subtitle {
      font-size: 0.7rem;
      color: var(--text-secondary);
      font-weight: 400;
    }
    .widget-close {
      width: 32px; height: 32px;
      border: none;
      background: rgba(255,255,255,0.06);
      border-radius: 8px;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1.1rem;
      display: flex; align-items: center; justify-content: center;
      transition: background 0.15s;
    }
    .widget-close:hover { background: rgba(255,255,255,0.12); color: var(--text); }

    /* ── Messages ─────────────────────────── */
    .widget-messages {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .widget-messages::-webkit-scrollbar { width: 4px; }
    .widget-messages::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 4px; }

    .w-msg {
      max-width: 85%;
      animation: w-msg-in 0.3s ease both;
    }
    @keyframes w-msg-in {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .w-msg--user { align-self: flex-end; }
    .w-msg--bot { align-self: flex-start; }

    .w-msg__bubble {
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 0.85rem;
      line-height: 1.55;
      word-break: break-word;
    }
    .w-msg--user .w-msg__bubble {
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .w-msg--bot .w-msg__bubble {
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text);
      border-bottom-left-radius: 4px;
    }
    .w-msg--error .w-msg__bubble {
      background: rgba(239,68,68,0.1);
      border: 1px solid rgba(239,68,68,0.3);
      color: #fca5a5;
    }

    /* ── Sources ──────────────────────────── */
    .w-sources-btn {
      background: none; border: none;
      color: #60a5fa; font-size: 0.7rem;
      cursor: pointer; padding: 4px 0;
      font-family: var(--font);
    }
    .w-source {
      font-size: 0.7rem;
      color: var(--text-muted);
      padding: 4px 8px;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-top: 4px;
    }
    .w-source__title { color: var(--text); font-weight: 600; }

    /* ── Typing ───────────────────────────── */
    .w-typing { display: flex; gap: 4px; padding: 10px 14px; }
    .w-typing__dot {
      width: 6px; height: 6px;
      background: var(--text-muted);
      border-radius: 50%;
      animation: w-bounce 1.4s ease infinite;
    }
    .w-typing__dot:nth-child(2) { animation-delay: 0.15s; }
    .w-typing__dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes w-bounce {
      0%,60%,100% { transform: translateY(0); opacity: 0.4; }
      30% { transform: translateY(-5px); opacity: 1; }
    }

    /* ── Input ────────────────────────────── */
    .widget-input-area {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      flex-shrink: 0;
    }
    .widget-input {
      flex: 1;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      color: var(--text);
      font-size: 0.85rem;
      font-family: var(--font);
      outline: none;
      resize: none;
      max-height: 80px;
      line-height: 1.4;
    }
    .widget-input::placeholder { color: var(--text-muted); }
    .widget-input:focus { border-color: rgba(96,165,250,0.5); }
    .widget-send {
      width: 40px; height: 40px;
      border: none;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font-size: 1rem;
      display: flex; align-items: center; justify-content: center;
      transition: background 0.15s, transform 0.15s;
      flex-shrink: 0;
    }
    .widget-send:hover:not(:disabled) { background: var(--accent-hover); transform: scale(1.05); }
    .widget-send:disabled { opacity: 0.4; cursor: not-allowed; }

    /* ── Welcome ──────────────────────────── */
    .w-welcome {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 24px;
      gap: 12px;
    }
    .w-welcome__icon { font-size: 2.5rem; }
    .w-welcome__title {
      font-size: 1.1rem; font-weight: 700;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .w-welcome__text { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; }
  `;
  shadow.appendChild(style);

  // ─── Load Inter font in host document ─────────────────────────────
  if (!document.querySelector('link[href*="Inter"]')) {
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }

  // ─── Build HTML ───────────────────────────────────────────────────
  const container = document.createElement('div');
  container.innerHTML = `
    <button class="widget-fab" id="widget-fab">♻️</button>

    <div class="widget-panel widget-panel--hidden" id="widget-panel">
      <div class="widget-header">
        <div class="widget-header__icon">♻️</div>
        <div>
          <div class="widget-header__title">Waste Management Assistant</div>
          <div class="widget-header__subtitle">AI-powered book knowledge</div>
        </div>
        <button class="widget-close" id="widget-close">✕</button>
      </div>

      <div class="widget-messages" id="widget-messages">
        <div class="w-welcome" id="w-welcome">
          <div class="w-welcome__icon">♻️</div>
          <div class="w-welcome__title">Ask me anything</div>
          <div class="w-welcome__text">Get answers about waste management grounded in technical books.</div>
        </div>
      </div>

      <div class="widget-input-area">
        <textarea class="widget-input" id="widget-input" placeholder="Ask a question..." rows="1"></textarea>
        <button class="widget-send" id="widget-send">➤</button>
      </div>
    </div>
  `;
  shadow.appendChild(container);

  // ─── DOM Refs ─────────────────────────────────────────────────────
  const fab = shadow.getElementById('widget-fab');
  const panel = shadow.getElementById('widget-panel');
  const closeBtn = shadow.getElementById('widget-close');
  const messagesEl = shadow.getElementById('widget-messages');
  const inputEl = shadow.getElementById('widget-input');
  const sendBtn = shadow.getElementById('widget-send');
  const welcomeEl = shadow.getElementById('w-welcome');

  // ─── Toggle ───────────────────────────────────────────────────────
  fab.addEventListener('click', () => {
    isOpen = true;
    fab.classList.add('widget-fab--hidden');
    panel.classList.remove('widget-panel--hidden');
    inputEl.focus();
  });

  closeBtn.addEventListener('click', () => {
    isOpen = false;
    panel.classList.add('widget-panel--hidden');
    fab.classList.remove('widget-fab--hidden');
  });

  // ─── Send ─────────────────────────────────────────────────────────
  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  async function sendMessage() {
    const msg = inputEl.value.trim();
    if (!msg || isLoading) return;

    if (welcomeEl) welcomeEl.style.display = 'none';
    addMsg('user', msg);
    inputEl.value = '';

    isLoading = true;
    sendBtn.disabled = true;
    const typing = addTyping();

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });
      typing.remove();

      if (!res.ok) throw new Error(`Error ${res.status}`);
      const data = await res.json();
      sessionId = data.session_id;
      try { localStorage.setItem(SESSION_KEY, sessionId); } catch (e) { /* no-op */ }

      addMsg('bot', data.answer, data.sources);
    } catch (err) {
      typing.remove();
      addMsg('error', `Sorry: ${err.message}`);
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // ─── Message Helpers ──────────────────────────────────────────────
  function addMsg(type, text, sources) {
    const el = document.createElement('div');
    el.className = `w-msg w-msg--${type}`;

    let safe = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    let html = `<div class="w-msg__bubble">${safe}</div>`;

    if (sources && sources.length) {
      const id = 'src-' + Math.random().toString(36).slice(2, 8);
      html += `<button class="w-sources-btn" data-target="${id}">📖 ${sources.length} source${sources.length>1?'s':''}</button>`;
      html += `<div id="${id}" style="display:none">`;
      sources.forEach(s => {
        html += `<div class="w-source"><span class="w-source__title">${esc(s.book_title||'')}</span> · ${esc(s.chapter||'')} · ${s.page_range||'p.'+s.page_number}</div>`;
      });
      html += '</div>';
    }

    el.innerHTML = html;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Toggle sources
    const btn = el.querySelector('.w-sources-btn');
    if (btn) {
      btn.addEventListener('click', () => {
        const t = shadow.getElementById(btn.dataset.target);
        t.style.display = t.style.display === 'none' ? 'block' : 'none';
      });
    }
  }

  function addTyping() {
    const el = document.createElement('div');
    el.className = 'w-msg w-msg--bot';
    el.innerHTML = `<div class="w-msg__bubble"><div class="w-typing"><div class="w-typing__dot"></div><div class="w-typing__dot"></div><div class="w-typing__dot"></div></div></div>`;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }
})();
