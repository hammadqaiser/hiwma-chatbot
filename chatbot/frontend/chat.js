/**
 * Waste Management Chatbot — Core Chat Logic
 *
 * Handles: API communication, message rendering, session management,
 * typing indicators, and source citation display.
 *
 * API Contract:
 *   POST /api/chat  { message, session_id? } → { answer, sources[], session_id, usage }
 *   GET  /api/health → { status, version }
 */

const ChatApp = (() => {
  // ─── Configuration ────────────────────────────────────────────────
  const API_BASE = window.CHATBOT_API_URL || '';
  const SESSION_KEY = 'wm_chatbot_session_id';

  // ─── State ────────────────────────────────────────────────────────
  let sessionId = localStorage.getItem(SESSION_KEY) || null;
  let isLoading = false;

  // ─── DOM References ───────────────────────────────────────────────
  const elements = {};

  function init() {
    elements.messagesContainer = document.getElementById('chat-messages');
    elements.inputField = document.getElementById('chat-input');
    elements.sendButton = document.getElementById('chat-send-btn');
    elements.welcomeMessage = document.getElementById('welcome-message');

    // Event listeners
    elements.sendButton.addEventListener('click', handleSend);
    elements.inputField.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    // Auto-resize textarea
    elements.inputField.addEventListener('input', () => {
      elements.inputField.style.height = 'auto';
      elements.inputField.style.height = Math.min(elements.inputField.scrollHeight, 120) + 'px';
    });

    // Suggestion chips
    document.querySelectorAll('.suggestion-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        elements.inputField.value = chip.textContent;
        handleSend();
      });
    });
  }

  // ─── Send Message ─────────────────────────────────────────────────
  async function handleSend() {
    const message = elements.inputField.value.trim();
    if (!message || isLoading) return;

    // Hide welcome message
    if (elements.welcomeMessage) {
      elements.welcomeMessage.style.display = 'none';
    }

    // Render user message
    renderMessage('user', message);
    elements.inputField.value = '';
    elements.inputField.style.height = 'auto';

    // Show typing indicator
    isLoading = true;
    elements.sendButton.disabled = true;
    const typingEl = showTypingIndicator();

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
        }),
      });

      // Remove typing indicator
      typingEl.remove();

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();

      // Save session
      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);

      // Render bot response
      renderMessage('bot', data.answer, data.sources, data.usage);

    } catch (error) {
      typingEl.remove();
      renderMessage('error', `Sorry, something went wrong: ${error.message}`);
    } finally {
      isLoading = false;
      elements.sendButton.disabled = false;
      elements.inputField.focus();
    }
  }

  // ─── Render Message ───────────────────────────────────────────────
  function renderMessage(type, text, sources = [], usage = null) {
    const messageEl = document.createElement('div');
    messageEl.className = `message message--${type}`;

    const avatarEmoji = type === 'user' ? '👤' : type === 'error' ? '⚠️' : '♻️';

    let html = `
      <div class="message__avatar">${avatarEmoji}</div>
      <div class="message__content">
        <div class="message__bubble">${formatText(text)}</div>
    `;

    // Source citations
    if (sources && sources.length > 0) {
      html += renderSources(sources);
    }

    // Token usage
    if (usage && usage.total_tokens > 0) {
      html += `
        <div class="message__usage">
          <span class="usage-badge">⚡ ${usage.total_tokens} tokens</span>
          <span class="usage-badge">${usage.prompt_tokens} in / ${usage.completion_tokens} out</span>
        </div>
      `;
    }

    html += '</div>';
    messageEl.innerHTML = html;

    elements.messagesContainer.appendChild(messageEl);
    scrollToBottom();

    // Attach source toggle listener
    const toggle = messageEl.querySelector('.source-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const cards = messageEl.querySelector('.source-cards');
        const icon = toggle.querySelector('.source-toggle__icon');
        cards.classList.toggle('source-cards--open');
        icon.classList.toggle('source-toggle__icon--open');
      });
    }
  }

  // ─── Format Text ──────────────────────────────────────────────────
  function formatText(text) {
    // Escape HTML
    let safe = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Bold: **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text*
    safe = safe.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Inline code: `code`
    safe = safe.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:0.85em;">$1</code>');

    // Newlines
    safe = safe.replace(/\n/g, '<br>');

    return safe;
  }

  // ─── Render Source Citations ───────────────────────────────────────
  function renderSources(sources) {
    const cards = sources.map((s) => {
      const score = Math.round((s.relevance_score || 0) * 100);
      return `
        <div class="source-card">
          <span class="source-card__icon">📖</span>
          <div class="source-card__info">
            <span class="source-card__title">${escapeHtml(s.book_title || 'Unknown')}</span>
            <span class="source-card__detail">${escapeHtml(s.chapter || '')} · ${s.page_range || `p.${s.page_number}`}</span>
          </div>
          <span class="source-card__score">${score}%</span>
        </div>
      `;
    }).join('');

    return `
      <div class="message__sources">
        <button class="source-toggle">
          <span class="source-toggle__icon">▶</span>
          <span>${sources.length} source${sources.length > 1 ? 's' : ''} cited</span>
        </button>
        <div class="source-cards">${cards}</div>
      </div>
    `;
  }

  // ─── Typing Indicator ────────────────────────────────────────────
  function showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message message--bot';
    el.innerHTML = `
      <div class="message__avatar">♻️</div>
      <div class="message__content">
        <div class="message__bubble">
          <div class="typing-indicator">
            <div class="typing-indicator__dot"></div>
            <div class="typing-indicator__dot"></div>
            <div class="typing-indicator__dot"></div>
          </div>
        </div>
      </div>
    `;
    elements.messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
  }

  // ─── Helpers ──────────────────────────────────────────────────────
  function scrollToBottom() {
    requestAnimationFrame(() => {
      elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ─── Public API (for widget.js) ───────────────────────────────────
  return { init };
})();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', ChatApp.init);
