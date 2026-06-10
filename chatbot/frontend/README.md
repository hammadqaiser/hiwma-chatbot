# Frontend Chat UI

The frontend is built entirely with vanilla HTML, CSS, and JavaScript — no build step, no node modules, no framework dependencies. It is served as static files by the FastAPI backend.

## Two Delivery Modes

### Standalone Chat Application

A full-page dark-themed chat interface designed for direct use.

- **Entry:** `index.html`
- **Logic:** `chat.js`
- **Styling:** `styles.css` — dark glassmorphism design with animated gradients, backdrop blur, and micro-interactions.

Features:
- Welcome screen with suggested questions
- Real-time message rendering with typing indicators
- Expandable citation cards (book title, chapter, page range, relevance score)
- Session management (conversations persist via the API)
- Responsive layout (desktop and mobile)

### Embeddable Widget

A floating chat bubble designed to be embedded in the HIWMA waste management simulator or any external web application.

- **Entry:** `widget.js` — include via a single `<script>` tag
- **Demo:** `widget-demo.html` — shows the widget running on a simulated dashboard

Features:
- Shadow DOM encapsulation — all HTML and CSS are isolated inside a shadow root. The widget's styles cannot leak into the host page, and the host page's styles cannot break the widget.
- Floating action button (bottom-right) with open/close animation
- Same chat functionality as the standalone page
- One-line integration: `<script src="/widget.js"></script>`

## Development

No build tools are required. Edit the source files directly. With the backend running in `--reload` mode, refresh the browser to see changes.

```bash
# Start the backend (serves the frontend automatically)
uvicorn chatbot.api.main:app --reload --port 8000

# Then open:
# http://localhost:8000/          — standalone chat
# http://localhost:8000/widget-demo.html — widget demo
```
