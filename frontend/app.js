const API_URL = 'http://127.0.0.1:8000';

const form = document.querySelector('#chatForm');
const input = document.querySelector('#messageInput');
const messages = document.querySelector('#messages');
const sendButton = document.querySelector('#sendButton');
const widget = document.querySelector('#gemmyWidget');
const launcher = document.querySelector('#gemmyLauncher');
const openButton = document.querySelector('#openGemmy');
const closeButton = document.querySelector('#closeGemmy');

if (widget) {
  widget.classList.add('is-open');
  widget.setAttribute('aria-hidden', 'false');
}

function addMessage(role, content, temporary = false) {
  const article = document.createElement('article');
  article.className = `message ${role}-message${temporary ? ' typing' : ''}`;
  article.innerHTML = `
    <div class="message-avatar" aria-hidden="true">${role === 'assistant' ? 'G' : 'Y'}</div>
    <div class="message-body">
      <span class="message-label">${role === 'assistant' ? 'Gemmy' : 'You'}</span>
      <p></p>
    </div>
  `;
  article.querySelector('p').textContent = content;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function setBusy(busy) {
  input.disabled = busy;
  sendButton.disabled = busy;
  sendButton.querySelector('span').textContent = busy ? '...' : '➤';
}

async function sendMessage(message) {
  const typingMessage = addMessage('assistant', 'Gemmy is thinking...', true);
  setBusy(true);
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error('Gemmy is temporarily busy. Please try again in a moment.');
    }
    typingMessage.remove();
    addMessage('assistant', data.response);
  } catch (error) {
    typingMessage.remove();
    addMessage('assistant', error.message);
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || sendButton.disabled) return;
  addMessage('user', message);
  input.value = '';
  input.style.height = 'auto';
  sendMessage(message);
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function toggleWidget(open) {
  widget.classList.toggle('is-open', open);
  widget.setAttribute('aria-hidden', String(!open));
  if (open) input.focus();
}

if (launcher) launcher.addEventListener('click', () => toggleWidget(true));
if (openButton) openButton.addEventListener('click', () => toggleWidget(true));
if (closeButton) closeButton.addEventListener('click', () => toggleWidget(false));

document.querySelectorAll('[data-scroll-target]').forEach((button) => {
  const target = document.querySelector(button.dataset.scrollTarget);
  if (target) {
    button.addEventListener('click', () => target.scrollIntoView({ behavior: 'smooth' }));
  }
});
