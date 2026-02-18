const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const newChatBtn = document.getElementById("new-chat");

let currentStreamAbort = null;

function createMessageElement(role, text, isStreaming = false) {
  const container = document.createElement("article");
  container.className = `message message--${role}`;

  const content = document.createElement("div");
  content.className = "message__content";
  content.textContent = text;
  container.appendChild(content);

  const meta = document.createElement("div");
  meta.className = "message__meta";

  const roleLabel = document.createElement("span");
  roleLabel.className = "message__role";
  roleLabel.textContent = role === "user" ? "You" : "Assistant";
  meta.appendChild(roleLabel);

  if (isStreaming) {
    const indicator = document.createElement("span");
    indicator.className = "stream-indicator";
    indicator.innerHTML = `
      <span class="stream-indicator__dot"></span>
      <span class="stream-indicator__dot"></span>
      <span class="stream-indicator__dot"></span>
    `;
    meta.appendChild(indicator);
  }

  const timeLabel = document.createElement("span");
  timeLabel.className = "message__time";
  const now = new Date();
  timeLabel.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  meta.appendChild(timeLabel);

  container.appendChild(meta);

  return { container, content, meta };
}

function scrollToBottom() {
  if (messagesEl) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

async function sendMessage(text) {
  if (!messagesEl) return;

  const userMessage = createMessageElement("user", text);
  messagesEl.appendChild(userMessage.container);
  scrollToBottom();

  const botMessage = createMessageElement("bot", "", true);
  messagesEl.appendChild(botMessage.container);
  scrollToBottom();

  const controller = new AbortController();
  currentStreamAbort = controller;

  try {
    const response = await fetch("/chatbot/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: text }),
      credentials: "include",
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      botMessage.content.textContent = "Unable to reach chatbot. Please try again.";
      const streamingIndicator = botMessage.meta.querySelector(".stream-indicator");
      if (streamingIndicator) streamingIndicator.remove();
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let done = false;

    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;
      if (value) {
        const chunk = decoder.decode(value, { stream: true });
        botMessage.content.textContent += chunk;
        scrollToBottom();
      }
    }
  } catch (error) {
    if (controller.signal.aborted) {
      botMessage.content.textContent = "Stream cancelled.";
    } else {
      botMessage.content.textContent = "Something went wrong while streaming the response.";
    }
  } finally {
    const streamingIndicator = botMessage.meta.querySelector(".stream-indicator");
    if (streamingIndicator) streamingIndicator.remove();
    currentStreamAbort = null;
  }
}

if (formEl && inputEl) {
  formEl.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    inputEl.style.height = "auto";

    if (currentStreamAbort) {
      currentStreamAbort.abort();
    }

    await sendMessage(text);
  });

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${inputEl.scrollHeight}px`;
  });
}

if (newChatBtn && messagesEl) {
  newChatBtn.addEventListener("click", () => {
    messagesEl.innerHTML = "";
    if (currentStreamAbort) {
      currentStreamAbort.abort();
      currentStreamAbort = null;
    }
  });
}

