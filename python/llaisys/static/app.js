const API_URL = "/v1/chat/completions";
const STORAGE_KEY = "llaisys-chat-ui-v1";
const DEFAULT_MODEL = "deepseek-r1-distill-qwen-1.5b";

const elements = {
    conversationList: document.getElementById("conversation-list"),
    conversationTitle: document.getElementById("conversation-title"),
    conversationStatus: document.getElementById("conversation-status"),
    messageList: document.getElementById("message-list"),
    newChatButton: document.getElementById("new-chat-btn"),
    renameChatButton: document.getElementById("rename-chat-btn"),
    deleteChatButton: document.getElementById("delete-chat-btn"),
    composerForm: document.getElementById("composer-form"),
    composerInput: document.getElementById("composer-input"),
    sendButton: document.getElementById("send-btn"),
    streamToggle: document.getElementById("stream-toggle"),
    temperatureInput: document.getElementById("temperature-input"),
    maxTokensInput: document.getElementById("max-tokens-input"),
};

const state = {
    conversations: [],
    activeConversationId: null,
    editorMessageId: null,
    expandedThinkBlocks: {},
    requestControllers: {},
};

function makeId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return `${prefix}-${window.crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createConversation() {
    return {
        id: makeId("conversation"),
        sessionId: makeId("session"),
        title: "新建对话",
        messages: [],
        updatedAt: Date.now(),
        inFlight: false,
        stopping: false,
        abortRequested: false,
        error: "",
    };
}

function getActiveConversation() {
    return state.conversations.find((item) => item.id === state.activeConversationId) || null;
}

function saveState() {
    const payload = {
        activeConversationId: state.activeConversationId,
        conversations: state.conversations.map(({ id, sessionId, title, messages, updatedAt }) => ({
            id,
            sessionId,
            title,
            messages,
            updatedAt,
        })),
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function loadState() {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
        const initialConversation = createConversation();
        state.conversations = [initialConversation];
        state.activeConversationId = initialConversation.id;
        return;
    }

    try {
        const parsed = JSON.parse(raw);
        const conversations = Array.isArray(parsed.conversations)
            ? parsed.conversations
                  .filter((item) => item && item.id && item.sessionId && Array.isArray(item.messages))
                  .map((item) => ({
                      id: item.id,
                      sessionId: item.sessionId,
                      title: item.title || "新建对话",
                      messages: item.messages.map((message) => ({
                          id: message.id || makeId("message"),
                          role: message.role,
                          content: message.content || "",
                      })),
                      updatedAt: item.updatedAt || Date.now(),
                      inFlight: false,
                      stopping: false,
                      abortRequested: false,
                      error: "",
                  }))
            : [];

        if (conversations.length === 0) {
            const initialConversation = createConversation();
            state.conversations = [initialConversation];
            state.activeConversationId = initialConversation.id;
            return;
        }

        state.conversations = conversations;
        state.activeConversationId = conversations.some((item) => item.id === parsed.activeConversationId)
            ? parsed.activeConversationId
            : conversations[0].id;
    } catch (error) {
        console.error("Failed to parse saved conversations", error);
        const initialConversation = createConversation();
        state.conversations = [initialConversation];
        state.activeConversationId = initialConversation.id;
    }
}

function updateConversationTitle(conversation) {
    const firstUserMessage = conversation.messages.find((message) => message.role === "user");
    if (!firstUserMessage || !firstUserMessage.content.trim()) {
        conversation.title = "新建对话";
        return;
    }

    const compact = firstUserMessage.content.trim().replace(/\s+/g, " ");
    conversation.title = compact.length > 28 ? `${compact.slice(0, 28)}...` : compact;
}

function getConversationStatus(conversation) {
    return conversation.error || "";
}

function escapeHtml(text) {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function normalizeAssistantMarkup(text) {
    const raw = text || "";
    const closeIndex = raw.indexOf("</think>");
    const openIndex = raw.indexOf("<think>");

    if (closeIndex !== -1 && (openIndex === -1 || openIndex > closeIndex)) {
        return raw.startsWith("\n") ? `<think>${raw}` : `<think>\n${raw}`;
    }
    return raw;
}

function parseAssistantMessage(text) {
    const normalized = normalizeAssistantMarkup(text);
    const openTag = "<think>";
    const closeTag = "</think>";
    const openIndex = normalized.indexOf(openTag);
    const closeIndex = normalized.indexOf(closeTag);

    if (openIndex === -1) {
        return {
            thinking: "",
            answer: normalized,
            hasThink: false,
            incomplete: false,
        };
    }

    const thinkStart = openIndex + openTag.length;
    if (closeIndex === -1 || closeIndex < openIndex) {
        return {
            thinking: normalized.slice(thinkStart).replace(/^\n/, ""),
            answer: "",
            hasThink: true,
            incomplete: true,
        };
    }

    return {
        thinking: normalized.slice(thinkStart, closeIndex).replace(/^\n/, ""),
        answer: normalized.slice(closeIndex + closeTag.length).replace(/^\n+/, ""),
        hasThink: true,
        incomplete: false,
    };
}

function findPendingAssistantIndex(conversation) {
    for (let index = conversation.messages.length - 1; index >= 0; index -= 1) {
        const message = conversation.messages[index];
        if (message.role === "assistant" && message.pending) {
            return index;
        }
    }
    return -1;
}

function focusEditor(messageId) {
    const textarea = document.querySelector(`[data-editor-id="${messageId}"]`);
    if (!textarea) {
        return;
    }
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function renderSidebar() {
    const previousScrollTop = elements.conversationList.scrollTop;
    elements.conversationList.innerHTML = "";

    state.conversations.forEach((conversation) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `conversation-card${conversation.id === state.activeConversationId ? " active" : ""}`;
        button.dataset.conversationId = conversation.id;

        const updatedAt = new Date(conversation.updatedAt).toLocaleString();
        const messageCount = `${conversation.messages.length} 条消息`;
        button.innerHTML = `
            <p class="conversation-title">${escapeHtml(conversation.title)}</p>
            <p class="conversation-meta">${escapeHtml(messageCount)} · ${escapeHtml(updatedAt)}</p>
        `;

        button.addEventListener("click", () => {
            state.activeConversationId = conversation.id;
            state.editorMessageId = null;
            render();
        });
        elements.conversationList.appendChild(button);
    });

    elements.conversationList.scrollTop = previousScrollTop;
}

function renderMessages(conversation) {
    elements.messageList.innerHTML = "";

    if (conversation.messages.length === 0) {
        const emptyState = document.createElement("div");
        emptyState.className = "empty-state";
        emptyState.textContent = "开始新对话，切换不同会话，编辑之前的用户消息，或者重新生成上一条回答。";
        elements.messageList.appendChild(emptyState);
        return;
    }

    conversation.messages.forEach((message, index) => {
        const row = document.createElement("div");
        row.className = `message-row ${message.role}`;

        const bubble = document.createElement("article");
        bubble.className = "message-bubble";

        const label = document.createElement("p");
        label.className = "message-label";
        label.textContent = message.role === "user" ? "用户" : "助手";
        bubble.appendChild(label);

        if (state.editorMessageId === message.id && message.role === "user") {
            const editor = document.createElement("div");
            editor.className = "inline-editor";
            editor.innerHTML = `
                <textarea data-editor-id="${message.id}">${escapeHtml(message.content)}</textarea>
                <div class="inline-editor-actions">
                    <button class="ghost-button" type="button" data-cancel-edit="${message.id}">取消</button>
                    <button class="primary-button" type="button" data-save-edit="${message.id}" ${conversation.inFlight || conversation.stopping ? "disabled" : ""}>保存并重答</button>
                </div>
            `;
            bubble.appendChild(editor);
        } else {
            const parsedAssistant = message.role === "assistant"
                ? parseAssistantMessage(message.content)
                : null;

            if (message.role === "assistant" && parsedAssistant && parsedAssistant.hasThink) {
                const details = document.createElement("details");
                details.className = "think-block";
                details.dataset.messageId = message.id;
                details.open = Object.prototype.hasOwnProperty.call(state.expandedThinkBlocks, message.id)
                    ? state.expandedThinkBlocks[message.id]
                    : false;

                const summary = document.createElement("summary");
                summary.textContent = message.pending && parsedAssistant.incomplete
                    ? "思考中..."
                    : "思考过程";
                details.appendChild(summary);

                const thinkText = document.createElement("pre");
                thinkText.className = "think-text";
                thinkText.textContent = parsedAssistant.thinking || (message.pending ? "思考中..." : "");
                details.appendChild(thinkText);
                bubble.appendChild(details);

                if (parsedAssistant.answer) {
                    const answer = document.createElement("p");
                    answer.className = "message-content answer-content";
                    answer.textContent = parsedAssistant.answer;
                    bubble.appendChild(answer);
                }
            } else {
                const content = document.createElement("p");
                content.className = `message-content${message.pending ? " pending" : ""}`;
                content.textContent = message.content || (message.pending ? "思考中..." : "");
                bubble.appendChild(content);
            }

            const actionRow = document.createElement("div");
            actionRow.className = "message-actions";

            if (message.role === "user") {
                const editButton = document.createElement("button");
                editButton.type = "button";
                editButton.className = "ghost-button";
                editButton.textContent = "编辑";
                editButton.disabled = Boolean(conversation.stopping);
                editButton.addEventListener("click", async () => {
                    await beginEditMessage(conversation.id, message.id);
                });
                actionRow.appendChild(editButton);
            }

            const isLatestAssistant = message.role === "assistant" && index === conversation.messages.length - 1;
            if (isLatestAssistant && message.pending && conversation.inFlight) {
                const stopButton = document.createElement("button");
                stopButton.type = "button";
                stopButton.className = "ghost-button";
                stopButton.textContent = "停止生成";
                stopButton.disabled = Boolean(conversation.stopping);
                stopButton.addEventListener("click", async () => {
                    await stopConversationGeneration(conversation.id);
                });
                actionRow.appendChild(stopButton);
            } else if (isLatestAssistant && !message.pending && !conversation.inFlight) {
                const regenerateButton = document.createElement("button");
                regenerateButton.type = "button";
                regenerateButton.className = "ghost-button";
                regenerateButton.textContent = "重答";
                regenerateButton.addEventListener("click", async () => {
                    await regenerateLastAnswer(conversation.id);
                });
                actionRow.appendChild(regenerateButton);
            }

            if (actionRow.childElementCount > 0) {
                bubble.appendChild(actionRow);
            }
        }

        row.appendChild(bubble);
        elements.messageList.appendChild(row);
    });

    elements.messageList.scrollTop = elements.messageList.scrollHeight;
}

function render() {
    const conversation = getActiveConversation();
    if (!conversation) {
        return;
    }

    renderSidebar();
    const statusText = getConversationStatus(conversation);
    elements.conversationTitle.textContent = conversation.title;
    elements.conversationStatus.textContent = statusText;
    elements.conversationStatus.hidden = !statusText;
    elements.sendButton.disabled = conversation.inFlight || conversation.stopping;
    elements.renameChatButton.disabled = conversation.inFlight || conversation.stopping;
    elements.deleteChatButton.disabled = conversation.inFlight || conversation.stopping;
    renderMessages(conversation);

    document.querySelectorAll("[data-cancel-edit]").forEach((button) => {
        button.addEventListener("click", () => {
            state.editorMessageId = null;
            render();
        });
    });

    document.querySelectorAll("[data-save-edit]").forEach((button) => {
        button.addEventListener("click", async () => {
            const messageId = button.dataset.saveEdit;
            const textarea = document.querySelector(`[data-editor-id="${messageId}"]`);
            if (!textarea) {
                return;
            }
            await saveEditedMessage(conversation.id, messageId, textarea.value);
        });
    });

    document.querySelectorAll(".think-block").forEach((details) => {
        details.addEventListener("toggle", () => {
            state.expandedThinkBlocks[details.dataset.messageId] = details.open;
        });
    });
}

function buildPayload(conversation, messages, stream) {
    return {
        model: DEFAULT_MODEL,
        messages: messages.map((message) => ({ role: message.role, content: message.content })),
        session_id: conversation.sessionId,
        max_tokens: Number.parseInt(elements.maxTokensInput.value, 10) || 256,
        temperature: Number.parseFloat(elements.temperatureInput.value) || 0.7,
        top_p: 0.9,
        top_k: 50,
        seed: -1,
        stream,
    };
}

async function parseError(response) {
    const fallback = `请求失败，状态码 ${response.status}`;
    try {
        const payload = await response.json();
        if (payload && typeof payload.detail === "string") {
            return payload.detail;
        }
        return fallback;
    } catch (error) {
        return fallback;
    }
}

function applyAssistantDelta(conversation, assistantMessageId, delta) {
    const assistantMessage = conversation.messages.find((message) => message.id === assistantMessageId);
    if (!assistantMessage) {
        return;
    }
    assistantMessage.content += delta;
    conversation.updatedAt = Date.now();
    saveState();
    if (conversation.id === state.activeConversationId) {
        render();
    }
}

function finishAssistantMessage(conversation, assistantMessageId) {
    const assistantMessage = conversation.messages.find((message) => message.id === assistantMessageId);
    if (!assistantMessage) {
        return;
    }
    assistantMessage.pending = false;
    conversation.inFlight = false;
    conversation.stopping = false;
    conversation.abortRequested = false;
    conversation.error = "";
    conversation.updatedAt = Date.now();
    saveState();
    if (conversation.id === state.activeConversationId) {
        render();
    }
}

function failAssistantMessage(conversation, assistantMessageId, errorMessage) {
    const assistantIndex = conversation.messages.findIndex((message) => message.id === assistantMessageId);
    if (assistantIndex !== -1 && !conversation.messages[assistantIndex].content) {
        conversation.messages.splice(assistantIndex, 1);
    } else if (assistantIndex !== -1) {
        conversation.messages[assistantIndex].pending = false;
    }
    conversation.inFlight = false;
    conversation.stopping = false;
    conversation.abortRequested = false;
    conversation.error = errorMessage;
    conversation.updatedAt = Date.now();
    saveState();
    if (conversation.id === state.activeConversationId) {
        render();
    }
}

function parseSseBlock(block, onDelta) {
    let eventName = "message";
    const dataLines = [];

    for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
            continue;
        }
        if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
        }
    }

    const data = dataLines.join("\n");
    if (!data) {
        return false;
    }
    if (data === "[DONE]") {
        return true;
    }
    if (eventName === "error") {
        throw new Error(data);
    }

    const payload = JSON.parse(data);
    const delta = payload.choices && payload.choices[0] && payload.choices[0].delta
        ? payload.choices[0].delta.content
        : "";
    if (delta) {
        onDelta(delta);
    }
    return false;
}

async function requestStream(payload, onDelta, signal) {
    const response = await window.fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }
    if (!response.body) {
        throw new Error("当前浏览器不支持流式输出。");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        const chunks = buffer.split(/\r?\n\r?\n/);
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
            const isDone = parseSseBlock(chunk, onDelta);
            if (isDone) {
                return;
            }
        }

        if (done) {
            break;
        }
    }

    if (buffer.trim()) {
        parseSseBlock(buffer, onDelta);
    }
}

async function requestJson(payload, signal) {
    const response = await window.fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    const payloadJson = await response.json();
    return payloadJson.choices[0].message.content;
}

async function releaseSession(sessionId) {
    const response = await window.fetch(`/v1/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }
}

async function cancelSessionGeneration(sessionId) {
    const response = await window.fetch(`/v1/chat/sessions/${encodeURIComponent(sessionId)}/cancel`, {
        method: "POST",
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    const payload = await response.json();
    return Boolean(payload && payload.cancelled);
}

function stopAssistantMessage(conversation, errorMessage = "") {
    const assistantIndex = findPendingAssistantIndex(conversation);
    if (assistantIndex !== -1 && !conversation.messages[assistantIndex].content) {
        conversation.messages.splice(assistantIndex, 1);
    } else if (assistantIndex !== -1) {
        conversation.messages[assistantIndex].pending = false;
    }

    conversation.inFlight = false;
    conversation.stopping = false;
    conversation.abortRequested = false;
    conversation.error = errorMessage;
    conversation.updatedAt = Date.now();
    saveState();
    if (conversation.id === state.activeConversationId) {
        render();
    }
}

async function stopConversationGeneration(conversationId) {
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (!conversation) {
        return false;
    }
    if (!conversation.inFlight) {
        return true;
    }
    if (conversation.stopping) {
        return false;
    }

    conversation.stopping = true;
    conversation.error = "";

    const controller = state.requestControllers[conversation.id];
    if (controller) {
        controller.abort();
    }

    stopAssistantMessage(conversation);

    try {
        return await cancelSessionGeneration(conversation.sessionId);
    } catch (error) {
        console.warn("停止生成失败", error);
        return false;
    }
}

async function beginEditMessage(conversationId, messageId) {
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (!conversation || conversation.stopping) {
        return;
    }

    state.editorMessageId = messageId;
    render();
    focusEditor(messageId);

    if (conversation.inFlight) {
        void stopConversationGeneration(conversationId).then(() => {
            if (state.editorMessageId === messageId) {
                focusEditor(messageId);
            }
        });
    }
}

async function runAssistantTurn(conversation) {
    if (conversation.inFlight || conversation.stopping) {
        return;
    }

    const promptMessages = conversation.messages.map((message) => ({
        role: message.role,
        content: message.content,
    }));
    const assistantMessageId = makeId("message");
    conversation.messages.push({
        id: assistantMessageId,
        role: "assistant",
        content: "",
        pending: true,
    });
    conversation.inFlight = true;
    conversation.stopping = false;
    conversation.abortRequested = false;
    conversation.error = "";
    conversation.updatedAt = Date.now();
    saveState();
    render();

    const payload = buildPayload(conversation, promptMessages, elements.streamToggle.checked);
    const controller = new AbortController();
    state.requestControllers[conversation.id] = controller;

    try {
        if (payload.stream) {
            await requestStream(payload, (delta) => applyAssistantDelta(conversation, assistantMessageId, delta), controller.signal);
        } else {
            const content = await requestJson(payload, controller.signal);
            applyAssistantDelta(conversation, assistantMessageId, content);
        }
        if (!controller.signal.aborted) {
            finishAssistantMessage(conversation, assistantMessageId);
        }
    } catch (error) {
        if (error && error.name === "AbortError" && controller.signal.aborted) {
            return;
        }
        if (!controller.signal.aborted) {
            failAssistantMessage(conversation, assistantMessageId, error.message || "请求失败。");
        }
    } finally {
        if (state.requestControllers[conversation.id] === controller) {
            delete state.requestControllers[conversation.id];
        }
    }
}

async function submitPrompt(promptText) {
    const conversation = getActiveConversation();
    if (!conversation || conversation.inFlight || conversation.stopping) {
        return;
    }

    const content = promptText.trim();
    if (!content) {
        return;
    }

    conversation.messages.push({
        id: makeId("message"),
        role: "user",
        content,
    });
    updateConversationTitle(conversation);
    conversation.updatedAt = Date.now();
    conversation.error = "";
    state.editorMessageId = null;
    elements.composerInput.value = "";
    saveState();
    render();
    await runAssistantTurn(conversation);
}

async function saveEditedMessage(conversationId, messageId, newContent) {
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (!conversation || conversation.inFlight || conversation.stopping) {
        return;
    }

    const content = newContent.trim();
    if (!content) {
        window.alert("输入内容不能为空。");
        return;
    }

    const messageIndex = conversation.messages.findIndex((message) => message.id === messageId);
    if (messageIndex === -1) {
        return;
    }

    conversation.messages[messageIndex].content = content;
    conversation.messages = conversation.messages.slice(0, messageIndex + 1);
    updateConversationTitle(conversation);
    conversation.updatedAt = Date.now();
    conversation.error = "";
    state.editorMessageId = null;
    saveState();
    render();
    await runAssistantTurn(conversation);
}

async function regenerateLastAnswer(conversationId) {
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (!conversation || conversation.stopping || conversation.messages.length === 0) {
        return;
    }

    if (conversation.inFlight) {
        const stopped = await stopConversationGeneration(conversationId);
        if (!stopped) {
            return;
        }
    }

    const lastMessage = conversation.messages[conversation.messages.length - 1];
    if (!lastMessage) {
        return;
    }
    if (lastMessage.role !== "assistant") {
        return;
    }

    conversation.messages.pop();
    conversation.updatedAt = Date.now();
    conversation.error = "";
    saveState();
    render();
    await runAssistantTurn(conversation);
}

function createNewConversation() {
    const conversation = createConversation();
    state.conversations.unshift(conversation);
    state.activeConversationId = conversation.id;
    state.editorMessageId = null;
    saveState();
    render();
    elements.composerInput.focus();
}

function renameConversation() {
    const conversation = getActiveConversation();
    if (!conversation || conversation.inFlight) {
        return;
    }

    const nextTitle = window.prompt("请输入新的会话名称", conversation.title);
    if (!nextTitle) {
        return;
    }

    conversation.title = nextTitle.trim() || conversation.title;
    conversation.updatedAt = Date.now();
    saveState();
    render();
}

async function deleteConversation() {
    const conversation = getActiveConversation();
    if (!conversation || conversation.inFlight) {
        return;
    }

    if (!window.confirm(`确定删除“${conversation.title}”吗？`)) {
        return;
    }

    try {
        await releaseSession(conversation.sessionId);
    } catch (error) {
        console.warn("释放会话失败", error);
    }

    state.conversations = state.conversations.filter((item) => item.id !== conversation.id);
    if (state.conversations.length === 0) {
        state.conversations = [createConversation()];
    }
    state.activeConversationId = state.conversations[0].id;
    state.editorMessageId = null;
    saveState();
    render();
}

function installEventHandlers() {
    elements.newChatButton.addEventListener("click", createNewConversation);
    elements.renameChatButton.addEventListener("click", renameConversation);
    elements.deleteChatButton.addEventListener("click", async () => {
        await deleteConversation();
    });

    elements.composerForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        await submitPrompt(elements.composerInput.value);
    });

    elements.composerInput.addEventListener("keydown", async (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            await submitPrompt(elements.composerInput.value);
        }
    });

    window.addEventListener("storage", (event) => {
        if (event.key !== STORAGE_KEY) {
            return;
        }
        loadState();
        render();
    });
}

loadState();
installEventHandlers();
render();
