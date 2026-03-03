const API_BASE = window.location.origin;

// ─── Status do Sistema ────────────────────────────────────────
async function checkSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        updateOllamaIndicator(data.ollama);
        updateDatabaseIndicator(data.database);
        return data;
    } catch (err) {
        console.error("Erro ao verificar status:", err);
        updateOllamaIndicator({ status: "offline" });
        updateDatabaseIndicator({ status: "erro" });
    }
}

function updateOllamaIndicator(ollama) {
    const el = document.getElementById("ollama-status");
    if (!el) return;
    if (ollama.status === "online") {
        el.innerHTML = `<span class="status-dot online"></span> Ollama: online`;
        el.className = "status-item online";
    } else {
        el.innerHTML = `<span class="status-dot offline"></span> Ollama: offline`;
        el.className = "status-item offline";
    }
}

function updateDatabaseIndicator(db) {
    const el = document.getElementById("db-status");
    if (!el) return;
    if (db.status === "online") {
        el.innerHTML = `<span class="status-dot online"></span> Base: ${db.documents_count || 0} docs`;
        el.className = "status-item online";
    } else {
        el.innerHTML = `<span class="status-dot offline"></span> Base: erro`;
        el.className = "status-item offline";
    }
}

// ─── Chat ─────────────────────────────────────────────────────
async function sendMessage(message, model = "llama3.2") {
    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                model: model,
                use_context: true
            })
        });
        const data = await res.json();
        return data;
    } catch (err) {
        console.error("Erro ao enviar mensagem:", err);
        return { response: "Erro de conexão com o servidor.", error: true };
    }
}

// ─── Upload/Treinar ───────────────────────────────────────────
async function uploadFile(file) {
    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            return { success: false, error: err.detail || "Erro no upload" };
        }

        const data = await res.json();
        return { success: true, ...data };
    } catch (err) {
        console.error("Erro no upload:", err);
        return { success: false, error: "Erro de conexão" };
    }
}

// ─── Documentos ───────────────────────────────────────────────
async function listDocuments() {
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        return await res.json();
    } catch (err) {
        return { documents: [], error: err.message };
    }
}

async function deleteDocument(docId) {
    try {
        const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: "DELETE"
        });
        return await res.json();
    } catch (err) {
        return { error: err.message };
    }
}

// ─── Modelos Ollama ───────────────────────────────────────────
async function getModels() {
    try {
        const res = await fetch(`${API_BASE}/api/ollama/models`);
        return await res.json();
    } catch (err) {
        return { models: [] };
    }
}

// ─── Inicialização ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // Verificar status a cada 10 segundos
    checkSystemStatus();
    setInterval(checkSystemStatus, 10000);
});