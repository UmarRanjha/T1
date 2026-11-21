"""A local chat interface for Ollama, built with Streamlit."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

import ollama
import streamlit as st

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
HISTORY_PANEL_HEIGHT = 320
TITLE_LENGTH = 44

st.set_page_config(
    page_title="Ollama chat",
    page_icon="💬",
    layout="wide",
)


class OllamaChatError(RuntimeError):
    """Raised when a streaming chat request to Ollama fails."""


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_client(host: str) -> ollama.Client:
    """Return a shared Ollama client (no read timeout, so long replies work)."""
    return ollama.Client(host=host)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_models(host: str) -> list[str]:
    """Return the names of the models installed on the Ollama server."""
    client = ollama.Client(host=host, timeout=5.0)
    return sorted(model.model for model in client.list().models)


def stream_chat(model: str, messages: list[dict[str, str]]) -> Iterator[str]:
    """Yield response tokens from Ollama one chunk at a time."""
    try:
        for chunk in get_client(OLLAMA_HOST).chat(
            model=model,
            messages=messages,
            stream=True,
        ):
            content = chunk.message.content
            if content:
                yield content
    except Exception as exc:  # noqa: BLE001 - surfaced to the user below
        if isinstance(exc, ollama.ResponseError):
            detail = exc.error
        else:
            detail = f"{type(exc).__name__}: {exc}"
        raise OllamaChatError(f"Could not complete the response. {detail}") from exc


# ---------------------------------------------------------------------------
# Session state and chat helpers
# ---------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("chats", {})
    st.session_state.setdefault("active_chat_id", None)
    st.session_state.setdefault("available_models", [])


def active_chat() -> dict[str, Any] | None:
    """Return the chat currently shown in the main area, if any."""
    return st.session_state.chats.get(st.session_state.active_chat_id)


def create_chat(model: str | None) -> str:
    """Create an empty chat and make it the active one."""
    chat_id = uuid.uuid4().hex[:8]
    now = time.time()
    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": "New chat",
        "model": model,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    st.session_state.active_chat_id = chat_id
    return chat_id


def make_title(text: str, limit: int = TITLE_LENGTH) -> str:
    title = " ".join(text.split())
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip() + "…"


def time_ago(timestamp: float) -> str:
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    days = hours // 24
    return f"{days} d ago"


# ---------------------------------------------------------------------------
# Sidebar callbacks
# ---------------------------------------------------------------------------


def prune_empty_chats(keep: str | None = None) -> None:
    """Drop empty conversations so they don't pile up in the history panel."""
    for chat_id, chat in list(st.session_state.chats.items()):
        if chat_id != keep and not chat["messages"]:
            del st.session_state.chats[chat_id]


def start_new_chat() -> None:
    current = active_chat()
    if current is not None and not current["messages"]:
        return  # Already sitting on an empty chat; don't create clutter.

    models = st.session_state.available_models
    model = st.session_state.get("model_select") or (models[0] if models else None)
    prune_empty_chats()
    create_chat(model)


def reset_active_chat() -> None:
    chat = active_chat()
    if chat is None:
        return
    chat["messages"] = []
    chat["title"] = "New chat"
    chat["updated_at"] = time.time()


def switch_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    prune_empty_chats(keep=chat_id)
    chat = st.session_state.chats.get(chat_id)
    models = st.session_state.available_models
    if chat and chat.get("model") in models:
        st.session_state.model_select = chat["model"]
    elif models:
        st.session_state.model_select = models[0]


def sync_chat_model() -> None:
    chat = active_chat()
    if chat is not None:
        chat["model"] = st.session_state.model_select


def refresh_models() -> None:
    fetch_models.clear()


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

init_state()

try:
    models = fetch_models(OLLAMA_HOST)
    model_error: str | None = None
except Exception as exc:  # noqa: BLE001 - shown in the UI
    models = []
    model_error = f"{type(exc).__name__}: {exc}"

st.session_state.available_models = models

if models and st.session_state.active_chat_id not in st.session_state.chats:
    create_chat(models[0])

with st.sidebar:
    st.title(":material/forum: Ollama chat")
    st.caption(f"Server: `{OLLAMA_HOST}`")

    if models:
        if st.session_state.get("model_select") not in models:
            st.session_state.model_select = models[0]

        st.selectbox(
            "Model",
            models,
            key="model_select",
            on_change=sync_chat_model,
            help="Models installed on your local Ollama server.",
        )
        st.button(
            "Refresh models",
            icon=":material/refresh:",
            help="Reload the list of installed models",
            width="stretch",
            on_click=refresh_models,
        )
    elif model_error:
        st.error(
            "Ollama isn't reachable. Start it, then retry.",
            icon=":material/cloud_off:",
        )
        st.button(
            "Retry connection",
            icon=":material/refresh:",
            width="stretch",
            on_click=refresh_models,
        )
    else:
        st.warning(
            "No models installed. Pull one, then refresh.",
            icon=":material/download:",
        )
        st.button(
            "Refresh models",
            icon=":material/refresh:",
            width="stretch",
            on_click=refresh_models,
        )

    current_chat = active_chat()
    new_col, reset_col = st.columns(2)
    with new_col:
        st.button(
            "New chat",
            icon=":material/add_comment:",
            width="stretch",
            on_click=start_new_chat,
            disabled=not models,
        )
    with reset_col:
        st.button(
            "Reset chat",
            icon=":material/restart_alt:",
            width="stretch",
            on_click=reset_active_chat,
            disabled=current_chat is None or not current_chat["messages"],
        )

    st.divider()
    st.markdown("**Conversation history**")

    chats = sorted(
        st.session_state.chats.values(),
        key=lambda chat: chat["updated_at"],
        reverse=True,
    )
    if not chats:
        st.caption("No saved chats yet.")
    else:
        with st.container(height=HISTORY_PANEL_HEIGHT, border=True):
            for saved in chats:
                is_active = saved["id"] == st.session_state.active_chat_id
                st.button(
                    saved["title"],
                    key=f"history_{saved['id']}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                    on_click=switch_chat,
                    args=(saved["id"],),
                )
                count = len(saved["messages"])
                st.caption(
                    f"{count} message{'s' if count != 1 else ''} · "
                    f"{time_ago(saved['updated_at'])}"
                )

    st.caption("History is kept for this browser session.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if model_error:
    st.title("Ollama chat")
    st.error(
        f"Could not connect to Ollama at `{OLLAMA_HOST}`. "
        "Make sure the server is running, then retry the connection.",
        title="Ollama is not running",
        icon=":material/cloud_off:",
    )
    st.markdown("Start Ollama and pull a model if you haven't already:")
    st.code("ollama serve\nollama pull llama3.2", language="bash")
    with st.expander("Technical details"):
        st.code(model_error)
    st.stop()

if not models:
    st.title("Ollama chat")
    st.warning(
        "Ollama is running, but no models are installed.",
        title="No models found",
        icon=":material/download:",
    )
    st.markdown("Pull a model, then refresh the list in the sidebar:")
    st.code("ollama pull llama3.2", language="bash")
    st.stop()

chat = active_chat()
if chat is None:
    create_chat(models[0])
    chat = active_chat()

selected_model = st.session_state.model_select
chat["model"] = selected_model

st.title(chat["title"])
st.caption(f"**{selected_model}** · running locally on `{OLLAMA_HOST}`")

if not chat["messages"]:
    st.info(
        f"Ask **{selected_model}** anything. Replies stream in as they are generated.",
        icon=":material/chat:",
    )

for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(f"Message {selected_model}…", submit_mode="stop")

if prompt:
    chat["model"] = selected_model
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "New chat":
        chat["title"] = make_title(prompt)
    chat["updated_at"] = time.time()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                stream_chat(selected_model, chat["messages"])
            )
        except OllamaChatError as exc:
            st.error(str(exc), title="Response failed", icon=":material/error:")
            response = ""

    if response:
        chat["messages"].append({"role": "assistant", "content": response})
        chat["updated_at"] = time.time()
        st.rerun()
