# Ollama chat

A small Streamlit chat app that talks to a local [Ollama](https://ollama.com)
server. Responses stream in token by token, and each conversation keeps its
full multi-turn history.

## Features

- Dynamically lists installed Ollama models in a sidebar dropdown.
- Streams replies with `ollama.chat(..., stream=True)` and `st.write_stream()`.
- Sends the whole conversation back to the model on every turn.
- **New chat** starts a fresh conversation; **Reset chat** clears the active one.
- **Conversation history** panel lets you switch between saved chats.
- Clear in-app guidance when Ollama is offline or has no models installed.

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and running

## Setup

1. Start the Ollama server (the desktop app usually starts it for you):

   ```bash
   ollama serve
   ```

2. Pull at least one model:

   ```bash
   ollama pull llama3.2
   ```

3. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open <http://localhost:8501>, pick a model in the sidebar, and start chatting.

## Configuration

The app connects to `http://localhost:11434` by default. To use a different
Ollama address, set `OLLAMA_HOST` before starting Streamlit:

```bash
OLLAMA_HOST=http://192.168.1.50:11434 streamlit run app.py
```

## Notes

- Conversations live in Streamlit session state, so history is kept per
  browser tab and is cleared when that tab or the server restarts.
- If the model list changes, use the refresh button next to the model selector.
- Chat history is sent to the selected model as context, so very long
  conversations may eventually exceed the model's context window.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Ollama is not running" | Start it with `ollama serve` and reload the page. |
| "No models found" | Run `ollama pull <model>`, then refresh the model list. |
| Model missing from the dropdown | Pull it with Ollama and click the refresh button. |
| Slow first reply | The model is being loaded into memory; later replies are faster. |
