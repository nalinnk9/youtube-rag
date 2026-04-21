# YouTube RAG

A local retrieval-augmented-generation pipeline over any YouTube playlist. Ask questions, get answers with **timestamped citations** that deep-link back to the exact moment in the video.

## What it does

1. **Ingest** — point it at a YouTube playlist (or single video) URL. For each video it:
   - Tries YouTube's auto-captions first
   - Falls back to Whisper ASR (via `faster-whisper`) if captions are disabled
   - Splits the transcript into ~1000-character chunks that preserve start/end timestamps
   - Embeds each chunk with OpenAI `text-embedding-3-small`
   - Stores everything in a persistent Chroma vector DB
2. **Query** — you ask a question in the web UI. The pipeline:
   - Embeds the question, retrieves the top-15 chunks from Chroma
   - (Optional) Reranks with Cohere down to the top 4 most relevant
   - Sends those chunks as context to Claude (or GPT) with strict citation instructions
   - Returns the answer with `[1]`, `[2]` markers that are clickable
3. **Watch** — clicking any citation seeks the embedded YouTube player to the exact second the answer came from.

## Project layout

```
youtube-rag/
├── backend/
│   ├── main.py             # FastAPI app (POST /ask, POST /ingest, GET /stats, GET /videos)
│   ├── config.py           # Env-driven settings
│   ├── ingest.py           # Orchestrates ingestion
│   ├── playlist.py         # yt-dlp playlist enumeration
│   └── pipeline/
│       ├── transcripts.py  # Captions + Whisper fallback
│       ├── chunking.py     # Time-aware sliding-window chunker
│       ├── vectorstore.py  # Chroma + OpenAI embeddings
│       ├── retrieval.py    # Vector search + Cohere rerank
│       └── generation.py   # Claude / GPT answer generation with citation parsing
├── scripts/
│   └── ingest_playlist.py  # CLI: python -m scripts.ingest_playlist <url>
├── frontend/
│   └── index.html          # Single-file React UI (CDN-loaded, no build step)
├── requirements.txt
├── .env.example
├── run.sh
└── README.md
```

## Setup

### 1. Clone / unzip and enter the folder

```bash
cd youtube-rag
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> First run of the Whisper fallback downloads the model (~500 MB for `small.en`). That's only needed if some videos lack captions.

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

- `OPENAI_API_KEY` — used for embeddings (always)
- `ANTHROPIC_API_KEY` — for the default Claude-based answer generation

Optional:

- `COHERE_API_KEY` — enables reranking for noticeably sharper citations
- Switch `LLM_PROVIDER=openai` + set `LLM_MODEL=gpt-4o-mini` if you prefer GPT

### 4. Run

```bash
# Linux / macOS
bash run.sh

# or directly
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Usage

### Option A — from the UI

Paste a playlist or video URL into the **Ingest** box at the top. Ingestion runs in the background; watch the terminal for per-video progress. Once it's done, refresh the counter and ask away.

### Option B — from the CLI (recommended for large playlists)

```bash
python -m scripts.ingest_playlist "https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"
```

You'll see per-video progress in the terminal. The script is idempotent — rerunning it will skip videos already in the DB.

### Asking questions

Type a question in the UI and press **Ask**. The answer appears with inline `[1]`, `[2]` markers — click any marker (or a source card below) to jump the embedded player to that timestamp.

## Configuration reference

All settings live in `.env`:

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Embeddings, and the LLM if `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | *(required for Claude)* | |
| `COHERE_API_KEY` | *(optional)* | Enables rerank; empty = skip rerank |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | `claude-sonnet-4-6` | e.g. `claude-opus-4-7`, `gpt-4o-mini`, `gpt-4o` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `WHISPER_MODEL` | `small.en` | `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3` |
| `CHUNK_TARGET_CHARS` | `1000` | ~45–60 sec of speech per chunk |
| `CHUNK_OVERLAP_CHARS` | `150` | Overlap between adjacent chunks |
| `TOP_K_RETRIEVE` | `15` | How many chunks to pull from the vector DB |
| `TOP_K_RERANK` | `4` | How many chunks survive reranking (sent to LLM) |
| `CHROMA_PATH` | `./chroma_db` | Where the persistent vector DB lives |
| `TRANSCRIPT_LANGUAGES` | `en` | Comma-separated language priority — e.g. `hi,en` for Hindi-then-English |

## Typical costs (for <100 videos)

- **Embeddings** (one-time): ~$0.01 for a ~100-video library
- **Per query**: ~$0.01 with Claude Sonnet, ~$0.002 with Cohere rerank, ~$0.00001 for the query embedding
- **Whisper fallback**: free (runs locally)

## Troubleshooting

**`youtube-transcript-api` errors** — YouTube occasionally rate-limits. Wait a minute and try again. The script auto-retries 3× per video.

**`yt-dlp` can't find a video** — YouTube changes things frequently. Upgrade:
```bash
pip install -U yt-dlp
```

**Whisper fails to load** — `faster-whisper` needs `ctranslate2`. On Linux/macOS it should install cleanly from pip. On Windows, if you hit issues, try `pip install faster-whisper --no-binary ctranslate2`.

**A video's transcript is clearly wrong** — that usually means the auto-captions are low quality. Either disable them for that video, or force Whisper by setting `TRANSCRIPT_LANGUAGES` to a language the video doesn't have (so captions lookup fails and Whisper runs). A cleaner approach for future iterations would be to add a "force whisper" flag.

**The player doesn't seek** — first click loads the player; subsequent clicks seek. If nothing happens, open the browser console and look for YouTube API errors (usually an ad-blocker or network issue).

**I want to wipe the index and start over** — stop the server and `rm -rf chroma_db/`.

## How to extend

- **Visual/frame understanding** ("show me the slide where…") — sample frames with `ffmpeg`, embed with CLIP, keep them in a second collection, and add a multimodal retrieval path.
- **Streaming responses** — swap `/ask` for a streaming endpoint using `anthropic.Anthropic().messages.stream(...)` and an SSE response in FastAPI.
- **Per-playlist namespaces** — pass a `namespace` into the ingest flow, store as a metadata field, and filter on it in `retrieve()` with Chroma's `where` clause.
- **Upgrade to semantic chunking** — after getting captions, run a small model to split on topic boundaries instead of fixed character counts.

## License

MIT — do whatever you want with it.
