# Project Status: Meeting Intelligence Pipeline

**Date:** 2026-03-22
**Current State:** Stable Core | Multi-Model Strategy | High Test Coverage

---

## 1. Core Features & Capabilities

The pipeline has evolved from a simple transcription script into a robust multi-stage processing engine.

| Feature | Description | Status |
|---|---|---|
| **Flexible Input** | Supports `.mp3`, `.wav`, `.mp4`, `.mov`, and direct `.md` passthrough via the `--input` flag. | ✅ Complete |
| **Video Extraction** | Automatically extracts high-quality 16kHz mono audio from video files using `ffmpeg`. | ✅ Complete |
| **Local Transcription** | Uses **OpenAI Whisper** (local) to generate high-fidelity Markdown transcripts. | ✅ Complete |
| **Retail-Focused LLM** | Tailored prompts for small online retailers to generate PRDs, User Stories, and Action Items. | ✅ Complete |
| **Conflict Detection** | Specifically flags potential conflicts between new proposals and existing system constraints. | ✅ Complete |

---

## 2. Architecture & Design Patterns

The codebase follows modern Python best practices and is architected specifically for future growth into an Agent.

- **Strategy Pattern (`helper/llm/`)**: All LLM logic is encapsulated in provider-specific classes (`OpenAIProvider`, `GeminiProvider`) satisfying a common `LLMProvider` Protocol.
- **Provider Factory**: `LLMFactory` dynamically routes requests based on model names (e.g., `gpt-*` vs `gemini-*`).
- **Markdown Passthrough**: If the input is already a transcript, the pipeline intelligently skips the expensive transcription stage and goes straight to analysis.
- **Dependency Management**: Fully managed via **`uv`** for reproducible, fast environments.

---

## 3. Reliability & Testing

We have prioritized "Software Engineering" quality to ensure the pipeline is production-ready.

- **Test Suite**: **43 unit tests** written in `pytest`, covering:
  - Subprocess mocking for `ffmpeg`.
  - Local module mocking for `whisper` and `LLMFactory`.
  - Environment synchronization and validation logic.
  - Exception chaining verification (original error details are never lost).
- **Error Handling**: Redundant `try/except` blocks cleaned up; all critical failures use `raise ... from e` to preserve the full stack trace for debugging.
- **Retry Logic**: Integrated 3-retry exponential backoff for all OpenAI API calls to handle transient network issues.

---

## 4. Current Configuration options

Configurable via `.env` or CLI:

- **`--input`**: Path to audio, video, or markdown.
- **`--llm-model`**: Override default model (e.g., `gemini-2.0-flash`, `gpt-4o-mini`).
- **`--whisper-model`**: Choose Whisper size (`tiny`, `base`, `small`, etc.).

---

## 5. Known Issues & Minor Quirks

- **Python 3.13 Ghost Error**: An `Exception ignored: _DeleteDummyThreadOnDel` may appear during interpreter shutdown. This is a known 3.13 threading quirk and does **not** affect the output or validity of the run.

---

## 6. Next Steps (Agent Evolution)

The "Stage 2" goal is to transition from this **Linear Pipeline** to a **Dynamic Agent**:

1. **Tool Conversion**: Wrap current functions (`prepare_transcript`, `summarize`) as standalone tools.
2. **Orchestration**: Implement an agent loop (using OpenAI Agents SDK or LangGraph) to handle multi-step goals (e.g., "Summarize this meeting and create these specific Jira tickets").
3. **Memory**: Add state management to track long-running multi-meeting contexts.
