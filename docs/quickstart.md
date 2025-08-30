### Quickstart

#### Prerequisites
- **Python**: 3.10+
- **Shell**: Windows PowerShell 7
- **API keys** (as needed):
  - `GOOGLE_API_KEY` (Gemini 1.5 Flash)
  - `COHERE_API_KEY` (only if enabling reranker)
  - Optional: Neo4j instance (local or remote)

#### 1) Setup
```powershell
# From your project root
cd C:\Development\FinalProjectBGU\FinalProjectBGU

# Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create uploads dir used by the app
mkdir data\uploads -Force
```

#### 2) Run the app
```powershell
# Inside the venv, from project root
python -m src.ui.gradio_app
```
Open the Gradio URL printed in the terminal.

#### 3) Use the app
- Q&A tab:
  - Set your Google key or export it before launching:
    ```powershell
    $env:GOOGLE_API_KEY = "your_key_here"
    ```
  - Upload a 10‑Q PDF (≤ ~50 pages), click “Process File”.
  - Ask a question; the system routes to the appropriate tool (`general_tool`, `table_tool`, `mda_tool`, `risk_tool`).
  - Optional reranker: toggle “Use Reranker” and provide `COHERE_API_KEY`.
    ```powershell
    $env:COHERE_API_KEY = "your_key_here"
    ```
- Add to Graph:
  - Provide `Neo4j URI`, `User`, `Password`, then click “Add to Graph”.
- Evaluation tab:
  - Enter question and ground truth, click “Evaluate” for RAGAS metrics.

#### 4) Run tests
```powershell
pytest -q
```

#### 5) Notes
- Model: uses `gemini-1.5-flash` via `langchain_google_genai`.
- Parsing: preserves SEC semantic elements; chunking emits LangChain `Document`s with metadata (`element_type`, plus `page_number`/`section_path`/`content_type` when available).
- Retrieval: Chroma (dense) + TF‑IDF with effective financial term boosting; combined via `EnsembleRetriever` weights from `src/config.py`.
- Tools: `SimpleTool` invokes LLM; `TableTool` uses a LlamaIndex program for tabular answers.
- Reranker: requires a valid `COHERE_API_KEY` when enabled.
- If activation is blocked:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```


