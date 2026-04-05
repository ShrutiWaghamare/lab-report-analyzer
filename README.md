# Lab Report Analyzer

Upload any lab report (PDF, scanned image, phone photo) and get:
- All abnormal tests flagged with severity
- Detected medical conditions
- Risk assessments
- Recommendations and follow-up plan
- Plain language explanation (if LLM configured)

---

## How to run

### Step 1 — Install system dependencies
```bash
# Ubuntu / WSL
sudo apt-get install tesseract-ocr poppler-utils

# Windows (use WSL, or install manually)
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
```

### Step 2 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 3 — Start the server
```bash
python run.py
```

### Step 4 — Open browser
```
http://localhost:8000
```
Upload your report → click Analyze → done.

---

## Project structure

```
lab_analyzer/
├── run.py                  ← START HERE — just run this
├── requirements.txt
│
├── api/
│   └── main.py             ← FastAPI app + browser upload UI
│
├── agents/
│   └── lab_crew.py         ← Multi-agent pipeline (OCR → Parse → Analyze → Explain)
│
├── ocr/
│   ├── extractor.py        ← Handles PDF / image / text input
│   ├── opencv_preprocessor.py  ← Deskew, shadow removal, denoise, table detection
│   └── parser.py           ← Converts raw OCR text → structured JSON
│
├── core/
│   ├── medical_analyzer.py ← Main orchestrator
│   ├── test_evaluator.py   ← Flags abnormal values
│   ├── condition_analyzer.py
│   ├── risk_assessor.py
│   ├── recommendation_engine.py
│   ├── summary_generator.py
│   └── emergency_checker.py
│
└── samples/
    └── input/              ← Put your sample reports here to test
        ├── Sarvodhya/
        ├── Thyrocare/
        ├── Redcliffe/
        └── ...
```

---

## Optional: LLM for plain-language explanation

Set any one of these environment variables before running:

```bash
# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

# OR standard OpenAI
export OPENAI_API_KEY="sk-..."

# OR Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

Without any key set, the system still works fully — just without the plain-language rewrite step.

---

## Testing with sample reports

Put any lab report PDF in `samples/input/` and either:
- Upload via browser at http://localhost:8000
- Or use curl:
  ```bash
  curl -X POST http://localhost:8000/analyze/file \
       -F "file=@samples/input/Sarvodhya/report.pdf"
  ```
