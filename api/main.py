"""
Medical Lab Report Analyzer — FastAPI Server
=============================================

HOW TO RUN:
    1. Open terminal in your project root (where requirements.txt is)
    2. Install dependencies:
           pip install -r requirements.txt
    3. Start server:
           python run.py
       OR directly:
           uvicorn api.main:app --reload --port 8000

HOW TO USE:
    Open browser → http://localhost:8000
    → Upload any lab report PDF or image
    → Get instant analysis

API DOCS (auto-generated):
    http://localhost:8000/docs

ENDPOINTS:
    GET  /              → upload UI (browser form)
    GET  /health        → health check
    POST /analyze/file  → upload file, get JSON analysis
"""

import logging
import os
import sys

# ── make sure project root is on path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from agents.lab_crew import LabReportCrew

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lab Report Analyzer",
    description="Upload a lab report PDF or image and get a structured medical analysis.",
    version="3.0.0",
)

# One crew instance reused for all requests
_use_llm = bool(
    os.getenv("AZURE_OPENAI_API_KEY")   # standard Azure SDK name
    or os.getenv("AZURE_OPENAI_KEY")    # legacy fallback
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
)
crew = LabReportCrew(use_llm=_use_llm)
logger.info(f"LabReportCrew ready  |  LLM enabled: {_use_llm}")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".txt"}


# ── response model ────────────────────────────────────────────────────────────
class AnalysisResponse(BaseModel):
    patient_info: dict
    test_results: list
    analysis_result: Optional[dict] = None   # best available (azure or rule-based)
    azure_analysis: Optional[dict] = None    # full Azure OpenAI structured analysis
    patient_report: Optional[str] = None     # rule-based markdown report
    llm_explanation: Optional[str] = None    # patient-friendly explanation from Azure
    validation_notes: list = []
    errors: list = []


# ── browser upload UI ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def upload_ui():
    """Simple browser UI so you can test without Postman/curl."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lab Report Analyzer</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; color: #222; }
    h1   { font-size: 1.6rem; margin-bottom: 4px; }
    p    { color: #555; margin-top: 4px; }
    .box { border: 2px dashed #aaa; border-radius: 10px; padding: 30px; text-align: center;
           cursor: pointer; transition: border-color .2s; margin: 24px 0; }
    .box:hover { border-color: #555; }
    input[type=file] { display: none; }
    label.btn, button { background: #1a1a1a; color: #fff; border: none; padding: 10px 24px;
                        border-radius: 6px; font-size: 1rem; cursor: pointer; }
    label.btn:hover, button:hover { background: #333; }
    #status { margin-top: 16px; font-size: .9rem; color: #555; }
    #result { margin-top: 24px; }
    .section { background: #f6f6f6; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .section h3 { margin: 0 0 8px; font-size: 1rem; }
    .badge { display:inline-block; padding:2px 10px; border-radius:99px; font-size:.8rem; font-weight:600; margin:2px; }
    .high  { background:#fde8e8; color:#991b1b; }
    .normal{ background:#e6f4ea; color:#166534; }
    .warn  { background:#fef9c3; color:#854d0e; }
    pre    { white-space: pre-wrap; font-size: .82rem; background: #f0f0f0; padding: 12px; border-radius: 6px; max-height: 400px; overflow-y: auto; }
  </style>
</head>
<body>
  <h1>🔬 Lab Report Analyzer</h1>
  <p>Upload any lab report — PDF, image (JPG/PNG), or scanned document — and get instant analysis.</p>

  <div class="box" onclick="document.getElementById('fileInput').click()">
    <p style="font-size:2rem;margin:0">📄</p>
    <p>Click to select file, or drag & drop here</p>
    <p style="font-size:.85rem;color:#888">Supported: PDF, PNG, JPG, JPEG, TIFF, BMP, TXT</p>
    <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.txt">
  </div>

  <button onclick="analyze()">Analyze Report</button>
  <div id="status"></div>
  <div id="result"></div>

  <script>
    // drag & drop
    const box = document.querySelector('.box');
    box.addEventListener('dragover', e => { e.preventDefault(); box.style.borderColor='#333'; });
    box.addEventListener('dragleave', () => box.style.borderColor='#aaa');
    box.addEventListener('drop', e => {
      e.preventDefault();
      box.style.borderColor='#aaa';
      document.getElementById('fileInput').files = e.dataTransfer.files;
      document.querySelector('.box p:nth-child(2)').textContent = e.dataTransfer.files[0].name;
    });
    document.getElementById('fileInput').addEventListener('change', e => {
      if(e.target.files[0]) document.querySelector('.box p:nth-child(2)').textContent = e.target.files[0].name;
    });

    async function analyze() {
      const fileInput = document.getElementById('fileInput');
      if (!fileInput.files.length) { alert('Please select a file first.'); return; }

      const status = document.getElementById('status');
      const result = document.getElementById('result');
      status.textContent = '⏳ Analyzing... this may take 15-30 seconds for scanned PDFs.';
      result.innerHTML = '';

      const form = new FormData();
      form.append('file', fileInput.files[0]);

      try {
        const resp = await fetch('/analyze/file', { method: 'POST', body: form });
        console.log('Response status:', resp.status);
        
        let data;
        try {
          data = await resp.json();
        } catch (e) {
          console.error('Failed to parse JSON:', e, resp);
          status.textContent = '';
          result.innerHTML = `<div class="section" style="background:#fde8e8"><h3>Error</h3><p>Server returned invalid response. Check browser console.</p></div>`;
          return;
        }
        
        console.log('Response data:', data);
        status.textContent = '';

        if (!resp.ok) {
          result.innerHTML = `<div class="section" style="background:#fde8e8"><h3>Error (${resp.status})</h3><p>${data.detail || 'Unknown error'}</p></div>`;
          return;
        }
        renderResult(data);
      } catch(e) {
        console.error('Request error:', e);
        status.textContent = '';
        result.innerHTML = `<div class="section" style="background:#fde8e8"><h3>Error</h3><p>${e.message}</p></div>`;
      }
    }

    function renderResult(data) {
      const result = document.getElementById('result');
      let html = '';

      // Patient info
      const pi = data.patient_info || {};
      html += `<div class="section">
        <h3>👤 Patient</h3>
        <b>${pi.name || 'Unknown'}</b> &nbsp;|&nbsp; Age: ${pi.age || 'N/A'} &nbsp;|&nbsp; Gender: ${pi.gender || 'N/A'}
      </div>`;

      // Validation notes
      if(data.validation_notes?.length) {
        html += `<div class="section"><h3>⚠️ Notes</h3>${data.validation_notes.map(n=>`<p>${n}</p>`).join('')}</div>`;
      }

      // Azure OpenAI Analysis — show first if available
      if(data.azure_analysis) {
        const az = data.azure_analysis;
        html += `<div class="section" style="border-left:3px solid #2563eb">
          <h3>🤖 Azure OpenAI Clinical Analysis</h3>`;
        if(az.patient_summary) html += `<p><b>Summary:</b> ${az.patient_summary}</p>`;
        if(az.patient_friendly_explanation) {
          html += `<div style="background:#f0f7ff;padding:12px;border-radius:6px;margin:8px 0">
            <b>For the patient:</b><br>${az.patient_friendly_explanation.replace(/\n/g,'<br>')}
          </div>`;
        }
        if(az.key_findings?.length) {
          html += `<details open><summary><b>Key Findings (${az.key_findings.length})</b></summary>`;
          az.key_findings.forEach(f => {
            const cls = f.status==='High'||f.status==='Low'?'high':f.status==='Borderline'?'warn':'normal';
            html += `<div style="margin:6px 0">
              <span class="badge ${cls}">${f.status}</span> <b>${f.test}</b>: ${f.value}
              ${f.urgency&&f.urgency!=='Routine'?`<span class="badge warn">${f.urgency}</span>`:''}
              <br><span style="font-size:.85rem;color:#555">${f.clinical_meaning}</span></div>`;
          });
          html += `</details>`;
        }
        if(az.detected_conditions?.length) {
          html += `<details><summary><b>Detected Conditions</b></summary>`;
          az.detected_conditions.forEach(c => {
            html += `<div style="margin:6px 0"><span class="badge warn">${c.severity}</span> <b>${c.condition}</b>
              <span style="color:#888;font-size:.85rem"> — ${c.confidence} confidence</span><br>
              <span style="font-size:.85rem">${c.evidence}</span></div>`;
          });
          html += `</details>`;
        }
        if(az.risk_areas?.length) {
          html += `<details><summary><b>Risk Areas</b></summary>`;
          az.risk_areas.forEach(r => {
            const cls = r.risk_level==='High'?'high':r.risk_level==='Moderate'?'warn':'normal';
            html += `<div style="margin:4px 0"><span class="badge ${cls}">${r.risk_level}</span> <b>${r.area}</b>: ${r.reason}</div>`;
          });
          html += `</details>`;
        }
        if(az.recommendations) {
          const recs = az.recommendations;
          html += `<details><summary><b>Recommendations</b></summary>`;
          if(recs.immediate?.length) html += `<p><b>Immediate:</b></p><ul>${recs.immediate.map(r=>`<li>${r}</li>`).join('')}</ul>`;
          if(recs.short_term?.length) html += `<p><b>Short-term:</b></p><ul>${recs.short_term.map(r=>`<li>${r}</li>`).join('')}</ul>`;
          if(recs.lifestyle?.length) html += `<p><b>Lifestyle:</b></p><ul>${recs.lifestyle.map(r=>`<li>${r}</li>`).join('')}</ul>`;
          if(recs.tests_to_repeat?.length) html += `<p><b>Tests to repeat:</b></p><ul>${recs.tests_to_repeat.map(r=>`<li>${r}</li>`).join('')}</ul>`;
          html += `</details>`;
        }
        if(az.positive_findings?.length) {
          html += `<details><summary><b>✅ Positive Findings</b></summary>
            <ul>${az.positive_findings.map(f=>`<li>${f}</li>`).join('')}</ul></details>`;
        }
        html += `</div>`;
      }

      // Rule-based summary (always show as secondary reference)
      const ar = data.analysis_result || {};
      if(ar.summary && !data.azure_analysis) {
        html += `<div class="section"><h3>📋 Summary</h3><p>${ar.summary}</p></div>`;
      }
      if(data.llm_explanation && !data.azure_analysis) {
        html += `<div class="section"><h3>💬 Explanation</h3><pre>${data.llm_explanation}</pre></div>`;
      }

      // Abnormal tests
      const abnormal = ar.abnormal_tests || [];
      if(abnormal.length) {
        html += `<div class="section"><h3>🔴 Abnormal Tests (${abnormal.length})</h3>`;
        abnormal.forEach(t => {
          const cls = t.severity === 'Severe' || t.severity === 'Critical' ? 'high' : 'warn';
          html += `<div style="margin:6px 0">
            <span class="badge ${cls}">${t.severity}</span>
            <b>${t.test_name}</b>: ${t.value} ${t.unit}
            <span style="color:#888;font-size:.85rem"> (ref: ${t.reference_range})</span><br>
            <span style="font-size:.85rem;color:#555">${t.interpretation}</span>
          </div>`;
        });
        html += '</div>';
      }

      // Detected conditions
      const conditions = ar.detected_conditions || [];
      if(conditions.length) {
        html += `<div class="section"><h3>🏥 Detected Conditions</h3>`;
        conditions.forEach(c => {
          html += `<div style="margin:8px 0"><span class="badge warn">${c.severity}</span> <b>${c.name}</b><br>
          <span style="font-size:.85rem">${c.description}</span></div>`;
        });
        html += '</div>';
      }

      // Follow-up
      const fu = ar.follow_up_needed || {};
      if(fu.priority) {
        html += `<div class="section"><h3>📅 Follow-up</h3>
          <p><b>Priority:</b> ${fu.priority}</p>
          ${fu.specialists_needed?.length ? `<p><b>Specialists:</b> ${fu.specialists_needed.join(', ')}</p>` : ''}
          ${fu.tests_to_repeat?.length ? `<p><b>Tests to repeat:</b> ${fu.tests_to_repeat.join(', ')}</p>` : ''}
        </div>`;
      }

      // Full report markdown
      if(data.patient_report) {
        html += `<div class="section"><h3>📄 Full Report</h3><pre>${data.patient_report}</pre></div>`;
      }

      // Raw errors
      if(data.errors?.length) {
        html += `<div class="section" style="background:#fde8e8"><h3>Errors</h3>${data.errors.map(e=>`<p>${e}</p>`).join('')}</div>`;
      }

      result.innerHTML = html;
    }
  </script>
</body>
</html>
"""


# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "llm_enabled": _use_llm}


# ── main endpoint: upload file ────────────────────────────────────────────────
@app.post(
    "/analyze/file",
    response_model=AnalysisResponse,
    summary="Upload a lab report file",
    description="Upload PDF or image. Returns structured analysis with detected conditions, abnormal tests, recommendations.",
)
async def analyze_file(file: UploadFile = File(...)):
    # Validate extension
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    file_bytes = await file.read()
    logger.info(f"Received: {file.filename}  ({len(file_bytes):,} bytes)")

    try:
        result = crew.run(file_bytes=file_bytes, file_ext=ext)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return AnalysisResponse(
        patient_info=result.get("patient_info", {}),
        test_results=result.get("test_results", []),
        analysis_result=result.get("analysis_result"),
        azure_analysis=result.get("azure_analysis"),
        patient_report=result.get("patient_report"),
        llm_explanation=result.get("llm_explanation"),
        validation_notes=result.get("validation_notes", []),
        errors=result.get("errors", []),
    )