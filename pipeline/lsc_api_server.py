"""
lsc_api_server.py
Unified FastAPI backend & Web server for LSC Translation
"""
import sys
import os
import logging
import json
import mimetypes
import subprocess
import tempfile
import shutil

# FORCE application/javascript for .js files globally in this process
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# Add pipeline to path
sys.path.insert(0, os.path.dirname(__file__))
from nlp_translator import translate_to_glosses, load_vocabulary
from motion_synthesizer import synthesize_sequence, export_to_json, GLOSS_TO_SIGNID

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LSC Unified Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR   = os.path.normpath(os.path.join(PIPELINE_DIR, "..", "webapp"))
VIDEO_DIR    = "D:/LSC/LSC50/VIDEOS/COLOR_BODY"
FFMPEG_BIN   = shutil.which("ffmpeg") or "ffmpeg"

# In-memory cache for converted video paths
VIDEO_CACHE  = {}  # gloss → mp4 temp path
VIDEO_STATUS = {}  # gloss → 'ready' | 'converting' | 'error'
VIDEO_LOCK   = __import__('threading').Lock()

def _convert_one(gloss, sign_id):
    """Convert a single AVI to MP4 and cache it."""
    avi_path = os.path.join(VIDEO_DIR, f"{sign_id}_0000_0000.avi")
    if not os.path.exists(avi_path):
        with VIDEO_LOCK: VIDEO_STATUS[gloss] = 'error'
        return
    tmp_dir  = tempfile.mkdtemp(prefix="lsc_video_")
    mp4_path = os.path.join(tmp_dir, f"{gloss}.mp4")
    cmd = [FFMPEG_BIN, "-y", "-i", avi_path, "-c:v", "libx264",
           "-preset", "ultrafast", "-crf", "23",
           "-movflags", "+faststart", "-an", mp4_path]
    try:
        with VIDEO_LOCK: VIDEO_STATUS[gloss] = 'converting'
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0:
            with VIDEO_LOCK:
                VIDEO_CACHE[gloss]  = mp4_path
                VIDEO_STATUS[gloss] = 'ready'
            logging.info(f"✅ Video ready: {gloss}")
        else:
            with VIDEO_LOCK: VIDEO_STATUS[gloss] = 'error'
    except Exception as e:
        logging.warning(f"⚠️ Video error {gloss}: {e}")
        with VIDEO_LOCK: VIDEO_STATUS[gloss] = 'error'

def _prewarm_all_videos():
    """Background thread: convert all 50 videos sequentially."""
    logging.info("🚀 Pre-warming all LSC50 videos in background...")
    for gloss, sign_id in GLOSS_TO_SIGNID.items():
        with VIDEO_LOCK:
            if gloss in VIDEO_CACHE:  # already done
                VIDEO_STATUS[gloss] = 'ready'
                continue
        _convert_one(gloss, sign_id)
    logging.info("✅ All videos pre-warmed.")

# Kick off pre-warm at module load (non-blocking)
import threading
_prewarm_thread = threading.Thread(target=_prewarm_all_videos, daemon=True)
_prewarm_thread.start()

class TranslationRequest(BaseModel):
    text: str

@app.get("/status")
def status():
    return {"status": "online", "service": "LSC Unified Server"}

@app.get("/vocabulary")
def get_vocabulary():
    from motion_synthesizer import GLOSS_TO_SIGNID, MONTH_CONFIG
    import string
    
    # 1. Base Words (LSC50 organic extractions)
    base_words = list(GLOSS_TO_SIGNID.keys())
    
    # 2. Procedural Months (Strip 'MES_' prefix for UI display)
    month_words = [m.replace('MES_', '') for m in MONTH_CONFIG.keys()]
    
    # 3. Letters/Fingerspelling capability
    letters = list(string.ascii_uppercase)
    
    # Sort and merge unique elements
    all_words = sorted(list(set(base_words + month_words + letters)))
    
    return {"vocabulary": all_words, "count": len(all_words)}

@app.get("/api/video_list")
def api_video_list():
    """Return all glosses that have an associated video file."""
    available = {}
    for gloss, sign_id in GLOSS_TO_SIGNID.items():
        avi_path = os.path.join(VIDEO_DIR, f"{sign_id}_0000_0000.avi")
        available[gloss] = os.path.exists(avi_path)
    return {"videos": available}

@app.get("/api/video_status")
def api_video_status():
    """Return per-gloss conversion status: ready | converting | error | pending."""
    with VIDEO_LOCK:
        result = {}
        for gloss in GLOSS_TO_SIGNID:
            result[gloss] = VIDEO_STATUS.get(gloss, 'pending')
    return {"status": result, "ready_count": sum(1 for v in result.values() if v == 'ready')}

@app.get("/api/video/{gloss}")
def api_get_video(gloss: str):
    """Serve LSC50 video for a given gloss as MP4."""
    gloss_upper = gloss.upper()
    if gloss_upper not in GLOSS_TO_SIGNID:
        # Fallback para LSCPROPIO: buscar recursivamente cualquier mp4 que contenga el nombre de la glosa
        lsc_propio_dir = "D:/LSC/LSCPROPIO"
        search_pattern = os.path.join(lsc_propio_dir, "**", "*.mp4")
        import glob
        matches = glob.glob(search_pattern, recursive=True)
        clean_gloss = gloss_upper.replace('MES_', '')
        # Buscar coincidencia exacta en el nombre base (ignorando -personaX)
        for m in matches:
            filename = os.path.basename(m).upper()
            # Si el filename contiene la gloss seguida de un delimitador o extensiones (soportando m4v)
            if filename.startswith(clean_gloss + "-") or filename.startswith(clean_gloss + "_") or filename.startswith(clean_gloss + "PERSONA") or filename == clean_gloss + ".MP4" or filename == clean_gloss + ".M4V":
                return FileResponse(m, media_type="video/mp4", headers={"Cache-Control": "public, max-age=86400"})
                
        raise HTTPException(status_code=404, detail=f"Gloss '{gloss}' not in vocabulary and no video found in LSCPROPIO.")

    # Check background cache first
    with VIDEO_LOCK:
        cached_path = VIDEO_CACHE.get(gloss_upper)
        state = VIDEO_STATUS.get(gloss_upper, 'pending')

    if cached_path and os.path.exists(cached_path):
        return FileResponse(cached_path, media_type="video/mp4",
                            headers={"Cache-Control": "public, max-age=86400"})

    if state == 'converting':
        # Background thread is already working on it — return 202 so client retries
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "converting"}, status_code=202)

    # Not yet queued or errored — convert now (on-demand fallback)
    sign_id  = GLOSS_TO_SIGNID[gloss_upper]
    avi_path = os.path.join(VIDEO_DIR, f"{sign_id}_0000_0000.avi")
    if not os.path.exists(avi_path):
        raise HTTPException(status_code=404, detail=f"AVI not found for '{gloss}'.")

    tmp_dir  = tempfile.mkdtemp(prefix="lsc_video_")
    mp4_path = os.path.join(tmp_dir, f"{gloss_upper}.mp4")
    cmd = [FFMPEG_BIN, "-y", "-i", avi_path, "-c:v", "libx264",
           "-preset", "ultrafast", "-crf", "23",
           "-movflags", "+faststart", "-an", mp4_path]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="FFmpeg failed.")
        with VIDEO_LOCK:
            VIDEO_CACHE[gloss_upper]  = mp4_path
            VIDEO_STATUS[gloss_upper] = 'ready'
        return FileResponse(mp4_path, media_type="video/mp4",
                            headers={"Cache-Control": "public, max-age=86400"})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Conversion timed out.")


@app.post("/translate")
def translate(request: TranslationRequest):
    try:
        vocab = load_vocabulary()
        glosses = translate_to_glosses(request.text, vocab)
        if not glosses:
            raise HTTPException(status_code=422, detail="No se encontraron señas.")
        
        motion_data = synthesize_sequence(glosses)
        export_to_json(motion_data, "lsc_translation_live.json")
        
        return {
            "input_text": request.text,
            "glosses": glosses,
            "frames": motion_data.get("frames", []),
            "total_frames": motion_data.get("total_frames", 0)
        }
    except Exception as e:
        logging.exception("Error in translation")
        raise HTTPException(status_code=500, detail=str(e))

# --- Frontend Routes ---

@app.get("/")
def read_index():
    # Return translator.html by default
    fpath = os.path.join(WEBAPP_DIR, "translator.html")
    if os.path.exists(fpath):
        return FileResponse(fpath)
    return HTMLResponse("<h1>translator.html not found in webapp folder</h1>", status_code=404)

@app.get("/visualizer")
def read_visualizer():
    fpath = os.path.join(WEBAPP_DIR, "visualizer.html")
    if os.path.exists(fpath):
        return FileResponse(fpath)
    return HTMLResponse("<h1>visualizer.html not found</h1>", status_code=404)

@app.get("/gallery")
def read_gallery():
    fpath = os.path.join(WEBAPP_DIR, "gallery.html")
    if os.path.exists(fpath):
        return FileResponse(fpath)
    return HTMLResponse("<h1>gallery.html not found</h1>", status_code=404)

@app.get("/translator")
def read_translator():
    fpath = os.path.join(WEBAPP_DIR, "translator.html")
    if os.path.exists(fpath):
        return FileResponse(fpath)
    return HTMLResponse("<h1>translator.html not found</h1>", status_code=404)

# Mount the webapp directory for all other assets (JS, CSS, etc.)
# FastAPI will use the initialized mimetypes module
if os.path.exists(WEBAPP_DIR):
    app.mount("/", StaticFiles(directory=WEBAPP_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Servidor LSC Unificado iniciado en puerto 8000")
    print(f"👉 Frontend: http://localhost:8000/")
    print(f"📁 WebApp Dir: {WEBAPP_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
