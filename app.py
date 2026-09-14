import io
import json
import os
import queue
import threading
import uuid
import zipfile

from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory

from multi_llm_work import MultiLLMWork

app = Flask(__name__)

JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# job_id -> {"queue": Queue, "output_dir": str, "files": list, "status": str}
JOBS = {}


def run_job(job_id: str, prompt: str, claude_key: str, openai_key: str, gemini_key: str):
    job = JOBS[job_id]
    q = job["queue"]

    def on_event(event: dict):
        q.put(event)

    try:
        team = MultiLLMWork(
            prompt=prompt,
            claude_key=claude_key or None,
            openai_key=openai_key or None,
            gemini_key=gemini_key or None,
            output_dir=job["output_dir"],
            on_event=on_event,
        )
        team.build()
    except Exception as e:
        on_event({"type": "error", "message": str(e)})
    finally:
        q.put({"type": "__stream_end__"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/build", methods=["POST"])
def api_build():
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Mission brief khali nahi ho sakti."}), 400

    claude_key = (data.get("claude_key") or "").strip()
    openai_key = (data.get("openai_key") or "").strip()
    gemini_key = (data.get("gemini_key") or "").strip()

    if not any([claude_key, openai_key, gemini_key]):
        return jsonify({"error": "Kam se kam ek provider ki access code do."}), 400

    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {
        "queue": queue.Queue(),
        "output_dir": os.path.join(JOBS_DIR, job_id),
        "files": [],
        "status": "running",
    }

    thread = threading.Thread(
        target=run_job,
        args=(job_id, prompt, claude_key, openai_key, gemini_key),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q = job["queue"]
        while True:
            event = q.get()
            if event.get("type") == "__stream_end__":
                break
            if event.get("type") == "file_written":
                job["files"].append(event["filename"])
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'stream_closed'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/download/<job_id>/<path:filename>")
def api_download(job_id, filename):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return send_from_directory(job["output_dir"], filename, as_attachment=True)


@app.route("/api/preview/<job_id>/<path:filename>")
def api_preview(job_id, filename):
    """Serves a generated file un-attached so the browser renders it live
    (used for a real running preview of index.html, styles, scripts, etc.)."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return send_from_directory(job["output_dir"], filename, as_attachment=False)


@app.route("/api/download-zip/<job_id>")
def api_download_zip(job_id):
    """Zips every generated file and streams it down -- lands straight in
    the browser's default Downloads folder, whole project in one file."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in job["files"]:
            file_path = os.path.join(job["output_dir"], filename)
            if os.path.isfile(file_path):
                zf.write(file_path, arcname=filename)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"multillmwork_{job_id}.zip",
    )


if __name__ == "__main__":
    # Render द्वारा दिए गए PORT को पढ़ें, अगर नहीं है तो डिफ़ॉल्ट 5000 रखें
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

