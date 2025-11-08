from flask import Flask, render_template, request, redirect, url_for, flash
import requests
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

FASTAPI_BASE = "http://127.0.0.1:8000"  # our existing FastAPI service
MAX_MB = 10  # must match backend, so user gets same expectation

app = Flask(__name__)
app.secret_key = "dev-secret"  # needed for flash() messages


# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """
    Show main UI:
    - upload a file (POST /upload_file here -> forwards to FastAPI /upload)
    - ask a question (POST /ask_question here -> forwards to FastAPI /ask)
    """
    return render_template("home.html")


@app.route("/upload_file", methods=["POST"])
def upload_file():
    """
    User picks a file in the browser, submits form.
    We send that file to FastAPI /upload and show success or error.
    """
    file = request.files.get("file")

    if not file or file.filename == "":
        flash("Please choose a file before uploading.", "error")
        return redirect(url_for("home"))

    # size guard on the Flask side too (not just backend)
    file.seek(0, os.SEEK_END)
    size_bytes = file.tell()
    file.seek(0)

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_MB:
        flash(f"File too large ({size_mb:.2f} MB). Max is {MAX_MB} MB.", "error")
        return redirect(url_for("home"))

    # forward to FastAPI
    try:
        resp = requests.post(
            f"{FASTAPI_BASE}/upload",
            files={"file": (file.filename, file.stream, file.mimetype)},
            timeout=60,
        )
    except Exception as e:
        flash(f"Upload failed: could not reach backend ({e})", "error")
        return redirect(url_for("home"))

    if resp.status_code != 200:
        # backend rejected
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        flash(f"Upload failed: {detail}", "error")
        return redirect(url_for("home"))

    data = resp.json()
    saved_as = data.get("saved_as")
    flash(f"Uploaded and ingested: {saved_as}", "success")
    return redirect(url_for("home"))


@app.route("/ask_question", methods=["POST"])
def ask_question():
    """
    User asks a question from the browser.
    We forward that question to FastAPI /ask.
    Then we render the answer and citations on a nice page.
    """
    user_q = request.form.get("question", "").strip()
    use_web = request.form.get("use_web") == "on"
    reasoner = request.form.get("reasoner", "gemini")
    source_hint = request.form.get("source_hint", "").strip()

    if len(user_q) < 3:
        flash("Question must be at least 3 characters.", "error")
        return redirect(url_for("home"))

    payload = {
        "q": user_q,
        "source": source_hint if source_hint else None,
        "k": 20,
        "overfetch": 50,
        "use_web": use_web,
        "reasoner": reasoner,
    }

    try:
        resp = requests.post(
            f"{FASTAPI_BASE}/ask",
            json=payload,
            timeout=120,
        )
    except Exception as e:
        flash(f"Ask failed: could not reach backend ({e})", "error")
        return redirect(url_for("home"))

    # Two main cases:
    # 200 = success (we got an answer)
    # 404 = "no relevant context found"
    # anything else = backend/server issue
    if resp.status_code == 200:
        data = resp.json()
        return render_template("answer.html", result=data)

    elif resp.status_code == 404:
        # no context found, friendly message
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        flash(f"No grounded answer: {detail}", "warning")
        return redirect(url_for("home"))

    else:
        # unexpected error
        flash(f"Error from backend ({resp.status_code}): {resp.text}", "error")
        return redirect(url_for("home"))


if __name__ == "__main__":
    # dev run:
    # python frontend_app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
