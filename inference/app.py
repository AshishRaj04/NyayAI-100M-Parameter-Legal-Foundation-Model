"""
NyayAI Flask Server - serves the frontend and handles inference
using the locally trained 103M parameter GPT model.

Run from the project root:
    python inference/app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request, jsonify
from inference.infer import NyayAIInference

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

# Load model once at startup
CHECKPOINT = os.environ.get(
    "NYAYAI_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "epoch_1_model_and_optimizer.pth"),
)
engine = None


def get_engine():
    global engine
    if engine is None:
        engine = NyayAIInference(checkpoint_path=CHECKPOINT)
    return engine


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def handle_generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400

        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 40)

        model = get_engine()
        generated_text = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return jsonify({"generated_text": generated_text, "prompt": prompt})

    except FileNotFoundError:
        return jsonify(
            {"error": "Model checkpoint not found. Train the model first!"}
        ), 500
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": engine is not None,
            "checkpoint": CHECKPOINT,
        }
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  NyayAI - Legal Intelligence Server")
    print("  103M Parameter GPT | Trained from Scratch")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("=" * 50)

    # Pre-load the model
    get_engine()

    print("\nOpen http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=False, port=5000)
