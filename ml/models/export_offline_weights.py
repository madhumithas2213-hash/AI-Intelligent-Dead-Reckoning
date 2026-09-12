"""
Model Weight Exporter for On-Device Offline Browser Inference.
Extracts exact parameters from PyTorch checkpoint (best_gru_velocity.pt)
and StandardScaler (velocity_scaler.pkl) and saves them to a lightweight JSON file
and client-side JavaScript engine for 100% offline, on-device forward velocity regression.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from ml.training.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    MODELS_OUTPUT_DIR,
    INPUT_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    WINDOW_SIZE
)
from ml.models.gru_velocity_model import ForwardVelocityGRU


def export_offline_weights(
    model_path: Path = BEST_MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    output_json_path: Path = MODELS_OUTPUT_DIR / "offline_model_weights.json",
    output_js_path: Path = PROJECT_ROOT / "offline_ml_engine.js"
) -> Dict[str, Any]:
    """
    Extract weights and generate standalone offline JS inference engine.
    """
    print(f">> Exporting offline model weights from: {model_path}")
    print(f">> Scaler path: {scaler_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler pickle not found: {scaler_path}")

    # 1. Load Scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()

    # 2. Load PyTorch Model Checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # 3. Extract parameter arrays
    weights_dict = {
        "architecture": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "window_size": WINDOW_SIZE
        },
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale
        },
        "gru_l0": {
            "w_ih": state_dict["gru.weight_ih_l0"].detach().cpu().numpy().tolist(),
            "w_hh": state_dict["gru.weight_hh_l0"].detach().cpu().numpy().tolist(),
            "b_ih": state_dict["gru.bias_ih_l0"].detach().cpu().numpy().tolist(),
            "b_hh": state_dict["gru.bias_hh_l0"].detach().cpu().numpy().tolist()
        },
        "gru_l1": {
            "w_ih": state_dict["gru.weight_ih_l1"].detach().cpu().numpy().tolist(),
            "w_hh": state_dict["gru.weight_hh_l1"].detach().cpu().numpy().tolist(),
            "b_ih": state_dict["gru.bias_ih_l1"].detach().cpu().numpy().tolist(),
            "b_hh": state_dict["gru.bias_hh_l1"].detach().cpu().numpy().tolist()
        },
        "fc_head": {
            "fc0_w": state_dict["fc_head.0.weight"].detach().cpu().numpy().tolist(),
            "fc0_b": state_dict["fc_head.0.bias"].detach().cpu().numpy().tolist(),
            "fc3_w": state_dict["fc_head.3.weight"].detach().cpu().numpy().tolist(),
            "fc3_b": state_dict["fc_head.3.bias"].detach().cpu().numpy().tolist()
        }
    }

    # Save JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f)
    json_size_kb = output_json_path.stat().st_size / 1024.0
    print(f"[Export PASS] Offline weights saved to: {output_json_path} ({json_size_kb:.1f} KB)")

    # 4. Generate standalone offline JavaScript module with full vector math runtime
    js_content = f"""/**
 * Standalone On-Device Offline ML Inference Engine for AI-IDR Forward Velocity Regression.
 * 100% Client-Side Pure JavaScript Implementation of 2-Layer GRU Neural Network.
 * Zero external libraries, zero backend, zero network connection required.
 */
(function(window) {{
    'use strict';

    const MODEL_WEIGHTS = {json.dumps(weights_dict)};

    // Sigmoid Activation
    function sigmoid(x) {{
        return 1.0 / (1.0 + Math.exp(-Math.max(-45.0, Math.min(45.0, x))));
    }}

    // Tanh Activation
    function tanh(x) {{
        return Math.tanh(Math.max(-45.0, Math.min(45.0, x)));
    }}

    // Matrix-Vector Product: y = W * x
    function matVec(W, x, rows, cols) {{
        const y = new Float32Array(rows);
        for (let i = 0; i < rows; i++) {{
            let sum = 0.0;
            const row = W[i];
            for (let j = 0; j < cols; j++) {{
                sum += row[j] * x[j];
            }}
            y[i] = sum;
        }}
        return y;
    }}

    /**
     * Single GRU Layer Forward Pass over sequence.
     * PyTorch GRU gate ordering: [reset, update, new] (each of size hidden_dim).
     */
    function runGruLayer(inputSeq, w_ih, w_hh, b_ih, b_hh, hiddenDim, inputDim) {{
        const seqLen = inputSeq.length;
        let h = new Float32Array(hiddenDim); // Initial hidden state h0 = zeros

        for (let t = 0; t < seqLen; t++) {{
            const x = inputSeq[t];
            const gi = matVec(w_ih, x, 3 * hiddenDim, inputDim);
            const gh = matVec(w_hh, h, 3 * hiddenDim, hiddenDim);

            const nextH = new Float32Array(hiddenDim);
            for (let i = 0; i < hiddenDim; i++) {{
                const r_gate = sigmoid(gi[i] + b_ih[i] + gh[i] + b_hh[i]);
                const z_gate = sigmoid(gi[i + hiddenDim] + b_ih[i + hiddenDim] + gh[i + hiddenDim] + b_hh[i + hiddenDim]);
                const n_gate = tanh(gi[i + 2 * hiddenDim] + b_ih[i + 2 * hiddenDim] + r_gate * (gh[i + 2 * hiddenDim] + b_hh[i + 2 * hiddenDim]));
                nextH[i] = (1.0 - z_gate) * n_gate + z_gate * h[i];
            }}
            h = nextH;
        }}
        return h;
    }}

    /**
     * On-Device Velocity Predictor Class
     */
    class OfflineVelocityPredictor {{
        constructor() {{
            this.arch = MODEL_WEIGHTS.architecture;
            this.scaler = MODEL_WEIGHTS.scaler;
            this.l0 = MODEL_WEIGHTS.gru_l0;
            this.l1 = MODEL_WEIGHTS.gru_l1;
            this.fc = MODEL_WEIGHTS.fc_head;
            this.isReady = true;
        }}

        /**
         * Predict forward vehicle speed from 2D array [seq_len, 14] of unscaled sensor measurements.
         * Returns: {{ velocity_mps, velocity_kmh, latency_ms }}
         */
        predict(window2D) {{
            const t0 = performance.now();
            const seqLen = window2D.length;
            const inputDim = this.arch.input_dim;
            const hiddenDim = this.arch.hidden_dim;

            // 1. Standardize / Scale Inputs (Z-score normalization)
            const scaledSeq = [];
            for (let t = 0; t < seqLen; t++) {{
                const row = new Float32Array(inputDim);
                for (let f = 0; f < inputDim; f++) {{
                    const mean = this.scaler.mean[f];
                    const scale = this.scaler.scale[f] || 1.0;
                    row[f] = (window2D[t][f] - mean) / scale;
                }}
                scaledSeq.push(row);
            }}

            // 2. Layer 0 GRU: Input shape [seq_len, 14] -> Output h_l0 [hiddenDim = 64]
            // For multi-layer GRU, we need the full sequence output from Layer 0 as input to Layer 1
            let h0 = new Float32Array(hiddenDim);
            const l0Outputs = [];
            for (let t = 0; t < seqLen; t++) {{
                const x = scaledSeq[t];
                const gi = matVec(this.l0.w_ih, x, 3 * hiddenDim, inputDim);
                const gh = matVec(this.l0.w_hh, h0, 3 * hiddenDim, hiddenDim);

                const nextH = new Float32Array(hiddenDim);
                for (let i = 0; i < hiddenDim; i++) {{
                    const r = sigmoid(gi[i] + this.l0.b_ih[i] + gh[i] + this.l0.b_hh[i]);
                    const z = sigmoid(gi[i + hiddenDim] + this.l0.b_ih[i + hiddenDim] + gh[i + hiddenDim] + this.l0.b_hh[i + hiddenDim]);
                    const n = tanh(gi[i + 2 * hiddenDim] + this.l0.b_ih[i + 2 * hiddenDim] + r * (gh[i + 2 * hiddenDim] + this.l0.b_hh[i + 2 * hiddenDim]));
                    nextH[i] = (1.0 - z) * n + z * h0[i];
                }}
                h0 = nextH;
                l0Outputs.push(h0);
            }}

            // 3. Layer 1 GRU: Input shape [seq_len, 64] -> Final hidden state h1 [64]
            let h1 = new Float32Array(hiddenDim);
            for (let t = 0; t < seqLen; t++) {{
                const x = l0Outputs[t];
                const gi = matVec(this.l1.w_ih, x, 3 * hiddenDim, hiddenDim);
                const gh = matVec(this.l1.w_hh, h1, 3 * hiddenDim, hiddenDim);

                const nextH = new Float32Array(hiddenDim);
                for (let i = 0; i < hiddenDim; i++) {{
                    const r = sigmoid(gi[i] + this.l1.b_ih[i] + gh[i] + this.l1.b_hh[i]);
                    const z = sigmoid(gi[i + hiddenDim] + this.l1.b_ih[i + hiddenDim] + gh[i + hiddenDim] + this.l1.b_hh[i + hiddenDim]);
                    const n = tanh(gi[i + 2 * hiddenDim] + this.l1.b_ih[i + 2 * hiddenDim] + r * (gh[i + 2 * hiddenDim] + this.l1.b_hh[i + 2 * hiddenDim]));
                    nextH[i] = (1.0 - z) * n + z * h1[i];
                }}
                h1 = nextH;
            }}

            // 4. Fully Connected Head: Linear(64 -> 32) -> ReLU -> Linear(32 -> 1)
            // Dense 0 (32 units)
            const dense0 = new Float32Array(32);
            for (let i = 0; i < 32; i++) {{
                let sum = this.fc.fc0_b[i];
                const row = this.fc.fc0_w[i];
                for (let j = 0; j < 64; j++) {{
                    sum += row[j] * h1[j];
                }}
                dense0[i] = Math.max(0.0, sum); // ReLU
            }}

            // Dense 1 (1 unit -> predicted velocity m/s)
            let velPred = this.fc.fc3_b[0];
            const outRow = this.fc.fc3_w[0];
            for (let j = 0; j < 32; j++) {{
                velPred += outRow[j] * dense0[j];
            }}

            // PyTorch F.softplus activation matching model definition exactly
            const velMps = velPred > 20.0 ? velPred : Math.log1p(Math.exp(Math.max(-40.0, velPred)));
            const latencyMs = performance.now() - t0;

            return {{
                velocity_mps: velMps,
                velocity_kmh: velMps * 3.6,
                latency_ms: Math.round(latencyMs * 100) / 100
            }};
        }}
    }}

    // Attach to global window
    window.OfflineVelocityPredictor = OfflineVelocityPredictor;
    window.offlineVelocityPredictor = new OfflineVelocityPredictor();
    console.log("[OfflineMLEngine] 100% Client-Side On-Device GRU Velocity Regressor Ready. Parameters:", 42433);
}})(typeof window !== 'undefined' ? window : globalThis);
"""

    with open(output_js_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    js_size_kb = output_js_path.stat().st_size / 1024.0
    print(f"[Export PASS] Standalone Offline JS Engine generated: {output_js_path} ({js_size_kb:.1f} KB)")

    # Also copy into ml/outputs/fusion/ for direct server delivery
    dist_js_path = PROJECT_ROOT / "ml" / "outputs" / "fusion" / "offline_ml_engine.js"
    with open(dist_js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    return {
        "status": "SUCCESS",
        "json_path": str(output_json_path),
        "json_size_kb": json_size_kb,
        "js_path": str(output_js_path),
        "js_size_kb": js_size_kb,
        "total_parameters": 42433
    }


if __name__ == "__main__":
    export_offline_weights()
