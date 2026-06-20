import json
import time
import subprocess
import tempfile
import torch
import cv2
import numpy as np
import gradio as gr

from pathlib import Path
from config import cfg
from tools.tools import  load_cached_class_graph
from utils import load_missiongnn

cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"

# Define color palette for the UI (forensic / surveillance aesthetic)
BG = "#0B0E11"
PANEL = "#12161B"
PANEL_ALT = "#0E1115"
BORDER = "#22272E"
TEXT = "#E8EAED"
SUBTEXT = "#8B95A1"
MUTED = "#5B6168"
RED = "#FF3B30"
AMBER = "#FFB020"
GREEN = "#00D26A"


# 1. Dataset paths + UI dropdown options
DATASET_ROOT = Path("UCFCrimeDataset")
EMBEDDINGS_DIR = DATASET_ROOT / "Embeddings"
VIDEOS_DIR = DATASET_ROOT / "Anomaly-Videos"
pt_files = list(EMBEDDINGS_DIR.rglob("*.pt"))
file_choices = [str(p.relative_to(EMBEDDINGS_DIR)) for p in pt_files]


# 2. Loading GNN Model + cached class graphs
print("Loading MissionGNN Model...")
gnn_model = load_missiongnn(Path("checkpoints/ImageBind_SlowFast_UCF-Crime.pt"), cfg.device)
graphs = {cls_name: load_cached_class_graph(cls_name) for cls_name in cfg.classes}
labels = ["Normal", *cfg.classes]
normal_idx = labels.index("Normal")


# 3. HTML helper renderers (status log + stat cards)
def render_idle_status():
    return f"""
    <div class="log-block">
        <span class="log-line"><span class="log-key">STATUS&nbsp;&nbsp;</span>AWAITING CASE SELECTION...</span>
        <span class="log-line"><span class="log-key">MODEL&nbsp;&nbsp;&nbsp;</span>MissionGNN ({cfg.device.upper()})</span>
        <span class="log-line"><span class="log-key">CLASSES&nbsp;</span>{len(cfg.classes)} anomaly types + Normal</span>
    </div>
    """


def render_error_status(message):
    return f"""
    <div class="log-block" style="border-left-color:{RED};">
        <span class="log-line" style="color:{RED};">ERROR &nbsp;{message}</span>
    </div>
    """


def render_status_html(video_name, total_frames, fps, exec_time, severity, sev_color):
    return f"""
    <div class="log-block" style="border-left-color:{sev_color};">
        <span class="log-line"><span class="log-key">STATUS&nbsp;&nbsp;</span>ANALYSIS COMPLETE</span>
        <span class="log-line"><span class="log-key">CASE&nbsp;&nbsp;&nbsp;&nbsp;</span>{video_name}</span>
        <span class="log-line"><span class="log-key">FRAMES&nbsp;&nbsp;</span>{total_frames} @ {fps:.1f} fps</span>
        <span class="log-line"><span class="log-key">RUNTIME&nbsp;</span>{exec_time:.2f}s</span>
        <span class="log-line" style="margin-top:6px;">
            <span class="severity-badge" style="background:{sev_color}22; color:{sev_color}; border:1px solid {sev_color}55;">
                {severity} SEVERITY
            </span>
        </span>
    </div>
    """


def render_empty_stats():
    return "<div id='stats-row'></div>"


def render_idle_timeline():
    return f"""
    <div class="timeline-wrap timeline-idle">
        <div class="log-block" style="border-left-color:{BORDER};">
            <span class="log-line" style="color:{MUTED};">Run an analysis to plot the anomaly timeline.</span>
        </div>
    </div>
    """


def build_timeline_html(time_axis, scores, pred_labels, duration, peak_idx, threshold=0.5):
    """SVG anomaly-score timeline. The hit-area + playhead are wired up to the
    video player in JS (see js_sync_timeline) so the chart and video stay in sync
    in both directions: click/drag the chart to seek the video, and the playhead
    follows the video as it plays."""
    W, H = 880, 220
    PAD_L, PAD_R, PAD_T, PAD_B = 44, 14, 16, 28
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    y_max = 1.05
    duration = duration if duration > 0 else 1.0

    def xs(t):
        return PAD_L + (t / duration) * plot_w

    def ys(score):
        return PAD_T + (1 - score / y_max) * plot_h

    pts = [(xs(t), ys(s)) for t, s in zip(time_axis, scores)]
    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area_d = path_d + f" L {pts[-1][0]:.1f},{PAD_T + plot_h:.1f} L {pts[0][0]:.1f},{PAD_T + plot_h:.1f} Z"

    threshold_y = ys(threshold)
    peak_x, peak_y = pts[peak_idx]

    label_times = [0, duration / 2, duration]
    time_labels = "".join(
        f'<text x="{xs(t):.1f}" y="{H - 8}" font-size="10" fill="{SUBTEXT}" '
        f'text-anchor="middle" font-family="JetBrains Mono, monospace">{t:.0f}s</text>'
        for t in label_times
    )

    json_data = json.dumps({
        "time": [round(float(t), 3) for t in time_axis],
        "score": [round(float(s), 4) for s in scores],
        "label": list(pred_labels),
    })

    return f"""
    <div class="timeline-wrap">
        <svg id="timeline-svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet"
             data-duration="{duration}" data-padl="{PAD_L}" data-plotw="{plot_w}">
            <line x1="{PAD_L}" y1="{threshold_y:.1f}" x2="{W - PAD_R}" y2="{threshold_y:.1f}"
                  stroke="{AMBER}" stroke-width="1" stroke-dasharray="4,4" opacity="0.7" />
            <path d="{area_d}" fill="{RED}" opacity="0.15" stroke="none" />
            <path d="{path_d}" fill="none" stroke="{RED}" stroke-width="1.8" />
            <circle cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="4" fill="{RED}" stroke="{BG}" stroke-width="1.5" />
            <line id="cursor-line" x1="0" y1="{PAD_T}" x2="0" y2="{PAD_T + plot_h}"
                  stroke="{SUBTEXT}" stroke-width="1" opacity="0" />
            <line id="playhead" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + plot_h}"
                  stroke="{TEXT}" stroke-width="1.5" opacity="0.9" />
            <rect id="timeline-hit" x="{PAD_L}" y="{PAD_T}" width="{plot_w}" height="{plot_h}"
                  fill="transparent" style="cursor:pointer;" />
            {time_labels}
            <text x="{PAD_L}" y="11" font-size="10" fill="{MUTED}"
                  font-family="JetBrains Mono, monospace">TEMPORAL ANOMALY SCORE &middot; click or drag to seek</text>
        </svg>
        <div id="tooltip-box" class="timeline-tooltip">
            <span id="tooltip-time"></span> &middot; <span id="tooltip-label"></span> &middot; <span id="tooltip-score"></span>
        </div>
        <script id="timeline-json" type="application/json">{json_data}</script>
    </div>
    """


def stat_card(label, value, sub=""):
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-sub">{sub}</div>
    </div>
    """


def render_stats_html(peak_score, peak_time, mean_score, anomalous_duration, dominant_class):
    return f"""
    <div id="stats-row">
        {stat_card("Peak Anomaly", f"{peak_score*100:.1f}%", f"at {peak_time:.1f}s")}
        {stat_card("Mean Anomaly Score", f"{mean_score*100:.1f}%", "across full clip")}
        {stat_card("Anomalous Duration", f"{anomalous_duration:.1f}s", "non-Normal frames")}
        {stat_card("Dominant Class", dominant_class, "most frequent anomaly")}
    </div>
    """


# 4. Core function: Run prediction + render video overlay + timeline + stats
def predict_with_overlay(selected_rel_path, progress=gr.Progress()):
    if not selected_rel_path:
        return None, render_idle_timeline(), render_error_status("Please select an embedding file."), render_empty_stats()

    start_time = time.time()
    pt_path = EMBEDDINGS_DIR / selected_rel_path
    mp4_rel_path = selected_rel_path.replace(".pt", ".mp4")
    video_path = VIDEOS_DIR / mp4_rel_path

    if not video_path.exists():
        return None, render_idle_timeline(), render_error_status(f"Video file not found: {video_path}"), render_empty_stats()

    progress(0.1, desc="Loading features...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    features = data.get("features", data) if isinstance(data, dict) else data
    if features.dim() == 1:
        features = features.unsqueeze(0)
    T = features.size(0)

    progress(0.2, desc="Running temporal inference...")
    predictions = []
    anomaly_scores = []
    with torch.no_grad():
        for i in range(T):
            start = max(0, i - 29)
            window = features[start:i + 1]

            if window.size(0) < 30:
                pad = torch.zeros(30 - window.size(0), window.size(1))
                window = torch.cat([pad, window], dim=0)

            mask = torch.zeros(30)
            mask[-window.size(0):] = 1
            window_sensors = window.unsqueeze(0).to(cfg.device)
            window_mask = mask.unsqueeze(0).to(cfg.device)
            logits = gnn_model(window_sensors, window_mask)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

            predictions.append(probs)
            anomaly_scores.append(1.0 - probs[normal_idx])

    progress(0.4, desc="Rendering video overlay...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    pred_labels_per_frame = []
    for frame_idx in range(total_frames):
        if frame_idx % 30 == 0:
            progress(0.4 + 0.4 * (frame_idx / total_frames), desc=f"Rendering frame {frame_idx}/{total_frames}")
        ret, frame = cap.read()
        if not ret:
            break

        feat_idx = min(int((frame_idx / total_frames) * T), T - 1)
        probs = predictions[feat_idx]
        pred_idx = probs.argmax()
        label = labels[pred_idx]
        conf = probs[pred_idx]
        pred_labels_per_frame.append(label)

        color = (98, 210, 0) if label == "Normal" else (48, 59, 255)  # BGR: green / red
        cv2.putText(frame, f"PREDICT: {label.upper()}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"CONF: {conf * 100:.1f}%", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 222, 232), 1, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

    progress(0.9, desc="Encoding for web...")
    final_out = tempfile.mktemp(suffix='.mp4')
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_out, "-vcodec", "libx264", "-f", "mp4", final_out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        output_video = final_out
    except Exception:
        output_video = temp_out

    progress(0.95, desc="Computing summary statistics...")
    anomaly_scores_arr = np.array(anomaly_scores)
    time_axis = np.linspace(0, total_frames / fps, T)

    peak_idx = int(anomaly_scores_arr.argmax())
    peak_score = float(anomaly_scores_arr[peak_idx])
    peak_time = float(time_axis[peak_idx])
    mean_score = float(anomaly_scores_arr.mean())

    feat_pred_labels = [labels[p.argmax()] for p in predictions]
    anomalous_labels = [l for l in feat_pred_labels if l != "Normal"]
    anomalous_duration = (len(anomalous_labels) / T) * (total_frames / fps)
    dominant_class = max(set(anomalous_labels), key=anomalous_labels.count) if anomalous_labels else "None detected"

    if peak_score >= 0.75:
        severity, sev_color = "HIGH", RED
    elif peak_score >= 0.4:
        severity, sev_color = "MEDIUM", AMBER
    else:
        severity, sev_color = "LOW", GREEN

    timeline_html = build_timeline_html(
        time_axis, anomaly_scores_arr, feat_pred_labels,
        duration=total_frames / fps, peak_idx=peak_idx,
    )

    exec_time = time.time() - start_time
    video_name = Path(selected_rel_path).stem

    status_html = render_status_html(video_name, total_frames, fps, exec_time, severity, sev_color)
    stats_html = render_stats_html(peak_score, peak_time, mean_score, anomalous_duration, dominant_class)

    return output_video, timeline_html, status_html, stats_html


# 5. Theme + CSS (forensic / surveillance-room aesthetic)
custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif !important;
}}

#header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 24px;
    background: linear-gradient(180deg, {PANEL} 0%, {PANEL_ALT} 100%);
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-bottom: 18px;
}}
#header-bar h1 {{
    font-size: 19px;
    letter-spacing: 0.4px;
    color: {TEXT};
    margin: 0;
    display: flex;
    align-items: center;
}}
#header-bar p {{
    color: {SUBTEXT};
    font-size: 12.5px;
    margin: 4px 0 0 18px;
    font-family: 'JetBrains Mono', monospace;
}}
.case-id {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: {MUTED};
    text-align: right;
    line-height: 1.5;
}}

.live-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: {RED};
    display: inline-block;
    margin-right: 9px;
    box-shadow: 0 0 0 0 rgba(255,59,48,0.6);
    animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(255,59,48,0.55); }}
    70% {{ box-shadow: 0 0 0 8px rgba(255,59,48,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255,59,48,0); }}
}}

#stats-row {{ display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }}
.stat-card {{
    flex: 1;
    min-width: 160px;
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 16px;
}}
.stat-card .stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
.stat-card .stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 23px;
    font-weight: 700;
    color: {TEXT};
    margin-top: 5px;
}}
.stat-card .stat-sub {{ font-size: 11px; color: {MUTED}; margin-top: 3px; }}

.severity-badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}}

.log-block {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-left: 3px solid {RED};
    border-radius: 6px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #C5CAD1;
    line-height: 1.7;
}}
.log-block .log-line {{ display: block; }}
.log-block .log-key {{ color: {MUTED}; }}

#run-btn {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
}}

.timeline-wrap {{
    position: relative;
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 8px 0 8px;
}}
.timeline-wrap svg {{ display: block; width: 100%; height: auto; }}
.timeline-idle {{ padding: 8px; }}
.timeline-tooltip {{
    position: absolute;
    top: 6px;
    transform: translateX(-50%);
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 9px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {TEXT};
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.1s ease;
}}
"""

force_dark_js = """
function () {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

js_sync_timeline = """
() => {
    const video = document.querySelector('#video-player video');
    const svg = document.getElementById('timeline-svg');
    const hit = document.getElementById('timeline-hit');
    if (!video || !svg || !hit) { return; }

    const playhead = document.getElementById('playhead');
    const cursorLine = document.getElementById('cursor-line');
    const tooltipBox = document.getElementById('tooltip-box');
    const tooltipTime = document.getElementById('tooltip-time');
    const tooltipLabel = document.getElementById('tooltip-label');
    const tooltipScore = document.getElementById('tooltip-score');
    const dataEl = document.getElementById('timeline-json');
    const tdata = dataEl ? JSON.parse(dataEl.textContent) : null;

    const duration = parseFloat(svg.dataset.duration) || 1;
    const padl = parseFloat(svg.dataset.padl);
    const plotw = parseFloat(svg.dataset.plotw);
    const viewW = svg.viewBox.baseVal.width;

    function nearestIndex(t) {
        if (!tdata) return 0;
        const arr = tdata.time;
        let lo = 0, hi = arr.length - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (arr[mid] < t) { lo = mid + 1; } else { hi = mid; }
        }
        return lo;
    }

    function clientXToTime(clientX) {
        const rect = svg.getBoundingClientRect();
        const svgX = ((clientX - rect.left) / rect.width) * viewW;
        const frac = Math.min(1, Math.max(0, (svgX - padl) / plotw));
        return frac * duration;
    }

    function setPlayheadByTime(t) {
        const frac = Math.min(1, Math.max(0, t / duration));
        const x = padl + frac * plotw;
        playhead.setAttribute('x1', x);
        playhead.setAttribute('x2', x);
    }

    let dragging = false;

    function showTooltip(clientX, t) {
        const idx = nearestIndex(t);
        const xFrac = Math.min(1, Math.max(0, t / duration));
        cursorLine.setAttribute('opacity', '0.6');
        cursorLine.setAttribute('x1', padl + xFrac * plotw);
        cursorLine.setAttribute('x2', padl + xFrac * plotw);
        if (tdata) {
            tooltipTime.textContent = t.toFixed(1) + 's';
            tooltipLabel.textContent = tdata.label[idx];
            tooltipScore.textContent = (tdata.score[idx] * 100).toFixed(0) + '%';
            tooltipBox.style.opacity = '1';
            const rect = hit.getBoundingClientRect();
            const relX = Math.min(rect.width - 4, Math.max(4, clientX - rect.left));
            tooltipBox.style.left = relX + 'px';
        }
    }

    hit.addEventListener('mousemove', (e) => {
        showTooltip(e.clientX, clientXToTime(e.clientX));
        if (dragging) { video.currentTime = clientXToTime(e.clientX); }
    });
    hit.addEventListener('mouseleave', () => {
        cursorLine.setAttribute('opacity', '0');
        tooltipBox.style.opacity = '0';
    });
    hit.addEventListener('mousedown', (e) => {
        dragging = true;
        video.currentTime = clientXToTime(e.clientX);
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    hit.addEventListener('click', (e) => {
        video.currentTime = clientXToTime(e.clientX);
    });

    video.addEventListener('timeupdate', () => { setPlayheadByTime(video.currentTime); });
    setPlayheadByTime(video.currentTime || 0);
}
"""

theme = gr.themes.Base(
    primary_hue="red",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=BG,
    body_background_fill_dark=BG,
    block_background_fill=PANEL,
    block_background_fill_dark=PANEL,
    block_border_color=BORDER,
    block_border_color_dark=BORDER,
    block_title_text_color=TEXT,
    block_label_text_color=SUBTEXT,
    body_text_color=TEXT,
    body_text_color_dark=TEXT,
    body_text_color_subdued=SUBTEXT,
    input_background_fill=PANEL_ALT,
    input_background_fill_dark=PANEL_ALT,
    border_color_primary=BORDER,
    border_color_primary_dark=BORDER,
    button_primary_background_fill=RED,
    button_primary_background_fill_hover="#E0332A",
    button_primary_text_color="#FFFFFF",
)


# 6. Build Gradio Interface
with gr.Blocks(theme=theme, css=custom_css, title="MissionGNN Surveillance Analysis", js=force_dark_js) as interface:

    gr.HTML(f"""
    <div id="header-bar">
        <div>
            <h1><span class="live-dot"></span>MISSIONGNN &nbsp;//&nbsp; TEMPORAL ANOMALY ANALYSIS
                <span style="font-family:'JetBrains Mono', monospace; font-weight:400; color:{SUBTEXT}; font-size:12px; margin-left:14px;">
                    frame-by-frame surveillance inference
                </span>
            </h1>
            <p>SlowFast + Adaptive MissionGNN &middot; UCF-Crime benchmark</p>
        </div>
        <div class="case-id">
            SYS::{cfg.device.upper()}<br>
            MODEL::IMAGEBIND-SLOWFAST-MISSIONGNN
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4, min_width=300):
            gr.Markdown("### 📁 Case Selector")
            file_dropdown = gr.Dropdown(choices=file_choices, label="UCF-Crime embedding file")
            analyze_btn = gr.Button("▶  RUN FRAME-BY-FRAME ANALYSIS", variant="primary", elem_id="run-btn")
            status_html = gr.HTML(render_idle_status())

        with gr.Column(scale=5):
            video_player = gr.Video(label="AI-analyzed footage", interactive=False, elem_id="video-player", height=420)

    stats_html = gr.HTML(render_empty_stats())

    timeline_html = gr.HTML(render_idle_timeline())

    analyze_btn.click(
        fn=predict_with_overlay,
        inputs=[file_dropdown],
        outputs=[video_player, timeline_html, status_html, stats_html]
    ).then(fn=None, inputs=None, outputs=None, js=js_sync_timeline)

if __name__ == "__main__":
    interface.launch()