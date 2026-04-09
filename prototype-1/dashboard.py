import json
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE = ROOT / 'prototype-1'
ARTIFACTS = PROTOTYPE / 'artifacts'
EXPERIMENTS = PROTOTYPE / 'experiments'
MEMORY = ROOT / 'MEMORY.md'
HEARTBEAT = ROOT / 'HEARTBEAT.md'
COORDINATION = ARTIFACTS / 'coordination_status.json'
THRESHOLD_RESULTS = ARTIFACTS / 'threshold_sweep_results.json'
PORT = 8765


def read_text(path: Path, default: str = '') -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return default


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def status_badge(status: str) -> str:
    status = str(status or 'unknown').lower()
    classes = {
        'completed': 'ok',
        'in_progress': 'live',
        'paused': 'paused',
        'pending': 'pending',
        'blocked': 'bad',
    }
    cls = classes.get(status, 'pending')
    label = status.replace('_', ' ')
    return f'<span class="badge {cls}">{escape(label)}</span>'


def latest_artifacts():
    items = []
    for path in sorted(ARTIFACTS.glob('*')):
        if not path.is_file() or path.name == COORDINATION.name:
            continue
        stat = path.stat()
        payload = read_json(path) if path.suffix == '.json' else None
        items.append({
            'name': path.name,
            'path': str(path.relative_to(ROOT)),
            'modified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'data': payload or {},
            'suffix': path.suffix.lower(),
        })
    items.sort(key=lambda x: x['modified'], reverse=True)
    return items


def recent_memory_blocks(limit: int = 5):
    text = read_text(MEMORY)
    blocks = [b.strip() for b in text.split('\n## ') if b.strip()]
    filtered = []
    for block in blocks:
        if block.startswith('Format') or block.startswith('Persistent Context'):
            continue
        filtered.append(block if block.startswith('202') else '## ' + block)
    return filtered[-limit:][::-1]


def categorize_files():
    scripts, reports, metrics, others = [], [], [], []
    for path in sorted(PROTOTYPE.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        name = path.name.lower()
        if path.suffix == '.py':
            scripts.append(rel)
        elif path.suffix in {'.md', '.txt'}:
            reports.append(rel)
        elif path.suffix in {'.json', '.csv', '.npz'}:
            metrics.append(rel)
        else:
            others.append(rel)
    return scripts, reports, metrics, others


def milestone_progress(milestones):
    if not milestones:
        return 0
    score = 0.0
    for item in milestones:
        status = item.get('status')
        if status == 'completed':
            score += 1.0
        elif status == 'in_progress':
            score += 0.5
    return round((score / len(milestones)) * 100)


def best_known_result():
    data = read_json(THRESHOLD_RESULTS) or {}
    best = data.get('best_quality_constrained') or {}
    if not best:
        return None
    return {
        'threshold': best.get('threshold'),
        'gain_proxy': best.get('gain_proxy'),
        'rel_mse': best.get('rel_mse'),
        'sparsity': best.get('sparsity'),
        'kept_fraction': best.get('kept_fraction'),
    }


def latest_artifact_table(item):
    if not item:
        return '<p class="muted">No artifact available yet.</p>'
    rows = ''.join(
        f'<tr><td>{escape(str(k))}</td><td><code>{escape(str(v))}</code></td></tr>'
        for k, v in item['data'].items()
    ) or '<tr><td colspan="2">No structured fields</td></tr>'
    return f"""
      <h3>Latest artifact snapshot</h3>
      <p><strong>{escape(item['name'])}</strong> <span class="muted">({escape(item['modified'])})</span></p>
      <table>{rows}</table>
    """


def render_html():
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    coordination = read_json(COORDINATION) or {}
    artifacts = latest_artifacts()
    memory_blocks = recent_memory_blocks()
    heartbeat = read_text(HEARTBEAT, '(missing)').strip() or '(empty)'
    scripts, reports, metrics, others = categorize_files()
    best = best_known_result()

    milestones = coordination.get('major_milestones', [])
    pending_tasks = coordination.get('pending_tasks', [])
    active_agents = coordination.get('active_subagents', [])
    completed_agents = coordination.get('completed_subagent_wave', [])
    next_agents = coordination.get('next_if_spawned', [])
    recent_pivots = coordination.get('recent_pivots', [])
    cleanup_backlog = coordination.get('cleanup_backlog', [])
    progress_pct = milestone_progress(milestones)
    phase_progress_pct = int(coordination.get('phase_progress_pct', 0) or 0)
    phase_note = coordination.get('phase_note', '')
    active_count = len(active_agents)
    execution_state = str(coordination.get('execution_state', '') or '').lower()
    if execution_state not in {'in_progress', 'paused', 'blocked', 'pending', 'completed'}:
        execution_state = 'in_progress' if active_count else 'paused'
    if execution_state == 'in_progress':
        live_label = f"{active_count} active sub-agent{'s' if active_count != 1 else ''}" if active_count else "Coordinator active"
    elif execution_state == 'paused':
        live_label = "Paused, no active workers"
    else:
        live_label = f"Execution state: {execution_state.replace('_', ' ')}"

    progress_bar = f"""
      <div class="progress-wrap">
        <div class="progress-bar"><div class="progress-fill shimmer" style="width:{progress_pct}%"></div></div>
        <div class="progress-label">Overall project progress: {progress_pct}%</div>
      </div>
    """

    phase_progress_bar = f"""
      <div class="progress-wrap">
        <div class="progress-bar"><div class="progress-fill shimmer" style="width:{phase_progress_pct}%"></div></div>
        <div class="progress-label">Current phase progress: {phase_progress_pct}%</div>
      </div>
    """

    executive_summary = "Software surrogate target achieved, hardware path not yet proven."
    hardware_status = "Do not buy hardware yet."
    if milestones and milestones[-1].get('status') == 'completed':
        hardware_status = "Hardware path validated. Purchase is now reasonable."

    best_card = "<div class='muted'>Best result not available yet.</div>"
    if best:
        best_card = f"""
        <div class="metric-number">{best['gain_proxy']:.2f}x</div>
        <div class="metric-sub">best efficiency gain proxy</div>
        <div class="metric-detail">threshold {best['threshold']}, rel MSE {best['rel_mse']:.4f}, sparsity {best['sparsity']:.2%}</div>
        """

    milestones_html = ''.join(
        f"<li class='timeline-item {('live-card' if m.get('status') == 'in_progress' else '')}'><div><strong>{escape(m.get('marker', '?'))} {escape(m.get('name', 'unknown'))}</strong> {status_badge(m.get('status'))}</div><div class='muted'>{escape(m.get('summary', ''))}</div></li>"
        for m in milestones
    ) or '<li>No milestones defined.</li>'

    pending_tasks_html = ''.join(f'<li>{escape(str(task))}</li>' for task in pending_tasks) or '<li>No pending tasks recorded.</li>'
    pivots_html = ''.join(f'<li>{escape(str(item))}</li>' for item in recent_pivots) or '<li>No recent pivots recorded.</li>'
    cleanup_html = ''.join(f'<li>{escape(str(item))}</li>' for item in cleanup_backlog) or '<li>No cleanup backlog recorded.</li>'

    active_agents_html = ''.join(
        f"<li class='agent-card live-card'><strong>{escape(agent.get('name', 'unknown'))}</strong> {status_badge(agent.get('status') or 'in_progress')}<br><span class='muted'>{escape(agent.get('summary', ''))}</span></li>"
        for agent in active_agents
    ) or '<li class="agent-card"><strong>No active sub-agents right now.</strong><br><span class="muted">The system is idle on sub-agent work, but the coordinator can still be progressing the main plan.</span></li>'

    completed_agents_html = ''.join(
        f"<li class='agent-card'><strong>{escape(agent.get('name', 'unknown'))}</strong> {status_badge(agent.get('status'))}<br><span class='muted'>{escape(agent.get('summary', ''))}</span></li>"
        for agent in completed_agents
    ) or '<li>No completed sub-agent work recorded yet.</li>'

    next_agents_html = ''.join(f'<li><code>{escape(str(name))}</code></li>' for name in next_agents) or '<li>None planned.</li>'

    artifact_list = ''.join(
        f"<li><strong>{escape(item['name'])}</strong> <span class='muted'>({escape(item['modified'])})</span></li>"
        for item in artifacts
    ) + f"<li><strong>{escape(COORDINATION.name)}</strong> <span class='muted'>(status / control plane)</span></li>"

    reports_html = ''.join(f'<li><code>{escape(name)}</code></li>' for name in reports) or '<li>No reports yet.</li>'
    scripts_html = ''.join(f'<li><code>{escape(name)}</code></li>' for name in scripts) or '<li>No scripts yet.</li>'
    metrics_html = ''.join(f'<li><code>{escape(name)}</code></li>' for name in metrics) or '<li>No metric artifacts yet.</li>'
    others_html = ''.join(f'<li><code>{escape(name)}</code></li>' for name in others) or '<li>None.</li>'

    memory_html = ''.join(f'<pre>{escape(block)}</pre>' for block in memory_blocks) or '<pre>No recent memory entries.</pre>'

    latest_snapshot_html = latest_artifact_table(artifacts[0] if artifacts else None)
    live_indicator = '<span class="live-dot"></span>' if execution_state == 'in_progress' else ''
    live_activity_html = f"""
      <div class=\"live-row\">{live_indicator}<strong>{escape(live_label)}</strong> {status_badge(execution_state)}</div>
      <div class=\"muted\" style=\"margin-top:8px;\">Main coordinator task: {escape(str(coordination.get('currently_doing', 'Unknown')))}</div>
      <div class=\"muted\" style=\"margin-top:8px;\">Status refreshes automatically every 5 seconds.</div>
      <div class=\"muted\" style=\"margin-top:8px;\">{escape(str(phase_note))}</div>
    """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>Prototype-1 Mission Dashboard</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #131a2a;
      --panel-2: #0f1726;
      --line: #26324a;
      --text: #e8ecf3;
      --muted: #94a3b8;
      --accent: #7dd3fc;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --live: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .page {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    .hero {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
    .hero h1 {{ margin: 0 0 8px 0; color: var(--accent); }}
    .subtle {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin: 18px 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); }}
    .pulse-border {{ position: relative; overflow: hidden; }}
    .pulse-border::after {{ content: ''; position: absolute; inset: 0; border-radius: 14px; border: 1px solid rgba(56,189,248,0.35); animation: borderPulse 2.2s ease-in-out infinite; pointer-events: none; }}
    .card h2, .card h3 {{ margin-top: 0; color: var(--accent); }}
    .metric-number {{ font-size: 30px; font-weight: 700; margin-bottom: 6px; }}
    .metric-sub {{ color: var(--muted); margin-bottom: 8px; }}
    .metric-detail {{ font-size: 14px; color: var(--text); }}
    .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .progress-wrap {{ margin-top: 10px; }}
    .progress-bar {{ width: 100%; height: 16px; background: #0a1220; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--live), var(--good), var(--live)); background-size: 200% 100%; }}
    .progress-label {{ margin-top: 8px; color: var(--muted); font-size: 14px; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }}
    .badge.ok {{ background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.35); }}
    .badge.live {{ background: rgba(56,189,248,0.15); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.35); animation: softPulse 1.6s ease-in-out infinite; }}
    .badge.paused {{ background: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }}
    .badge.pending {{ background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }}
    .badge.bad {{ background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }}
    .timeline, ul {{ margin: 0; padding-left: 20px; }}
    .timeline-item {{ margin-bottom: 12px; }}
    .agent-card {{ margin-bottom: 10px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(15,23,38,0.9); list-style: none; }}
    .live-card {{ box-shadow: 0 0 0 rgba(56,189,248,0.0); animation: cardPulse 1.8s ease-in-out infinite; }}
    .live-row {{ display: flex; align-items: center; gap: 10px; }}
    .live-dot {{ width: 10px; height: 10px; border-radius: 999px; background: var(--live); box-shadow: 0 0 10px rgba(56,189,248,0.7); animation: dotPulse 1.2s ease-in-out infinite; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    td {{ border-top: 1px solid var(--line); padding: 8px 6px; vertical-align: top; }}
    code, pre {{ background: var(--panel-2); color: #d8e8ff; border-radius: 8px; }}
    code {{ padding: 2px 6px; }}
    pre {{ padding: 12px; white-space: pre-wrap; overflow-x: auto; }}
    .section-title {{ margin-bottom: 10px; }}
    .shimmer {{ animation: shimmerMove 2.2s linear infinite; }}
    @keyframes dotPulse {{
      0%, 100% {{ transform: scale(0.85); opacity: 0.7; }}
      50% {{ transform: scale(1.25); opacity: 1; }}
    }}
    @keyframes softPulse {{
      0%, 100% {{ transform: translateY(0); opacity: 0.85; }}
      50% {{ transform: translateY(-1px); opacity: 1; }}
    }}
    @keyframes cardPulse {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(56,189,248,0.0); border-color: var(--line); }}
      50% {{ box-shadow: 0 0 0 1px rgba(56,189,248,0.35), 0 0 18px rgba(56,189,248,0.12); border-color: rgba(56,189,248,0.45); }}
    }}
    @keyframes borderPulse {{
      0%, 100% {{ opacity: 0.25; }}
      50% {{ opacity: 0.8; }}
    }}
    @keyframes shimmerMove {{
      0% {{ background-position: 200% 0; }}
      100% {{ background-position: -200% 0; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div>
        <h1>Prototype-1 Mission Dashboard</h1>
        <div class="subtle">Stakeholder view of progress from experiment to delivery.</div>
        <div class="subtle">Auto-refresh every 5 seconds. Rendered at {escape(now)}.</div>
      </div>
      <div class="card pulse-border" style="min-width:280px; max-width:420px;">
        <h3 class="section-title">Current mission state</h3>
        {live_activity_html}
        <div><strong>Phase:</strong> {escape(str(coordination.get('current_phase', 'Unknown')))}</div>
        <div style="margin-top:8px;"><strong>Currently doing:</strong> {escape(str(coordination.get('currently_doing', 'Unknown')))}</div>
        <div style="margin-top:8px;"><strong>Executive summary:</strong> {escape(executive_summary)}</div>
        <div style="margin-top:8px;"><strong>Hardware decision:</strong> {escape(hardware_status)}</div>
        <div class="subtle" style="margin-top:8px;">Status snapshot updated: {escape(str(coordination.get('last_updated', 'unknown')))}</div>
      </div>
    </div>

    <div class="cards">
      <div class="card pulse-border">
        <h3>Overall progress</h3>
        <div class="metric-number">{progress_pct}%</div>
        <div class="metric-sub">milestone completion</div>
        {progress_bar}
      </div>
      <div class="card pulse-border">
        <h3>Current phase progress</h3>
        <div class="metric-number">{phase_progress_pct}%</div>
        <div class="metric-sub">working progress inside the current milestone</div>
        {phase_progress_bar}
      </div>
      <div class="card">
        <h3>Best surrogate result</h3>
        {best_card}
      </div>
      <div class="card">
        <h3>Quality target</h3>
        <div class="metric-number">&lt; 1%</div>
        <div class="metric-sub">allowed relative quality loss</div>
        <div class="metric-detail">Current best is about 0.266% relative MSE in the surrogate.</div>
      </div>
      <div class="card">
        <h3>Purchase readiness</h3>
        <div class="metric-number">Not yet</div>
        <div class="metric-sub">hardware recommendation</div>
        <div class="metric-detail">Chip purchase should wait until the Akida-compatible surrogate survives compatibility, quantization, and conversion.</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Roadmap and major milestones</h2>
        <ul class="timeline">{milestones_html}</ul>
      </div>
      <div class="card">
        <h2>Pending tasks now</h2>
        <ul>{pending_tasks_html}</ul>
      </div>
    </div>

    <div class="grid" style="margin-top:16px;">
      <div class="card">
        <h2>Recent pivots</h2>
        <ul>{pivots_html}</ul>
      </div>
      <div class="card">
        <h2>Cleanup backlog</h2>
        <ul>{cleanup_html}</ul>
      </div>
    </div>

    <div class="grid" style="margin-top:16px;">
      <div class="card">
        <h2>Agent coordination</h2>
        <h3>Active sub-agents</h3>
        <ul>{active_agents_html}</ul>
        <h3 style="margin-top:16px;">Completed sub-agent wave</h3>
        <ul>{completed_agents_html}</ul>
        <h3 style="margin-top:16px;">Next likely sub-agents</h3>
        <ul>{next_agents_html}</ul>
      </div>
      <div class="card">
        {latest_snapshot_html}
        <h3 style="margin-top:16px;">Heartbeat</h3>
        <pre>{escape(heartbeat)}</pre>
      </div>
    </div>
llu
    <div class="grid" style="margin-top:16px;">
      <div class="card">
        <h2>Artifacts and reports</h2>
        <h3>Artifact timeline</h3>
        <ul>{artifact_list}</ul>
        <h3 style="margin-top:16px;">Reports</h3>
        <ul>{reports_html}</ul>
        <h3 style="margin-top:16px;">Metric artifacts</h3>
        <ul>{metrics_html}</ul>
      </div>
      <div class="card">
        <h2>Code inventory</h2>
        <h3>Experiment scripts</h3>
        <ul>{scripts_html}</ul>
        <h3 style="margin-top:16px;">Other files</h3>
        <ul>{others_html}</ul>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <h2>Recent logs from MEMORY.md</h2>
      {memory_html}
    </div>
  </div>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
            return
        page = render_html().encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(page)))
        self.end_headers()
        self.wfile.write(page)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Prototype-1 dashboard running on http://0.0.0.0:{PORT}')
    server.serve_forever()
