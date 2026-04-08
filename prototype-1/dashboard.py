import json
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE = ROOT / 'prototype-1'
ARTIFACTS = PROTOTYPE / 'artifacts'
MEMORY = ROOT / 'MEMORY.md'
HEARTBEAT = ROOT / 'HEARTBEAT.md'
COORDINATION = ARTIFACTS / 'coordination_status.json'
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


def latest_artifacts():
    items = []
    for path in sorted(ARTIFACTS.glob('*.json')):
        if path.name == 'coordination_status.json':
            continue
        stat = path.stat()
        items.append({
            'name': path.name,
            'modified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'data': read_json(path) or {},
        })
    items.sort(key=lambda x: x['modified'], reverse=True)
    return items


def recent_memory_blocks(limit: int = 4):
    text = read_text(MEMORY)
    blocks = [b.strip() for b in text.split('\n## ') if b.strip()]
    filtered = []
    for block in blocks:
        if block.startswith('Format') or block.startswith('Persistent Context'):
            continue
        filtered.append(block if block.startswith('202') else '## ' + block)
    return filtered[-limit:][::-1]


def file_inventory():
    files = []
    for path in sorted(PROTOTYPE.rglob('*')):
        if path.is_file():
            rel = path.relative_to(ROOT)
            files.append(str(rel))
    return files


def render_html():
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    artifacts = latest_artifacts()
    memory_blocks = recent_memory_blocks()
    heartbeat = read_text(HEARTBEAT, '(missing)').strip() or '(empty)'
    files = file_inventory()
    coordination = read_json(COORDINATION) or {}

    latest = artifacts[0] if artifacts else None
    latest_metrics = ''
    if latest:
        rows = ''.join(
            f'<tr><td>{escape(str(k))}</td><td><code>{escape(str(v))}</code></td></tr>'
            for k, v in latest['data'].items()
        )
        latest_metrics = f"""
        <h2>Latest artifact: {escape(latest['name'])}</h2>
        <p>Modified: {escape(latest['modified'])}</p>
        <table>{rows}</table>
        """

    artifact_list = ''.join(
        f"<li><strong>{escape(item['name'])}</strong> <span>({escape(item['modified'])})</span></li>"
        for item in artifacts
    ) + f"<li><strong>{escape(COORDINATION.name)}</strong> <span>(status/control plane)</span></li>"

    memory_html = ''.join(
        f'<pre>{escape(block)}</pre>' for block in memory_blocks
    ) or '<pre>No recent memory entries.</pre>'

    files_html = ''.join(f'<li><code>{escape(name)}</code></li>' for name in files) or '<li>No files yet.</li>'

    milestones = coordination.get('major_milestones', [])
    pending_tasks = coordination.get('pending_tasks', [])
    active_agents = coordination.get('active_subagents', [])
    completed_agents = coordination.get('completed_subagent_wave', [])
    next_agents = coordination.get('next_if_spawned', [])
    milestones_html = ''.join(
        f"<li><strong>{escape(m.get('marker', '?'))} {escape(m.get('name', 'unknown'))}</strong> <span>({escape(m.get('status', 'unknown'))})</span><br><span class='muted'>{escape(m.get('summary', ''))}</span></li>"
        for m in milestones
    ) or '<li>No milestones defined yet.</li>'
    pending_tasks_html = ''.join(f'<li>{escape(str(task))}</li>' for task in pending_tasks) or '<li>No pending tasks recorded.</li>'
    active_agents_html = ''.join(
        f"<li><strong>{escape(agent.get('name', 'unknown'))}</strong> <span>({escape(agent.get('status', 'unknown'))})</span><br><span class='muted'>{escape(agent.get('summary', ''))}</span></li>"
        for agent in active_agents
    ) or '<li>No active sub-agents right now.</li>'
    completed_agents_html = ''.join(
        f"<li><strong>{escape(agent.get('name', 'unknown'))}</strong> <span>({escape(agent.get('status', 'unknown'))})</span><br><span class='muted'>{escape(agent.get('summary', ''))}</span></li>"
        for agent in completed_agents
    ) or '<li>No completed sub-agent work recorded yet.</li>'
    next_agents_html = ''.join(f'<li><code>{escape(str(name))}</code></li>' for name in next_agents) or '<li>None planned.</li>'
    coordination_html = f"""
      <h2>Coordinator</h2>
      <p><strong>Phase:</strong> {escape(str(coordination.get('current_phase', 'Unknown')))}</p>
      <p><strong>Currently doing:</strong> {escape(str(coordination.get('currently_doing', 'Unknown')))}</p>
      <p class='muted'>Sub-agent snapshot updated: {escape(str(coordination.get('last_updated', 'unknown')))}</p>
      <h2>Major milestones</h2>
      <ul>{milestones_html}</ul>
      <h2>Pending tasks</h2>
      <ul>{pending_tasks_html}</ul>
      <h2>Active sub-agents</h2>
      <ul>{active_agents_html}</ul>
      <h2>Completed sub-agent wave</h2>
      <ul>{completed_agents_html}</ul>
      <h2>Next likely sub-agents</h2>
      <ul>{next_agents_html}</ul>
    """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <title>Prototype-1 Live Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #0b1020; color: #e8ecf3; }}
    h1, h2 {{ color: #9dd6ff; }}
    .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
    .card {{ background: #131a2a; border: 1px solid #26324a; border-radius: 10px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-top: 1px solid #26324a; padding: 6px; vertical-align: top; }}
    code, pre {{ background: #0d1423; color: #d5e7ff; border-radius: 6px; padding: 2px 6px; }}
    pre {{ white-space: pre-wrap; padding: 12px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .muted {{ color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Prototype-1 Live Dashboard</h1>
  <p class="muted">Auto-refresh every 15s. Generated at {escape(now)}.</p>
  <div class="grid">
    <div class="card">
      <h2>Current focus</h2>
      <p>Push sparse/spiking attention efficiency upward while preserving quality, then map the winning pattern toward Akida-compatible conversion.</p>
      {coordination_html}
      {latest_metrics}
      <h2>Recent memory</h2>
      {memory_html}
    </div>
    <div class="card">
      <h2>Artifacts</h2>
      <ul>{artifact_list}</ul>
      <h2>Prototype files</h2>
      <ul>{files_html}</ul>
      <h2>Heartbeat</h2>
      <pre>{escape(heartbeat)}</pre>
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
