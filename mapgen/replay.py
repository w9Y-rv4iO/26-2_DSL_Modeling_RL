"""Construction replay: a self-contained HTML page that steps through the
designer's edits for the best verified map of every (mode, tau) cell.

    python -m mapgen.replay --validate mapgen/results/validate_r1.jsonl \
        --out mapgen/results/replay.html
"""

import argparse
import json
import os

from .design import N_CELLS, STOP, TIMEOUT_CAP, TYPES

TYPE_KO = {"wall2": "벽 2×2", "wallH": "가로 벽 1×3", "wallV": "세로 벽 3×1",
           "bush": "덤불 2×2", "item": "아이템 패드", "warp": "워프 패드",
           "gen": "발전기", "erase": "3×3 지우기"}

PAGE = """<title>맵 설계 리플레이</title>
<meta charset="utf-8">
<style>
:root { --bg:#eef2f1; --ink:#172224; --muted:#566664; --line:#d2dbd9; --card:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1517; --ink:#e1e8e6; --muted:#93a29f; --line:#2a3639; --card:#151d20; } }
body { background:var(--bg); color:var(--ink); font-family:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif; margin:0; }
main { max-width:980px; margin:0 auto; padding:28px 20px 60px; }
h1 { font-size:22px; margin:0 0 4px; } p.sub { color:var(--muted); margin:0 0 18px; font-size:14px; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:12px; }
select, button { font:inherit; font-size:14px; padding:5px 10px; border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:4px; cursor:pointer; }
input[type=range] { flex:1; min-width:160px; }
.stage { display:grid; grid-template-columns:minmax(0,1fr) 220px; gap:16px; align-items:start; }
@media (max-width:760px) { .stage { grid-template-columns:minmax(0,1fr); } }
canvas { width:100%; height:auto; background:var(--card); border:1px solid var(--line); image-rendering:pixelated; }
.panel { background:var(--card); border:1px solid var(--line); padding:14px; font-size:14px; display:grid; gap:10px; }
.k { color:var(--muted); font-size:12px; } .v { font-family:Consolas,monospace; font-size:18px; }
.gauge { position:relative; height:14px; background:var(--bg); border:1px solid var(--line); }
.gauge i { position:absolute; top:0; bottom:0; left:0; background:#2a5ec2; }
.gauge b { position:absolute; top:-4px; bottom:-4px; width:2px; background:#b7780c; }
.legend { display:flex; flex-wrap:wrap; gap:4px 12px; font-size:12px; color:var(--muted); margin-top:10px; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; margin-right:4px; vertical-align:-1px; background:var(--c); }
</style>
<main>
<h1>맵 설계 리플레이</h1>
<p class="sub">PPO 설계자가 빈 공터(또는 Lv3)에서 구조물을 한 개씩 놓는 과정. 게이지는 서로게이트 예측 승률, 금색 선은 목표 τ, 마지막 줄은 실제 휴리스틱 대전 결과.</p>
<div class="bar">
  <select id="cell"></select>
  <button id="prev">◀</button><button id="play">재생</button><button id="next">▶</button>
  <input id="slider" type="range" min="0" value="0">
</div>
<div class="stage">
  <canvas id="cv" width="750" height="450"></canvas>
  <div class="panel">
    <div><div class="k">스텝</div><div class="v" id="step"></div></div>
    <div><div class="k">이번 편집</div><div id="act"></div></div>
    <div><div class="k">예측 승률 (서로게이트)</div><div class="v" id="pw"></div><div class="gauge"><i id="gi"></i><b id="gt"></b></div></div>
    <div><div class="k">예측 시간초과</div><div class="v" id="pt"></div></div>
    <div><div class="k">실제 결과 (보류 시드 __GAMES__판)</div><div id="real"></div></div>
  </div>
</div>
<div class="legend">
  <span style="--c:#2c383a">벽</span><span style="--c:#4f9a55">덤불</span><span style="--c:#e0a020">아이템 패드</span>
  <span style="--c:#8a63d2">워프(숫자=쌍)</span><span style="--c:#d2553f">발전기</span><span style="--c:#1c8465">출구 후보</span>
</div>
</main>
<script>
const DATA = __DATA__;
const COLORS = {"#":"#2c383a",".":"#e9eeec","B":"#4f9a55","I":"#e0a020","G":"#d2553f","E":"#1c8465"};
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const sel = document.getElementById("cell"), slider = document.getElementById("slider");
let cur = 0, frame = 0, timer = null;
DATA.forEach((d, i) => { const o = document.createElement("option"); o.value = i;
  o.textContent = `${d.mode} · τ=${d.tau.toFixed(1)} → 실제 ${d.win.toFixed(2)}`; sel.appendChild(o); });
function draw() {
  const d = DATA[cur], f = d.frames[frame], rows = f.rows, s = 30;
  rows.forEach((row, y) => { [...row].forEach((ch, x) => {
    ctx.fillStyle = /[0-9]/.test(ch) ? "#8a63d2" : (COLORS[ch] || "#e9eeec");
    ctx.fillRect(x * s, y * s, s - 1, s - 1);
    if (/[0-9]/.test(ch)) { ctx.fillStyle = "#fff"; ctx.font = "bold 16px Consolas"; ctx.textAlign = "center"; ctx.fillText(ch, x * s + s / 2, y * s + 21); }
  }); });
  if (f.cells) { ctx.strokeStyle = "#b7780c"; ctx.lineWidth = 3;
    f.cells.forEach(([x, y]) => ctx.strokeRect(x * s + 1.5, y * s + 1.5, s - 4, s - 4)); }
  document.getElementById("step").textContent = `${frame} / ${d.frames.length - 1}`;
  document.getElementById("act").textContent = f.label;
  document.getElementById("pw").textContent = f.win_hat.toFixed(3);
  document.getElementById("pt").textContent = f.to_hat.toFixed(3);
  document.getElementById("gi").style.width = (f.win_hat * 100) + "%";
  document.getElementById("gt").style.left = (d.tau * 100) + "%";
  document.getElementById("real").textContent = `승률 ${d.win.toFixed(3)} · 시간초과 ${d.timeout.toFixed(3)} · 평균 ${d.time.toFixed(1)}초`;
  slider.max = d.frames.length - 1; slider.value = frame;
}
sel.onchange = () => { cur = +sel.value; frame = 0; draw(); };
slider.oninput = () => { frame = +slider.value; draw(); };
document.getElementById("prev").onclick = () => { frame = Math.max(0, frame - 1); draw(); };
document.getElementById("next").onclick = () => { frame = Math.min(DATA[cur].frames.length - 1, frame + 1); draw(); };
document.getElementById("play").onclick = (e) => {
  if (timer) { clearInterval(timer); timer = null; e.target.textContent = "재생"; return; }
  e.target.textContent = "정지"; if (frame >= DATA[cur].frames.length - 1) frame = 0;
  timer = setInterval(() => { if (frame >= DATA[cur].frames.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "재생"; return; } frame++; draw(); }, 450);
};
draw();
</script>
"""


def describe(action):
    if action == STOP:
        return "종료(STOP)", []
    ti, cell = divmod(action, N_CELLS)
    x, y = cell % 23 + 1, cell // 23 + 1
    from .design import FOOT
    typ = TYPES[ti]
    cells = [(x + dx, y + dy) for dx, dy in FOOT[typ]]
    return f"{TYPE_KO[typ]} @ ({x}, {y})", cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=48)
    args = p.parse_args()
    recs = [json.loads(line) for line in open(args.validate, encoding="utf-8")]
    data = []
    for mode in ("1v1", "2v4"):
        taus = sorted({r["tau"] for r in recs if r["mode"] == mode})
        for tau in taus:
            rs = [r for r in recs if r["mode"] == mode and r["tau"] == tau
                  and r["method"] == "designer" and r.get("trace")]
            if not rs:
                continue
            best = min(rs, key=lambda r: abs(r["win"] - tau)
                       + 2 * max(0.0, r["timeout"] - TIMEOUT_CAP))
            frames = []
            for f in best["trace"]:
                label, cells = describe(f["a"]) if "a" in f else ("시작 상태", [])
                frames.append({"rows": f["rows"], "win_hat": f["win_hat"],
                               "to_hat": f["to_hat"], "label": label, "cells": cells})
            last = best["trace"][-1]
            frames.append({"rows": best["rows"], "win_hat": last["win_hat"],
                           "to_hat": last["to_hat"], "label": "남은 발전기 자동 배치 (채점용 최종 맵)",
                           "cells": []})
            data.append({"mode": mode, "tau": tau, "win": best["win"],
                         "timeout": best["timeout"], "time": best["time"], "frames": frames})
    page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace(
        "__GAMES__", str(args.games))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({len(data)} replays)")


if __name__ == "__main__":
    main()
