import "./style.css";
import { Scene } from "./render";
import type { ServerMsg, StateMsg } from "./types";
import { UI } from "./ui";

const canvas = document.getElementById("map") as HTMLCanvasElement;
const hover = document.getElementById("hover")!;
const banner = document.getElementById("banner")!;
const scene = new Scene(canvas);

const ui = new UI({
  select(id) {
    scene.selected = id;
    scene.follow = true;
    ui.select(id);
  },
  control(c) {
    fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) });
  },
});

// --- network -----------------------------------------------------------------

let lastTick = performance.now();
let tickMs = 600;

function applyState(s: StateMsg) {
  const t = new Date(s.time);
  scene.hour = t.getHours() + t.getMinutes() / 60;
  scene.activeObjects = s.objects;
  ui.state(s);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { banner.textContent = ""; banner.className = ""; };
  ws.onmessage = (ev) => {
    const msg: ServerMsg = JSON.parse(ev.data);
    if (msg.type === "init") {
      scene.sprites.clear();
      scene.setWorld(msg.world);
      scene.updateAgents(msg.agents, 1);
      applyState(msg);
      ui.events(msg.events);
    } else if (msg.type === "tick") {
      const now = performance.now();
      tickMs = Math.min(3000, 0.6 * tickMs + 0.4 * (now - lastTick));
      lastTick = now;
      scene.updateAgents(msg.agents, tickMs);
      applyState(msg);
    } else if (msg.type === "event") {
      ui.events([msg.event]);
    }
  };
  ws.onclose = () => {
    banner.textContent = "Lost contact with the simulation, retrying…";
    banner.className = "show";
    setTimeout(connect, 1500);
  };
}
connect();

// --- input -------------------------------------------------------------------

let drag: { x: number; y: number; moved: boolean } | null = null;
canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", (e) => {
  const r = canvas.getBoundingClientRect();
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.moved) {
      scene.cam.x -= dx; scene.cam.y -= dy; scene.clampCam();
      scene.follow = false;
      drag.x = e.clientX; drag.y = e.clientY;
    }
  }
  showHover(e.clientX - r.left, e.clientY - r.top);
});
canvas.addEventListener("pointerup", (e) => {
  if (drag && !drag.moved) {
    const r = canvas.getBoundingClientRect();
    const id = scene.agentAt(e.clientX - r.left, e.clientY - r.top);
    if (id) { scene.selected = id; scene.follow = true; ui.select(id); }
  }
  drag = null;
});
canvas.addEventListener("pointerleave", () => { hover.style.display = "none"; });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  scene.zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });

function showHover(sx: number, sy: number) {
  const data = scene.data;
  if (!data) return;
  const [fx, fy] = scene.screenToTile(sx, sy);
  const tx = Math.floor(fx), ty = Math.floor(fy);
  const area = data.areas.find((a) => a.tiles.some(([x, y]) => x === tx && y === ty));
  const obj = data.objects.find((o) => o.x === tx && o.y === ty);
  const agent = scene.agentAt(sx, sy);
  const lines: string[] = [];
  if (agent) { const v = scene.sprites.get(agent)!.view; lines.push(`<b>${v.name}</b>`); }
  if (obj) lines.push(`${obj.name}${scene.activeObjects[obj.id] ? ` — <i>${scene.activeObjects[obj.id]}</i>` : ""}`);
  if (area) lines.push(`<span class="muted">${area.name}</span>`);
  if (!lines.length) { hover.style.display = "none"; return; }
  hover.innerHTML = lines.join("<br>");
  hover.style.display = "block";
  hover.style.left = `${sx + 14}px`;
  hover.style.top = `${sy + 14}px`;
}

// --- frame loop --------------------------------------------------------------

let prev = performance.now();
function loop(now: number) {
  const dt = Math.min(0.1, (now - prev) / 1000);
  prev = now;
  scene.frame(now, dt);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
