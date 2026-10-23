// Canvas renderer: terrain, villagers, torchlight and the day/night cycle.
import { TILE, paintTerrain, collectWaterTiles } from "./tiles";
import type { AgentView, WorldData } from "./types";

interface Sprite {
  view: AgentView;
  route: [number, number][]; // tile coordinates to animate through this tick
  t0: number;
  dur: number;
  x: number; // current animated position, in tiles
  y: number;
  bubbleAt: number;
}

interface Mote { x: number; y: number; vx: number; vy: number; r: number; a: number }

const lerp = (a: number, b: number, t: number) => a + (b - a) * Math.max(0, Math.min(1, t));

export function darkness(hour: number): number {
  if (hour < 6.5) return 0.84;
  if (hour < 8) return lerp(0.84, 0.22, (hour - 6.5) / 1.5);
  if (hour < 15.5) return 0.22;
  if (hour < 18) return lerp(0.22, 0.84, (hour - 15.5) / 2.5);
  return 0.84;
}

export class Scene {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private lightCanvas = document.createElement("canvas");
  private terrain: HTMLCanvasElement | null = null;
  data: WorldData | null = null;
  sprites = new Map<string, Sprite>();
  cam = { x: 0, y: 0, zoom: 2 };
  selected: string | null = null;
  follow = true;
  hour = 12;
  activeObjects: Record<string, string> = {};
  private motes: Mote[] = [];
  private dpr = window.devicePixelRatio || 1;
  private waterTiles: [number, number][] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    for (let i = 0; i < 70; i++) this.motes.push(this.newMote(true));
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement!);
    this.resize();
  }

  private newMote(anywhere: boolean): Mote {
    const w = this.canvas.clientWidth || 800, h = this.canvas.clientHeight || 600;
    return {
      x: Math.random() * w, y: anywhere ? Math.random() * h : -5,
      vx: 4 + Math.random() * 8, vy: 6 + Math.random() * 10, r: 0.6 + Math.random() * 1.4, a: 0.15 + Math.random() * 0.35,
    };
  }

  resize() {
    const el = this.canvas.parentElement!;
    this.canvas.width = el.clientWidth * this.dpr;
    this.canvas.height = el.clientHeight * this.dpr;
    this.canvas.style.width = `${el.clientWidth}px`;
    this.canvas.style.height = `${el.clientHeight}px`;
    this.lightCanvas.width = this.canvas.width;
    this.lightCanvas.height = this.canvas.height;
  }

  setWorld(data: WorldData) {
    this.data = data;
    this.terrain = paintTerrain(data);
    this.waterTiles = collectWaterTiles(data);
    const vw = this.canvas.clientWidth, vh = this.canvas.clientHeight;
    this.cam.zoom = Math.max(1, Math.min(vw / (data.width * TILE), vh / (data.height * TILE)) * 1.6);
    this.centerOn(data.width / 2, data.height / 2);
  }

  centerOn(tx: number, ty: number) {
    this.cam.x = (tx + 0.5) * TILE * this.cam.zoom - this.canvas.clientWidth / 2;
    this.cam.y = (ty + 0.5) * TILE * this.cam.zoom - this.canvas.clientHeight / 2;
    this.clampCam();
  }

  clampCam() {
    if (!this.data) return;
    const ww = this.data.width * TILE * this.cam.zoom, wh = this.data.height * TILE * this.cam.zoom;
    const vw = this.canvas.clientWidth, vh = this.canvas.clientHeight;
    this.cam.x = ww < vw ? (ww - vw) / 2 : Math.max(0, Math.min(this.cam.x, ww - vw));
    this.cam.y = wh < vh ? (wh - vh) / 2 : Math.max(0, Math.min(this.cam.y, wh - vh));
  }

  zoomAt(factor: number, sx: number, sy: number) {
    const z0 = this.cam.zoom;
    const z1 = Math.max(0.75, Math.min(6, z0 * factor));
    this.cam.x = (this.cam.x + sx) * (z1 / z0) - sx;
    this.cam.y = (this.cam.y + sy) * (z1 / z0) - sy;
    this.cam.zoom = z1;
    this.clampCam();
  }

  screenToTile(sx: number, sy: number): [number, number] {
    return [(this.cam.x + sx) / (TILE * this.cam.zoom), (this.cam.y + sy) / (TILE * this.cam.zoom)];
  }

  agentAt(sx: number, sy: number): string | null {
    const [tx, ty] = this.screenToTile(sx, sy);
    let best: string | null = null, bestD = 0.9;
    for (const [id, s] of this.sprites) {
      const d = Math.hypot(s.x + 0.5 - tx, s.y + 0.3 - ty);
      if (d < bestD) { best = id; bestD = d; }
    }
    return best;
  }

  updateAgents(views: AgentView[], tickMs: number) {
    const now = performance.now();
    for (const v of views) {
      const s = this.sprites.get(v.id);
      if (!s) {
        this.sprites.set(v.id, { view: v, route: [[v.x, v.y]], t0: now, dur: 1, x: v.x, y: v.y, bubbleAt: 0 });
        continue;
      }
      const route: [number, number][] = [[s.x, s.y]];
      if (v.steps.length) route.push(...v.steps);
      else if (v.x !== s.x || v.y !== s.y) route.push([v.x, v.y]);
      if (v.bubble && v.bubble !== s.view.bubble) s.bubbleAt = now;
      s.view = v;
      s.route = route;
      s.t0 = now;
      s.dur = Math.max(120, tickMs * 0.95);
    }
  }

  private animate(s: Sprite, now: number) {
    const r = s.route;
    if (r.length < 2) { [s.x, s.y] = r[0]; return; }
    const t = Math.max(0, Math.min(1, (now - s.t0) / s.dur)) * (r.length - 1);
    const i = Math.min(r.length - 2, Math.floor(t));
    s.x = lerp(r[i][0], r[i + 1][0], t - i);
    s.y = lerp(r[i][1], r[i + 1][1], t - i);
  }

  frame(now: number, dt: number) {
    const c = this.ctx, data = this.data;
    if (!data || !this.terrain) return;
    for (const s of this.sprites.values()) this.animate(s, now);
    if (this.follow && this.selected) {
      const s = this.sprites.get(this.selected);
      if (s) {
        const tx = (s.x + 0.5) * TILE * this.cam.zoom - this.canvas.clientWidth / 2;
        const ty = (s.y + 0.5) * TILE * this.cam.zoom - this.canvas.clientHeight / 2;
        this.cam.x = lerp(this.cam.x, tx, dt * 3);
        this.cam.y = lerp(this.cam.y, ty, dt * 3);
        this.clampCam();
      }
    }

    const z = this.cam.zoom * this.dpr;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = "#07080b";
    c.fillRect(0, 0, this.canvas.width, this.canvas.height);
    c.imageSmoothingEnabled = false;
    c.setTransform(z, 0, 0, z, -this.cam.x * this.dpr, -this.cam.y * this.dpr);
    c.drawImage(this.terrain, 0, 0);
    this.drawWater(c, now);
    this.drawActiveObjects(c, now);

    const sorted = [...this.sprites.values()].sort((a, b) => a.y - b.y);
    for (const s of sorted) this.drawAgent(c, s, now);

    const dark = darkness(this.hour);
    this.drawLighting(dark, now);
    c.setTransform(z, 0, 0, z, -this.cam.x * this.dpr, -this.cam.y * this.dpr);
    c.globalAlpha = Math.max(0, dark - 0.3) * 0.75;
    for (const s of sorted) this.drawAgent(c, s, now);
    c.globalAlpha = 1;
    this.drawWeather(dt);
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    for (const s of sorted) this.drawLabels(c, s, now);
    this.drawVignette(c);
  }

  private drawWater(c: CanvasRenderingContext2D, now: number) {
    c.fillStyle = "rgba(120,150,170,0.10)";
    for (const [x, y] of this.waterTiles) {
      const ph = Math.sin(now / 900 + y * 0.7 + x) * 4;
      c.fillRect(x * TILE + 4 + ph, y * TILE + ((now / 120 + y * 5) % TILE), 5, 1);
    }
  }

  private drawActiveObjects(c: CanvasRenderingContext2D, now: number) {
    const data = this.data!;
    for (const o of data.objects) {
      if (!this.activeObjects[o.id]) continue;
      const pulse = 0.35 + Math.sin(now / 300) * 0.15;
      c.strokeStyle = `rgba(214,170,90,${pulse})`;
      c.lineWidth = 1;
      c.strokeRect(o.x * TILE + 0.5, o.y * TILE + 0.5, TILE - 1, TILE - 1);
    }
  }

  private drawAgent(c: CanvasRenderingContext2D, s: Sprite, now: number) {
    const v = s.view;
    const cx = s.x * TILE + 8, cy = s.y * TILE + 8;
    const moving = s.route.length > 1 && now - s.t0 < s.dur;
    const bob = moving ? Math.abs(Math.sin(now / 90)) * 1.2 : 0;
    if (v.id === this.selected) {
      c.strokeStyle = "rgba(230,190,100,0.9)";
      c.lineWidth = 1;
      c.beginPath(); c.ellipse(cx, cy + 6, 6, 3, 0, 0, Math.PI * 2); c.stroke();
    }
    c.fillStyle = "rgba(0,0,0,0.45)";
    c.beginPath(); c.ellipse(cx, cy + 6, 4.5, 2, 0, 0, Math.PI * 2); c.fill();
    if (v.species === "goblin") { this.drawGoblin(c, v, cx, cy - bob, now); return; }
    c.save();
    c.translate(cx, cy - bob);
    if (v.mode === "asleep") c.rotate(-Math.PI / 2);
    // cloak
    c.fillStyle = shade(v.color, -0.35);
    c.beginPath(); c.moveTo(-4.5, 6); c.lineTo(4.5, 6); c.lineTo(2.6, -2); c.lineTo(-2.6, -2); c.closePath(); c.fill();
    c.fillStyle = v.color;
    c.beginPath(); c.moveTo(-3.5, 6); c.lineTo(0.5, 6); c.lineTo(1.5, -2); c.lineTo(-2.6, -2); c.closePath(); c.fill();
    // head and hood
    c.fillStyle = "#d4b08a";
    c.beginPath(); c.arc(0, -4.2, 2.5, 0, Math.PI * 2); c.fill();
    c.fillStyle = shade(v.color, -0.5);
    c.beginPath(); c.arc(0, -4.6, 3, Math.PI * 1.05, Math.PI * 1.95); c.fill();
    c.restore();
  }

  private drawGoblin(c: CanvasRenderingContext2D, v: AgentView, cx: number, cy: number, now: number) {
    c.save();
    c.translate(cx, cy + 1);
    if (v.mode === "asleep") c.rotate(-Math.PI / 2);
    if (v.mode === "hiding") c.globalAlpha *= 0.35 + Math.sin(now / 400) * 0.05;
    const skin = v.color, dark = shade(v.color, -0.45);
    // hunched body in a rag
    c.fillStyle = "#3a3026";
    c.beginPath(); c.moveTo(-3.5, 5); c.lineTo(3.5, 5); c.lineTo(2.2, -0.5); c.lineTo(-2.2, -0.5); c.closePath(); c.fill();
    c.fillStyle = skin;
    c.fillRect(-3, 4, 1.5, 1.5); c.fillRect(1.5, 4, 1.5, 1.5);
    // big head, long ears
    c.beginPath(); c.arc(0, -2.8, 2.9, 0, Math.PI * 2); c.fill();
    c.beginPath(); c.moveTo(-2.4, -3.6); c.lineTo(-7, -5.6); c.lineTo(-2.2, -1.8); c.closePath(); c.fill();
    c.beginPath(); c.moveTo(2.4, -3.6); c.lineTo(7, -5.6); c.lineTo(2.2, -1.8); c.closePath(); c.fill();
    c.fillStyle = dark;
    c.fillRect(-2.4, -1.2, 4.8, 0.8);
    c.fillStyle = "#e8d84a";
    c.fillRect(-1.6, -3.4, 1, 1); c.fillRect(0.7, -3.4, 1, 1);
    c.restore();
  }

  private drawLabels(c: CanvasRenderingContext2D, s: Sprite, now: number) {
    const v = s.view;
    const z = this.cam.zoom;
    const sx = (s.x * TILE + 8) * z - this.cam.x, sy = (s.y * TILE + 8) * z - this.cam.y;
    if (sx < -200 || sy < -200 || sx > this.canvas.clientWidth + 200 || sy > this.canvas.clientHeight + 200) return;
    c.textAlign = "center";
    c.textBaseline = "middle";
    const selected = v.id === this.selected;
    if (z >= 1.5 || selected) {
      c.font = `${selected ? 600 : 400} 12px "EB Garamond", Georgia, serif`;
      c.lineWidth = 3;
      c.strokeStyle = "rgba(0,0,0,0.85)";
      c.strokeText(v.first, sx, sy + 9 * z + 6);
      c.fillStyle = selected ? "#f0d08a" : "#d9cfbd";
      c.fillText(v.first, sx, sy + 9 * z + 6);
    }
    const top = sy - 8 * z - 8;
    if (v.thinking) {
      const dots = ".".repeat(1 + Math.floor(now / 350) % 3);
      c.font = "bold 14px Georgia, serif";
      c.fillStyle = "#cfc6ff";
      c.fillText(dots, sx, top);
    } else if (v.emoji && v.mode !== "talking") {
      c.font = "14px 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif";
      c.fillText(v.emoji, sx, top);
    }
    if (v.mode === "hiding" && !v.thinking) {
      c.font = "italic 11px Georgia, serif";
      c.fillStyle = "rgba(170,200,140,0.8)";
      c.fillText("hiding", sx, top - 12);
    }
    if (v.mode === "asleep") {
      c.font = "italic 11px Georgia, serif";
      c.fillStyle = "rgba(200,200,255,0.7)";
      c.fillText("z", sx + 8 + Math.sin(now / 500) * 2, top - 4 - ((now / 60) % 10));
    }
    if (v.bubble && v.mode === "talking") this.drawBubble(c, v.bubble, sx, top - 4, now - s.bubbleAt);
  }

  private drawBubble(c: CanvasRenderingContext2D, text: string, x: number, y: number, age: number) {
    c.font = `14px "EB Garamond", Georgia, serif`;
    const lines = wrap(c, text, 230).slice(0, 5);
    const w = Math.max(...lines.map((l) => c.measureText(l).width)) + 18;
    const h = lines.length * 17 + 12;
    const bx = Math.max(6, Math.min(x - w / 2, this.canvas.clientWidth - w - 6));
    const by = y - h - 8;
    c.globalAlpha = Math.min(1, age / 200);
    c.fillStyle = "rgba(232,220,192,0.96)";
    c.strokeStyle = "#3a2a1a";
    c.lineWidth = 1.5;
    roundRect(c, bx, by, w, h, 6);
    c.fill(); c.stroke();
    c.beginPath(); c.moveTo(x - 6, by + h); c.lineTo(x, by + h + 8); c.lineTo(x + 6, by + h); c.closePath();
    c.fill();
    c.fillStyle = "#24170d";
    c.textAlign = "left";
    lines.forEach((l, i) => c.fillText(l, bx + 9, by + 14 + i * 17));
    c.globalAlpha = 1;
    c.textAlign = "center";
  }

  private drawLighting(dark: number, now: number) {
    const data = this.data!;
    const l = this.lightCanvas.getContext("2d")!;
    const z = this.cam.zoom * this.dpr;
    const toScreen = (tx: number, ty: number) => [tx * TILE * z - this.cam.x * this.dpr, ty * TILE * z - this.cam.y * this.dpr];
    l.globalCompositeOperation = "source-over";
    l.clearRect(0, 0, this.lightCanvas.width, this.lightCanvas.height);
    l.fillStyle = `rgba(6,8,18,${dark})`;
    l.fillRect(0, 0, this.lightCanvas.width, this.lightCanvas.height);
    const lights = data.lights.map((li, i) => ({ ...li, i }));
    for (const s of this.sprites.values()) {
      if (s.view.mode === "walking" && dark > 0.5) lights.push({ x: s.x + 0.5, y: s.y + 0.3, radius: 2.2, color: "#ffb060", flicker: true, i: 99 });
    }
    l.globalCompositeOperation = "destination-out";
    const glows: [number, number, number, string][] = [];
    for (const li of lights) {
      const f = li.flicker ? 1 + Math.sin(now / 110 + li.i * 13) * 0.04 + Math.sin(now / 47 + li.i) * 0.03 : 1;
      const [sx, sy] = toScreen(li.x, li.y);
      const r = li.radius * TILE * z * f;
      if (sx < -r || sy < -r || sx > this.lightCanvas.width + r || sy > this.lightCanvas.height + r) continue;
      const g = l.createRadialGradient(sx, sy, 0, sx, sy, r);
      g.addColorStop(0, "rgba(0,0,0,1)");
      g.addColorStop(0.45, "rgba(0,0,0,0.75)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      l.fillStyle = g;
      l.beginPath(); l.arc(sx, sy, r, 0, Math.PI * 2); l.fill();
      glows.push([sx, sy, r, li.color]);
    }
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.drawImage(this.lightCanvas, 0, 0);
    // a cold, desaturating wash by day; warm additive glow around fires by night
    c.fillStyle = "rgba(40,48,60,0.18)";
    c.fillRect(0, 0, this.canvas.width, this.canvas.height);
    c.globalCompositeOperation = "lighter";
    for (const [sx, sy, r, color] of glows) {
      const g = c.createRadialGradient(sx, sy, 0, sx, sy, r * 0.8);
      g.addColorStop(0, hexA(color, 0.10 + dark * 0.22));
      g.addColorStop(1, hexA(color, 0));
      c.fillStyle = g;
      c.beginPath(); c.arc(sx, sy, r * 0.8, 0, Math.PI * 2); c.fill();
    }
    c.globalCompositeOperation = "source-over";
  }

  private drawWeather(dt: number) {
    const c = this.ctx;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    for (let i = 0; i < this.motes.length; i++) {
      const m = this.motes[i];
      m.x += m.vx * dt; m.y += m.vy * dt;
      if (m.y > h + 5 || m.x > w + 5) this.motes[i] = this.newMote(false);
      c.fillStyle = `rgba(190,185,175,${m.a})`;
      c.beginPath(); c.arc(m.x, m.y, m.r, 0, Math.PI * 2); c.fill();
    }
  }

  private drawVignette(c: CanvasRenderingContext2D) {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const g = c.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.35, w / 2, h / 2, Math.max(w, h) * 0.75);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0,0.65)");
    c.fillStyle = g;
    c.fillRect(0, 0, w, h);
  }
}

function wrap(c: CanvasRenderingContext2D, text: string, max: number): string[] {
  const words = text.split(/\s+/), lines: string[] = [];
  let cur = "";
  for (const w of words) {
    const next = cur ? `${cur} ${w}` : w;
    if (c.measureText(next).width > max && cur) { lines.push(cur); cur = w; } else cur = next;
  }
  if (cur) lines.push(cur);
  if (lines.length > 5) lines[4] += "…";
  return lines;
}

function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function shade(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const f = (v: number) => Math.round(Math.max(0, Math.min(255, amt < 0 ? v * (1 + amt) : v + (255 - v) * amt)));
  return `rgb(${f(n >> 16)},${f((n >> 8) & 255)},${f(n & 255)})`;
}

function hexA(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}
