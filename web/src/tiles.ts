// Procedural pixel art: every tile and object is painted from code, no image assets.
import type { WorldData, WorldObject } from "./types";

export const TILE = 16;

// Fallback ids, overwritten by syncTileIds() from the server's authoritative
// WorldData.tileNames the moment a world loads, so a reordered Tile enum on
// the backend can't silently desync the colors/textures painted here.
const T = {
  GRASS: 0, DIRT: 1, COBBLE: 2, WOOD: 3, STONE: 4, WALL: 5, WATER: 6, TREE: 7,
  DOOR: 8, FENCE: 9, GRAVE_SOIL: 10, BRIDGE: 11, FIELD: 12,
};

function syncTileIds(tileNames: Record<number, string>) {
  for (const [idStr, name] of Object.entries(tileNames)) {
    const key = name.toUpperCase() as keyof typeof T;
    if (key in T) T[key] = Number(idStr);
  }
}

function hash(x: number, y: number, s = 0): number {
  let h = (x * 374761393 + y * 668265263 + s * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

type Ctx = CanvasRenderingContext2D;

function rect(c: Ctx, x: number, y: number, w: number, h: number, color: string) {
  c.fillStyle = color;
  c.fillRect(x, y, w, h);
}

function speckle(c: Ctx, px: number, py: number, tx: number, ty: number, colors: string[], n: number) {
  for (let i = 0; i < n; i++) {
    const r = hash(tx, ty, i + 11);
    rect(c, px + Math.floor(hash(tx, ty, i) * TILE), py + Math.floor(hash(tx, ty, i + 50) * TILE), 1, 1,
      colors[Math.floor(r * colors.length)]);
  }
}

function ground(c: Ctx, kind: number, tx: number, ty: number) {
  const px = tx * TILE, py = ty * TILE;
  switch (kind) {
    case T.GRASS:
    case T.TREE:
    case T.FENCE:
      rect(c, px, py, TILE, TILE, hash(tx, ty, 3) < 0.5 ? "#2c3222" : "#2a301f");
      speckle(c, px, py, tx, ty, ["#363d29", "#22281a", "#3f4430", "#4a4532"], 14);
      if (hash(tx, ty, 9) < 0.25) {
        c.fillStyle = "#5a5238";
        c.fillRect(px + 4 + Math.floor(hash(tx, ty, 4) * 8), py + 3 + Math.floor(hash(tx, ty, 5) * 8), 1, 3);
      }
      break;
    case T.DIRT:
      rect(c, px, py, TILE, TILE, "#3a2f25");
      speckle(c, px, py, tx, ty, ["#45382b", "#2e251d", "#524434"], 12);
      break;
    case T.GRAVE_SOIL:
      rect(c, px, py, TILE, TILE, "#29261f");
      speckle(c, px, py, tx, ty, ["#332f26", "#1f1c17", "#3b3a2c"], 12);
      break;
    case T.FIELD: {
      rect(c, px, py, TILE, TILE, "#362c20");
      for (let r = 2; r < TILE; r += 4) {
        rect(c, px, py + r, TILE, 1, "#2a2219");
        for (let k = 0; k < 4; k++) {
          if (hash(tx, ty, r * 7 + k) < 0.6) rect(c, px + 1 + k * 4, py + r - 2, 1, 2, "#6b5d3a");
        }
      }
      break;
    }
    case T.COBBLE: {
      rect(c, px, py, TILE, TILE, "#2b2927");
      for (let sy = 0; sy < 4; sy++) {
        const off = (sy + ty) % 2 ? 2 : 0;
        for (let sx = -1; sx < 4; sx++) {
          const x0 = px + sx * 4 + off, y0 = py + sy * 4;
          const shade = 60 + Math.floor(hash(tx * 4 + sx, ty * 4 + sy) * 22);
          c.fillStyle = `rgb(${shade},${shade - 3},${shade - 7})`;
          c.fillRect(Math.max(px, x0), y0, Math.min(3, px + TILE - x0), 3);
        }
      }
      break;
    }
    case T.WOOD:
    case T.DOOR:
    case T.BRIDGE: {
      const base = kind === T.BRIDGE ? [86, 66, 46] : [70, 50, 34];
      for (let r = 0; r < 4; r++) {
        const v = Math.floor(hash(tx, ty * 4 + r) * 12);
        c.fillStyle = `rgb(${base[0] + v},${base[1] + v},${base[2] + v})`;
        c.fillRect(px, py + r * 4, TILE, 4);
        rect(c, px, py + r * 4 + 3, TILE, 1, "#2a1d13");
        rect(c, px + Math.floor(hash(tx, ty, r) * 14), py + r * 4, 1, 3, "#2a1d13");
      }
      if (kind === T.DOOR) {
        rect(c, px + 1, py + 1, TILE - 2, TILE - 2, "#5a3a1e");
        rect(c, px + 3, py + 3, TILE - 6, TILE - 6, "#6e4826");
        rect(c, px + TILE - 5, py + 8, 2, 2, "#c9a24a");
      }
      break;
    }
    case T.STONE: {
      rect(c, px, py, TILE, TILE, "#2a2a2e");
      for (const [x0, y0, w, h] of [[0, 0, 8, 8], [8, 0, 8, 8], [0, 8, 8, 8], [8, 8, 8, 8]]) {
        const v = 52 + Math.floor(hash(tx * 2 + x0, ty * 2 + y0) * 14);
        c.fillStyle = `rgb(${v},${v},${v + 5})`;
        c.fillRect(px + x0, py + y0, w - 1, h - 1);
      }
      break;
    }
    case T.WATER: {
      rect(c, px, py, TILE, TILE, "#14222b");
      speckle(c, px, py, tx, ty, ["#1c2f3a", "#0f1a21"], 10);
      break;
    }
    case T.WALL:
      rect(c, px, py, TILE, TILE, "#1e1b21");
      break;
  }
}

function wall(c: Ctx, data: WorldData, tx: number, ty: number) {
  const px = tx * TILE, py = ty * TILE;
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= data.width || y >= data.height)
    ? -1 : data.tiles[y * data.width + x];
  rect(c, px, py, TILE, TILE, "#2b2730");
  speckle(c, px, py, tx, ty, ["#35303b", "#221f26"], 10);
  if (at(tx, ty + 1) !== T.WALL) {  // front face visible: brick courses
    rect(c, px, py + 8, TILE, 8, "#3c3540");
    for (let r = 0; r < 2; r++) {
      rect(c, px, py + 8 + r * 4 + 3, TILE, 1, "#1d1a20");
      const off = (r + tx) % 2 ? 0 : 4;
      for (let k = off; k < TILE; k += 8) rect(c, px + k, py + 8 + r * 4, 1, 3, "#1d1a20");
    }
  }
  rect(c, px, py, TILE, 1, "#4a4350");
}

function tree(c: Ctx, tx: number, ty: number) {
  const cx = tx * TILE + 8 + (hash(tx, ty, 1) - 0.5) * 6;
  const cy = ty * TILE + 6 + (hash(tx, ty, 2) - 0.5) * 6;
  const r = 8 + hash(tx, ty, 3) * 4;
  c.fillStyle = "rgba(0,0,0,0.35)";
  c.beginPath(); c.ellipse(cx + 2, cy + 8, r * 0.9, r * 0.45, 0, 0, Math.PI * 2); c.fill();
  rect(c, cx - 1, cy + 2, 3, 8, "#2a1e15");
  const dead = hash(tx, ty, 4) < 0.18;
  if (dead) {
    c.strokeStyle = "#2f251c"; c.lineWidth = 1.5;
    for (let k = 0; k < 4; k++) {
      const a = -Math.PI / 2 + (k - 1.5) * 0.6;
      c.beginPath(); c.moveTo(cx, cy + 3); c.lineTo(cx + Math.cos(a) * r, cy + 3 + Math.sin(a) * r); c.stroke();
    }
    return;
  }
  const shades = ["#141c12", "#1a2417", "#213020"];
  for (let i = 0; i < 3; i++) {
    c.fillStyle = shades[i];
    c.beginPath();
    c.arc(cx - i * 1.2, cy - i * 1.5, r - i * 2.4, 0, Math.PI * 2);
    c.fill();
  }
}

function fence(c: Ctx, data: WorldData, tx: number, ty: number) {
  const px = tx * TILE, py = ty * TILE;
  const at = (x: number, y: number) => data.tiles[y * data.width + x] === T.FENCE;
  c.fillStyle = "#4b3b2b";
  if (at(tx - 1, ty) || at(tx + 1, ty)) { c.fillRect(px, py + 6, TILE, 2); c.fillRect(px, py + 11, TILE, 2); }
  if (at(tx, ty - 1) || at(tx, ty + 1)) { c.fillRect(px + 6, py, 2, TILE); c.fillRect(px + 10, py, 2, TILE); }
  rect(c, px + 6, py + 4, 4, 10, "#3a2c1f");
}

export function paintTerrain(data: WorldData): HTMLCanvasElement {
  syncTileIds(data.tileNames);
  const canvas = document.createElement("canvas");
  canvas.width = data.width * TILE;
  canvas.height = data.height * TILE;
  const c = canvas.getContext("2d")!;
  for (let y = 0; y < data.height; y++)
    for (let x = 0; x < data.width; x++) ground(c, data.tiles[y * data.width + x], x, y);
  for (let y = 0; y < data.height; y++)
    for (let x = 0; x < data.width; x++) {
      const k = data.tiles[y * data.width + x];
      if (k === T.WALL) wall(c, data, x, y);
      if (k === T.FENCE) fence(c, data, x, y);
    }
  for (const o of data.objects) paintObject(c, o);
  for (let y = 0; y < data.height; y++)
    for (let x = 0; x < data.width; x++) if (data.tiles[y * data.width + x] === T.TREE) tree(c, x, y);
  return canvas;
}

export function isWater(data: WorldData, x: number, y: number) {
  return data.tiles[y * data.width + x] === T.WATER;
}

export function collectWaterTiles(data: WorldData): [number, number][] {
  const tiles: [number, number][] = [];
  for (let y = 0; y < data.height; y++)
    for (let x = 0; x < data.width; x++)
      if (data.tiles[y * data.width + x] === T.WATER) tiles.push([x, y]);
  return tiles;
}

// ---------------------------------------------------------------- objects

function paintObject(c: Ctx, o: WorldObject) {
  const px = o.x * TILE, py = o.y * TILE;
  const R = (x: number, y: number, w: number, h: number, col: string) => rect(c, px + x, py + y, w, h, col);
  const circle = (x: number, y: number, r: number, col: string) => {
    c.fillStyle = col; c.beginPath(); c.arc(px + x, py + y, r, 0, Math.PI * 2); c.fill();
  };
  switch (o.kind) {
    case "bed":
      R(1, 2, 14, 12, "#3a2414"); R(2, 3, 12, 10, "#6d6150"); R(2, 3, 12, 3, "#b8ae98"); R(2, 8, 12, 5, "#5a3b3b"); break;
    case "counter":
      R(-8, 3, 32, 9, "#2c1c10"); R(-8, 3, 32, 3, "#5d3f24"); R(-2, 1, 3, 3, "#8a8070"); R(10, 1, 3, 3, "#8a8070"); break;
    case "hearth":
      R(0, 0, 16, 16, "#3d3a3a"); R(3, 5, 10, 9, "#140c08"); R(5, 9, 6, 4, "#ff7a2a"); R(6, 8, 4, 2, "#ffc66b"); break;
    case "table":
      R(-4, 3, 24, 10, "#4e3420"); R(-4, 3, 24, 2, "#6e4a2c"); R(2, 6, 3, 2, "#a89b80"); R(11, 7, 3, 2, "#a89b80"); break;
    case "casks":
      circle(5, 5, 4.5, "#4a2f1a"); circle(12, 9, 4.5, "#4a2f1a"); circle(5, 13, 4.5, "#4a2f1a");
      for (const [x, y] of [[5, 5], [12, 9], [5, 13]]) circle(x, y, 2, "#2c1c10");
      break;
    case "chest":
      R(2, 4, 12, 9, "#3b2a1a"); R(2, 4, 12, 3, "#56402a"); R(7, 8, 2, 2, "#a08a50"); break;
    case "window":
      R(3, 0, 10, 4, "#1c2a38"); R(7, 0, 1, 4, "#3a3040"); break;
    case "forge":
      R(-2, -2, 20, 18, "#2a2626"); R(1, 2, 14, 11, "#140906"); R(3, 6, 10, 6, "#ff5a1f"); R(5, 7, 6, 3, "#ffb04a"); break;
    case "anvil":
      R(3, 9, 10, 5, "#222"); R(1, 5, 14, 4, "#555a60"); R(-1, 5, 3, 2, "#555a60"); R(6, 9, 4, 3, "#333"); break;
    case "barrel":
      circle(8, 8, 6, "#4a2f1a"); circle(8, 8, 4, "#1d2a30"); break;
    case "rack":
      R(1, 2, 14, 2, "#3b2a1a"); R(1, 12, 14, 2, "#3b2a1a");
      for (let k = 3; k < 15; k += 4) R(k, 0, 1, 14, "#8a8e95");
      break;
    case "brazier":
      R(4, 8, 8, 6, "#2a2626"); R(5, 5, 6, 4, "#ff7a2a"); R(6, 4, 4, 2, "#ffc66b"); break;
    case "gate":
      R(-6, 0, 28, 6, "#3b2a1a");
      for (let k = -5; k < 22; k += 3) R(k, 0, 1, 6, "#1d140c");
      break;
    case "well":
      circle(8, 8, 7, "#4b4743"); circle(8, 8, 4.5, "#0b1014"); R(0, 0, 2, 9, "#3b2a1a"); R(14, 0, 2, 9, "#3b2a1a"); R(0, 0, 16, 2, "#3b2a1a"); break;
    case "gallows":
      R(2, -10, 3, 26, "#2c1f14"); R(2, -10, 14, 3, "#2c1f14"); R(13, -7, 1, 9, "#8a7a5a"); circle(13.5, 3, 2, "#8a7a5a"); circle(13.5, 3, 1, "#2c3222"); break;
    case "stall":
      R(-4, 2, 24, 10, "#3b2a1a"); R(-4, -1, 24, 5, "#5a2222");
      for (let k = -4; k < 20; k += 6) R(k, -1, 3, 5, "#7a5a3a");
      break;
    case "post":
      R(7, 0, 2, 16, "#3b2a1a"); R(2, 2, 12, 8, "#6b5d45"); R(4, 4, 8, 1, "#3a3020"); R(4, 6, 6, 1, "#3a3020"); break;
    case "altar":
      R(-2, 4, 20, 9, "#8d887c"); R(-2, 4, 20, 2, "#c8c2b0"); R(6, 0, 4, 5, "#c9a24a"); break;
    case "candles":
      R(1, 8, 14, 6, "#3b2a1a");
      for (let k = 2; k < 14; k += 3) { R(k, 4, 2, 4, "#e8e0c8"); R(k, 2, 2, 2, "#ffc66b"); }
      break;
    case "bench":
      for (let r = 0; r < 3; r++) R(-12, r * 5, 40, 3, "#4e3420");
      break;
    case "icon":
      R(4, 1, 8, 10, "#c9a24a"); R(5, 2, 6, 8, "#5a3a2a"); R(7, 3, 2, 6, "#c9a24a"); break;
    case "tomb":
      R(1, 2, 14, 12, "#5a5850"); R(1, 2, 14, 3, "#77756b"); R(7, 6, 2, 6, "#3a3830"); R(5, 8, 6, 2, "#3a3830"); break;
    case "grave":
      for (const [x, y] of [[0, 0], [8, 3], [-6, 8], [4, 10]]) {
        R(x + 2, y, 5, 7, "#5d5b55"); R(x + 2, y, 5, 1, "#79776f"); R(x + 1, y + 7, 7, 4, "#3a3127");
      }
      break;
    case "pit":
      R(-6, 0, 28, 14, "#3a3127"); R(-4, 2, 24, 10, "#0f0c0a"); R(0, 5, 6, 2, "#a89b80"); R(9, 7, 5, 2, "#a89b80"); break;
    case "tree":
      c.strokeStyle = "#1f1813"; c.lineWidth = 2;
      c.beginPath(); c.moveTo(px + 8, py + 16); c.lineTo(px + 8, py + 4); c.lineTo(px + 2, py - 4);
      c.moveTo(px + 8, py + 6); c.lineTo(px + 15, py - 2); c.moveTo(px + 8, py + 4); c.lineTo(px + 9, py - 6); c.stroke();
      break;
    case "stove":
      R(2, 2, 12, 12, "#222"); R(5, 8, 6, 4, "#ff7a2a"); break;
    case "cauldron":
      circle(8, 9, 6.5, "#111"); circle(8, 8, 4.5, "#2f6b45"); circle(6, 7, 1, "#7dff9a"); break;
    case "herbs":
      R(0, 1, 16, 1, "#3b2a1a");
      for (let k = 1; k < 16; k += 3) R(k, 2, 2, 6 + (k % 3), k % 2 ? "#4f6b3a" : "#6b5a3a");
      break;
    case "plants":
      for (let k = 0; k < 12; k++) circle(-8 + (k % 6) * 6, 2 + Math.floor(k / 6) * 9, 2.5, k % 3 ? "#3f5a35" : "#5a3a5a");
      break;
    case "hive":
      R(3, 4, 10, 10, "#7a5a2a"); R(3, 7, 10, 1, "#4a3418"); R(3, 10, 10, 1, "#4a3418"); R(7, 12, 2, 2, "#140c08"); break;
    case "rock":
      circle(8, 9, 6, "#4a4a48"); circle(7, 8, 4, "#5a5a56"); break;
    case "scarecrow":
      R(7, 0, 2, 16, "#3b2a1a"); R(1, 4, 14, 2, "#3b2a1a"); R(4, 5, 8, 6, "#4a3a2a"); circle(8, 2, 3, "#8a7a4a"); break;
    case "stones":
      for (let k = 0; k < 7; k++) {
        const a = (k / 7) * Math.PI * 2;
        const x = 8 + Math.cos(a) * 26, y = 8 + Math.sin(a) * 18;
        R(x - 3, y - 8, 6, 10, "#55534d"); R(x - 3, y - 8, 6, 2, "#6f6c64");
      }
      break;
    case "oven":
      R(0, 0, 16, 16, "#4a4440"); circle(8, 9, 6, "#3a3430"); R(4, 8, 8, 5, "#140c08"); R(5, 10, 6, 2, "#ff7a2a"); break;
    case "shelf":
      R(0, 2, 16, 12, "#3b2a1a"); R(1, 5, 14, 1, "#24180e"); R(1, 9, 14, 1, "#24180e"); circle(4, 4, 1.6, "#b08850"); circle(11, 8, 1.6, "#b08850"); break;
    case "sacks":
      circle(5, 9, 4.5, "#bfb49a"); circle(11, 10, 4.5, "#aca286"); R(4, 3, 2, 3, "#8a8070"); break;
    case "loom":
      R(0, 1, 16, 2, "#4e3420"); R(0, 13, 16, 2, "#4e3420"); R(0, 1, 2, 14, "#4e3420"); R(14, 1, 2, 14, "#4e3420");
      for (let k = 3; k < 14; k += 2) R(k, 3, 1, 10, "#8a8478");
      R(2, 9, 12, 4, "#6b6a74"); break;
    case "wheel":
      c.strokeStyle = "#6e4a2c"; c.lineWidth = 1.5; c.beginPath(); c.arc(px + 8, py + 7, 5.5, 0, Math.PI * 2); c.stroke();
      R(7, 6, 2, 2, "#6e4a2c"); R(3, 12, 10, 2, "#4e3420"); break;
    case "cloth":
      for (let k = 0; k < 4; k++) R(1, 3 + k * 3, 14, 3, k % 2 ? "#cfc8b4" : "#bdb6a2"); break;
    case "sickbeds":
      for (const x of [-10, 2, 14]) { R(x, 0, 9, 14, "#5a4a30"); R(x + 1, 1, 7, 3, "#a89b80"); R(x + 1, 5, 7, 8, "#6f6650"); }
      break;
    case "basin":
      circle(8, 9, 6, "#5a5850"); circle(8, 9, 4, "#25323a"); break;
    case "book":
      R(3, 5, 10, 7, "#5a2222"); R(4, 6, 8, 5, "#d8cfb0"); R(8, 6, 1, 5, "#5a2222"); break;
    case "tubs":
      circle(5, 6, 4.5, "#5a4028"); circle(5, 6, 3, "#4a6070"); circle(12, 11, 4.5, "#5a4028"); circle(12, 11, 3, "#4a6070"); break;
    case "lines":
      R(-8, 3, 32, 1, "#8a7a5a");
      for (const [x, col] of [[-6, "#d8d0bc"], [2, "#bfb6a0"], [10, "#d8d0bc"], [17, "#a89880"]] as [number, string][]) R(x, 4, 6, 8, col);
      break;
    case "idols":
      for (const [x, h] of [[1, 9], [6, 12], [11, 8]] as [number, number][]) {
        R(x, 14 - h, 4, h, "#4a4a3a"); R(x + 1, 15 - h, 1, 1, "#9a9a50"); R(x + 2, 15 - h, 1, 1, "#9a9a50");
      }
      break;
    case "firepit":
      circle(8, 9, 5.5, "#2a2420");
      for (let k = 0; k < 6; k++) { const a = (k / 6) * Math.PI * 2; circle(8 + Math.cos(a) * 5.5, 9 + Math.sin(a) * 5.5, 1.4, "#55534d"); }
      R(6, 7, 4, 4, "#ff7a2a"); R(7, 6, 2, 2, "#ffc66b"); break;
    case "shrine":
      R(1, 10, 14, 4, "#3b2a1a");
      for (const x of [3, 8, 12]) { circle(x, 8, 2.2, "#d8d0b8"); R(x - 1, 8, 1, 1, "#1a1410"); R(x + 0.5, 8, 1, 1, "#1a1410"); }
      break;
    case "thicket":
      for (let k = 0; k < 7; k++) circle(2 + (k * 5) % 13, 4 + Math.floor(k / 3) * 4, 3.5, k % 2 ? "#1f2a17" : "#2a3520");
      for (let k = 0; k < 5; k++) R(1 + k * 3, 2 + (k % 2) * 6, 1, 1, "#5a2a3a");
      break;
    case "reeds":
      for (let k = 0; k < 9; k++) { R(1 + k * 1.7, 2 + (k % 3), 1, 12 - (k % 3), "#4f5a32"); R(1 + k * 1.7, 2 + (k % 3), 1, 2, "#6b5a3a"); }
      break;
    default:
      R(4, 4, 8, 8, "#555");
  }
}
