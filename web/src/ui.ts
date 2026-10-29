// Sidebar: clock, controls, townsfolk list, chronicle feed, and the villager inspector.
import type { AgentDetail, AgentView, Bond, ChatLogEntry, SimEvent, StateMsg } from "./types";

type DetailTab = "overview" | "bonds" | "chat" | "memory" | "prayers";
const GODS: Record<string, { name: string; color: string }> = {
  caedros: { name: "Caedros", color: "#9fb4c8" },
  yevra: { name: "Yevra", color: "#e08fb0" },
  agrath: { name: "Agrath", color: "#e0a050" },
  sevreth: { name: "Sevreth", color: "#7fc490" },
};
const godColor = (id: string) => GODS[id]?.color ?? "var(--muted)";
const CLARITY: Record<string, string> = { none: "unnoticed", doubt: "doubted", suspected: "suspected", certain: "certain" };
const clarityTag = (c?: string) => c ? ` · ${CLARITY[c] ?? c}` : "";
const godName = (id: string) => GODS[id]?.name ?? id;

const DETAIL_TABS: { id: DetailTab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "bonds", label: "Relationships" },
  { id: "chat", label: "Conversations" },
  { id: "memory", label: "Memory" },
  { id: "prayers", label: "Prayers" },
];

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

const esc = (s: string) => s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]!);
const hhmm = (iso: string) => iso.slice(11, 16);
const SPEEDS = [1, 2, 4, 8, 16];

export interface UIHandlers {
  select(id: string | null): void;
  control(c: { paused?: boolean; speed?: number }): void;
}

export class UI {
  private names = new Map<string, AgentView>();
  private feedCount = 0;
  private detailTimer = 0;
  private detailTab: DetailTab = "overview";
  private lastDetail: AgentDetail | null = null;
  selected: string | null = null;

  constructor(h: UIHandlers) {
    $("#pause").onclick = () => h.control({ paused: !this.paused });
    $("#speeds").innerHTML = SPEEDS.map((s) => `<button data-speed="${s}">${s}×</button>`).join("");
    $("#speeds").onclick = (e) => {
      const s = (e.target as HTMLElement).dataset.speed;
      if (s) h.control({ speed: Number(s) });
    };
    $("#tabs").onclick = (e) => {
      const tab = (e.target as HTMLElement).dataset.tab;
      if (!tab) return;
      document.querySelectorAll("#tabs button, .tab").forEach((el) => el.classList.remove("on"));
      $(`[data-tab="${tab}"]`).classList.add("on");
      $(`#${tab}`).classList.add("on");
    };
    $("#folk").onclick = (e) => {
      const row = (e.target as HTMLElement).closest<HTMLElement>("[data-agent]");
      if (row) h.select(row.dataset.agent!);
    };
    $("#detail").onclick = (e) => {
      const target = e.target as HTMLElement;
      if (target.closest(".close")) h.select(null);
      const bond = target.closest<HTMLElement>(".bonds [data-id]");
      if (bond) h.select(bond.dataset.id!);  // follow a relationship to that person
      const dtab = target.closest<HTMLElement>("[data-dtab]");
      if (dtab) { this.detailTab = dtab.dataset.dtab as DetailTab; this.renderDetail(true); }
    };
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && e.target === document.body) { e.preventDefault(); h.control({ paused: !this.paused }); }
      if (e.code === "Escape") h.select(null);
    });
  }

  paused = false;

  state(s: StateMsg) {
    this.paused = s.paused;
    const t = new Date(s.time);
    const day = t.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
    $("#clock").innerHTML = `<span class="time">${hhmm(s.time)}</span><span class="date">${day}, ${s.time.slice(0, 4)}</span>`;
    $("#pause").textContent = s.paused ? "▶" : "❚❚";
    document.querySelectorAll<HTMLElement>("#speeds button").forEach((b) =>
      b.classList.toggle("on", Number(b.dataset.speed) === s.speed));
    const l = s.llm;
    $("#llm").textContent = `mind: ${l.calls} thoughts · ${l.in_flight} thinking now · ${l.avg_latency}s each` +
      (l.failures ? ` · ${l.failures} failed` : "");
    for (const a of s.agents) this.names.set(a.id, a);
    $("#folk").innerHTML = s.agents.map((a) => `
      <div class="folk ${a.id === this.selected ? "sel" : ""}" data-agent="${a.id}">
        <span class="dot" style="background:${a.color}"></span>
        <div><b>${esc(a.name)}</b><small>${esc(this.doing(a))}</small></div>
        <span class="emo">${a.thinking ? "…" : esc(a.emoji || "")}</span>
      </div>`).join("");
  }

  private doing(a: AgentView): string {
    if (a.thinking) return a.thinking;
    if (a.mode === "away") return "away from town";
    if (a.absence === "jailed") return "locked in the cell";
    if (a.mode === "asleep") return "asleep";
    if (a.mode === "talking") return "in conversation";
    if (a.mode === "waiting") return "being approached";
    if (a.mode === "hiding") return "hiding";
    return (a.mode === "walking" ? "→ " : "") + (a.activity ?? "idle");
  }

  events(evs: SimEvent[]) {
    const feed = $("#chronicle");
    const html = evs.map((e) => this.eventHtml(e)).filter(Boolean).join("");
    if (!html) return;
    const stick = feed.scrollTop < 30;
    feed.insertAdjacentHTML("afterbegin", html);
    this.feedCount += evs.length;
    while (feed.childElementCount > 300) feed.lastElementChild!.remove();
    if (stick) feed.scrollTop = 0;
    if (this.selected && evs.some((e) => e.agents.includes(this.selected!))) this.refreshDetail();
  }

  private eventHtml(e: SimEvent): string {
    const who = (id: string) => this.names.get(id);
    const tag = (id: string) => { const a = who(id); return a ? `<b style="color:${a.color}">${esc(a.first)}</b>` : id; };
    switch (e.kind) {
      case "chat": {
        const sp = e.speaker ? tag(e.speaker) : "";
        return `<div class="ev chat"><time>${hhmm(e.time)}</time>${sp}: “${esc(e.text)}”</div>`;
      }
      case "intent":
        return `<div class="ev intent"><time>${hhmm(e.time)}</time>${esc(e.text)}</div>`;
      case "reflection":
        return `<div class="ev refl"><time>${hhmm(e.time)}</time>${tag(e.agents[0])} realises: <i>${esc(e.text)}</i></div>`;
      case "prayer": {
        const god = e.god ?? "";
        const body = e.text.split(": ").slice(1).join(": ");
        return `<div class="ev prayer${e.form === "buff" ? " buff" : ""}" style="--god:${godColor(god)}"><time>${hhmm(e.time)}</time>`
          + `🕯 ${tag(e.agents[0])} prays to <b class="god">${esc(godName(god))}</b> <span class="form">${esc(e.form ?? "")}${clarityTag(e.clarity)}</span>`
          + `<div class="answer">${esc(body)}</div></div>`;
      }
      case "plan":
        return `<div class="ev plan"><time>${hhmm(e.time)}</time>${esc(e.text)}</div>`;
      case "hide":
      case "flee":
      case "watch":
      case "seek":
        return `<div class="ev stealth"><time>${hhmm(e.time)}</time>${esc(e.text)}</div>`;
      case "sighting":
        return `<div class="ev sight"><time>${hhmm(e.time)}</time>⚠ ${esc(e.text)}</div>`;
      case "ally":
      case "warm":
        return `<div class="ev ally"><time>${hhmm(e.time)}</time>${esc(e.text)}</div>`;
      case "spurn":
        return `<div class="ev spurn"><time>${hhmm(e.time)}</time>${esc(e.text)}</div>`;
      case "chat_end":
        return `<div class="ev end"><time>${hhmm(e.time)}</time>${esc(e.text)}</div>`;
      default:
        return "";
    }
  }

  async select(id: string | null) {
    this.selected = id;
    this.detailTab = "overview";
    this.lastDetail = null;
    window.clearInterval(this.detailTimer);
    const panel = $("#detail");
    document.body.classList.toggle("inspecting", !!id);
    if (!id) { panel.hidden = true; return; }
    panel.hidden = false;
    panel.innerHTML = `<p class="muted">…</p>`;
    await this.refreshDetail();
    this.detailTimer = window.setInterval(() => this.refreshDetail(), 4000);
  }

  private async refreshDetail() {
    if (!this.selected) return;
    const res = await fetch(`/api/agents/${this.selected}`);
    if (!res.ok) return;
    const d: AgentDetail = await res.json();
    if (d.id !== this.selected) return;
    this.lastDetail = d;
    this.renderDetail();
  }

  private renderDetail(resetScroll = false) {
    const d = this.lastDetail;
    if (!d) return;
    const panel = $("#detail");
    const scroll = resetScroll ? 0 : panel.scrollTop;
    const body = this.detailTab === "bonds" ? this.renderBonds(d)
      : this.detailTab === "chat" ? this.renderChat(d)
      : this.detailTab === "memory" ? this.renderMemory(d)
      : this.detailTab === "prayers" ? this.renderPrayers(d)
      : this.renderOverview(d);
    panel.innerHTML = `
      <div class="dhead">
        <span class="dot big" style="background:${d.color}"></span>
        <div><h2>${esc(d.name)}</h2><small>${d.age} winters · ${esc(d.traits)}</small></div>
        <button class="close" title="Close (Esc)">✕</button>
      </div>
      <nav class="dtabs">${DETAIL_TABS.map((t) =>
        `<button data-dtab="${t.id}" class="${this.detailTab === t.id ? "on" : ""}">${t.label}</button>`).join("")}</nav>
      <div class="dbody">${body}</div>`;
    panel.scrollTop = scroll;
  }

  private renderOverview(d: AgentDetail): string {
    const now = d.tasks.find((t) => t.activity === d.activity);
    return `
      <p class="now">${d.emoji ? esc(d.emoji) + " " : ""}${esc(d.thinking ?? (d.mode === "asleep" ? "Asleep." : d.activity ?? "Idle."))}
        ${now ? `<small>until ${hhmm(now.end)}</small>` : ""}</p>
      <p class="muted">${esc(d.currently)}</p>
      ${d.drive ? `<p class="drive">${esc(d.drive)}</p>` : ""}
      ${d.states.length ? `<p class="states">${d.states.map((s) =>
        `<span class="state" title="since ${s.since ?? "the start"}">${esc(s.text)}</span>`).join("")}</p>` : ""}
      ${d.pregnancy ? `<p class="pregnant">🤰 Expecting a child${d.pregnancy.father ? ` by <b>${esc(d.pregnancy.father)}</b>` : ""}
        <small>due ${d.pregnancy.due.slice(5)} · ${d.pregnancy.days_left === 0 ? "any day now"
          : `in ${d.pregnancy.days_left} day${d.pregnancy.days_left === 1 ? "" : "s"}`}</small></p>` : ""}
      ${d.favor ? `<p class="favor" style="--god:${godColor(d.favor.god)}">✨ Favor of <b>${esc(d.favor.god_name)}</b>
        <small>until ${hhmm(d.favor.expires)}</small><br>${esc(d.favor.text)}</p>` : ""}
      <h3>Today</h3>
      <ol class="plan">${d.plan.map((b) => {
        const cur = d.activity && now && b.start <= now.start && now.start < b.end;
        return `<li class="${cur ? "cur" : ""}"><time>${hhmm(b.start)}</time>${esc(b.activity)} <small>${esc(b.place_name)}</small></li>`;
      }).join("") || "<li class='muted'>No plan yet.</li>"}</ol>
      ${d.seeking ? `<p class="muted">Creeping after ${esc(d.seeking)}.</p>` : ""}
      ${d.commitments.length ? `<h3>Promises</h3><ul class="mem">${d.commitments.map((c) =>
        `<li><time>${c.date.slice(5)} ${c.time}</time>${esc(c.activity)}</li>`).join("")}</ul>` : ""}`;
  }

  private renderPrayers(d: AgentDetail): string {
    if (!d.prayers.length) return `<p class="muted">Has not prayed yet.</p>`;
    return `<ul class="prayers">${d.prayers.map((p) =>
      `<li style="--god:${godColor(p.god)}"><time>${p.time.slice(5, 10)} ${hhmm(p.time)}</time><b class="god">${esc(godName(p.god))}</b>
        <span class="form">${esc(p.form)}${clarityTag(p.clarity)}</span><div>${esc(p.text)}</div></li>`).join("")}</ul>`;
  }

  private renderBonds(d: AgentDetail): string {
    return `<ul class="bonds">${d.bonds.map((b) => this.bondHtml(b, d.first)).join("") || "<li class='muted'>Knows no one yet.</li>"}</ul>`;
  }

  private renderMemory(d: AgentDetail): string {
    return `
      <h3>Reflections</h3>
      <ul class="mem">${d.reflections.map((m) => this.memHtml(m)).join("") || "<li class='muted'>None yet.</li>"}</ul>
      <h3>Memory stream <small>(${d.importance_acc} toward next reflection)</small></h3>
      <ul class="mem">${d.memories.map((m) => this.memHtml(m)).join("")}</ul>`;
  }

  private renderChat(d: AgentDetail): string {
    if (!d.chat_log.length) return `<p class="muted">No conversations yet.</p>`;
    const groups: ChatLogEntry[][] = [];
    for (const e of d.chat_log) {
      if (!groups.length || groups[groups.length - 1][0].convo !== e.convo) groups.push([]);
      groups[groups.length - 1].push(e);
    }
    return groups.slice().reverse().map((g) => {
      const partner = g.find((e) => e.with_name);
      const head = partner
        ? `<div class="convo-head"><b style="color:${partner.with_color}">${esc(partner.with_name!)}</b> <time>${hhmm(g[0].time)}</time></div>`
        : "";
      const lines = g.map((e) => {
        if (e.kind === "chat_end") return `<div class="convo-end">${esc(e.text)}</div>`;
        const mine = e.speaker === d.id;
        const name = mine ? d.name : (e.speaker_name ?? "");
        const color = mine ? d.color : (e.with_color ?? "var(--muted)");
        return `<div class="ev chat"><time>${hhmm(e.time)}</time><b style="color:${color}">${esc(name)}</b>: “${esc(e.text)}”</div>`;
      }).join("");
      return `<div class="convo">${head}${lines}</div>`;
    }).join("");
  }

  private bondHtml(b: Bond, viewer: string) {
    const tags = [b.protects ? "protects him" : "", b.protected ? "sworn to protect" : ""].filter(Boolean);
    const talks = b.talks ? `${b.talks} talk${b.talks === 1 ? "" : "s"}` : "not yet spoken";
    return `<li data-id="${b.id}"><span class="stance s-${b.stance}">${b.stance}</span>
      <b style="color:${b.color}">${esc(b.name)}</b>${b.kin ? ` <span class="kin">is ${esc(b.kin)} of ${esc(viewer)}</span>` : ""}
      <small>${talks}${tags.length ? " · " + tags.join(" · ") : ""}</small>
      ${b.note ? `<div>${esc(b.note)}</div>` : ""}
      ${b.base ? `<div class="muted">${esc(b.base)}</div>` : ""}</li>`;
  }

  private memHtml(m: AgentDetail["memories"][number]) {
    const bar = "▮".repeat(Math.max(1, Math.round(m.importance / 2)));
    return `<li class="k-${m.kind}"><time>${m.created.slice(5, 10)} ${hhmm(m.created)}</time>
      <span class="kind">${m.kind}</span> <span class="imp" title="importance ${m.importance}">${bar}</span>
      <div>${esc(m.text)}</div></li>`;
  }
}
