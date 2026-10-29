export interface Area {
  id: string;
  name: string;
  indoor: boolean;
  public: boolean;
  tiles: [number, number][];
}

export interface WorldObject {
  id: string;
  name: string;
  area: string;
  x: number;
  y: number;
  kind: string;
  state: string;
}

export interface Light {
  x: number;
  y: number;
  radius: number;
  color: string;
  flicker: boolean;
}

export interface WorldData {
  width: number;
  height: number;
  tiles: number[];
  tileNames: Record<number, string>;
  areas: Area[];
  objects: WorldObject[];
  lights: Light[];
}

export type Mode = "asleep" | "walking" | "acting" | "talking" | "waiting" | "hiding" | "sex" | "away";

export interface AgentView {
  id: string;
  name: string;
  first: string;
  color: string;
  x: number;
  y: number;
  steps: [number, number][];
  mode: Mode;
  absence: "away" | "jailed" | null;  // out of town, or locked in the watchhouse cell
  thinking: string | null;
  activity: string | null;
  emoji: string;
  place: string | null;
  convo: string | null;
  bubble: string | null;
  species: string;
  allies: string[];
}

export interface SimEvent {
  kind: "plan" | "chat" | "chat_start" | "chat_end" | "intent" | "reflection" | "hide" | "flee" | "sighting" | "ally" | "warm" | "spurn"
    | "watch" | "seek" | "agreement" | "prayer" | "error";
  agents: string[];
  god?: string;
  form?: string;
  clarity?: string;
  text: string;
  time: string;
  speaker?: string;
  transcript?: [string, string][];
}

export interface LLMStats {
  calls: number;
  failures: number;
  repairs: number;
  in_flight: number;
  avg_latency: number;
}

export interface StateMsg {
  time: string;
  paused: boolean;
  speed: number;
  agents: AgentView[];
  objects: Record<string, string>;
  llm: LLMStats;
}

export type ServerMsg =
  | ({ type: "init"; world: WorldData; events: SimEvent[] } & StateMsg)
  | ({ type: "tick" } & StateMsg)
  | { type: "event"; event: SimEvent };

export interface Memory {
  id: number;
  kind: string;
  text: string;
  created: string;
  importance: number;
  evidence: number[];
}

export type Stance = "fear" | "dislike" | "wary" | "neutral" | "fond" | "trust" | "close" | "love";

export interface Bond {
  id: string;
  name: string;
  color: string;
  stance: Stance;
  note: string;        // what the model made of them after the last talk
  base: string;        // what the persona knew of them from the start
  kin: string | null;  // what they are to this agent when family: "mother", "niece", "brother", ...
  talks: number;
  last: string | null;
  protects: boolean;   // the goblin's view: this human has sworn to protect him
  protected: boolean;  // a human's view: they have sworn to protect this goblin
}

export interface ChatLogEntry {
  time: string;
  kind: "chat" | "chat_end";
  text: string;
  convo: string | null;
  speaker: string | null;
  speaker_name: string | null;
  with: string | null;
  with_name: string | null;
  with_color: string | null;
}

export interface AgentDetail extends AgentView {
  age: number;
  traits: string;
  history: { text: string; since: string }[];
  states: { text: string; since: string | null }[];
  background: string;
  currently: string;
  drive: string;
  favor: { god: string; god_name: string; text: string; expires: string; clarity?: string } | null;
  prayers: { time: string; god: string; form: string; clarity: string; text: string }[];
  plan: { start: string; end: string; activity: string; place: string; place_name: string }[];
  tasks: { start: string; end: string; activity: string; place: string; emoji: string }[];
  memories: Memory[];
  reflections: Memory[];
  commitments: { date: string; time: string; activity: string; place: string }[];
  importance_acc: number;
  allies: string[];
  bonds: Bond[];
  seeking: string | null;
  pregnancy: { father: string | null; due: string; days_left: number; days_total: number } | null;
  chat_log: ChatLogEntry[];
}
