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

export type Mode = "asleep" | "walking" | "acting" | "talking" | "waiting" | "hiding";

export interface AgentView {
  id: string;
  name: string;
  first: string;
  color: string;
  x: number;
  y: number;
  steps: [number, number][];
  mode: Mode;
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
    | "watch" | "seek" | "agreement" | "error";
  agents: string[];
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
  talks: number;
  last: string | null;
  protects: boolean;   // the goblin's view: this human has sworn to protect him
  protected: boolean;  // a human's view: they have sworn to protect this goblin
}

export interface AgentDetail extends AgentView {
  age: number;
  traits: string;
  background: string;
  currently: string;
  plan: { start: string; end: string; activity: string; place: string; place_name: string }[];
  tasks: { start: string; end: string; activity: string; place: string; emoji: string }[];
  memories: Memory[];
  reflections: Memory[];
  commitments: { date: string; time: string; activity: string; place: string }[];
  importance_acc: number;
  allies: string[];
  bonds: Bond[];
  seeking: string | null;
}
