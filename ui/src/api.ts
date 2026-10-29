// All data goes through the backend API; the browser never touches database files.
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public action?: string | null) {
    super(message);
  }
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const method = opts.method ?? (opts.body !== undefined ? "POST" : "GET");
  const res = await fetch(`/api${path}`, {
    method,
    credentials: "same-origin",
    headers: { "X-TWB": "1", ...(opts.body !== undefined ? { "Content-Type": "application/json" } : {}) },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  let data: any = null;
  try {
    data = await res.json();
  } catch {
    /* empty body */
  }
  if (!res.ok) {
    const err = new ApiError(res.status, data?.error ?? "error", data?.message ?? res.statusText, data?.action);
    throw err;
  }
  return data as T;
}

export const post = <T = any>(path: string, body: unknown = {}) => api<T>(path, { method: "POST", body });

export type Filter = {
  search?: string;
  view_id?: number;
  review_status?: string;
  processing_state?: string;
  local_date_from?: string;
  local_date_to?: string;
  min_vehicles?: number;
};

export type ResultRow = {
  source_version_id: number;
  filename: string;
  capture_utc: string | null;
  capture_local: string | null;
  view_id: number | null;
  view: string | null;
  vehicle_count: number | null;
  candidate_count: number | null;
  candidate_count_max: number | null;
  confirmed_from_candidates: number;
  processing_state: string;
  review_status: string;
  condition: string | null;
  flags: string[];
};

export type Box = [number, number, number, number];

export type Obj = {
  uid: string;
  origin: "machine" | "human" | "change";
  status: "present" | "removed";
  box: Box;
  cls: string;
  in_region: number | null;
  machine_cls: string | null;
  score: number | null;
  correction: string;
  counted: boolean;
};

export type Candidate = {
  uid: string;
  origin: string;
  kind: string;
  member_ids: number[];
  box: Box;
  score: number | null;
  status: string;
  count_min: number;
  count_max: number;
  alternatives: string[];
  flags: string[];
  cls: string | null;
  band: string | null;
  recurrence: number | null;
};

export type NightBands = {
  far_below: number;
  close_from: number;
  operator_set: boolean;
  width_map: { a: number; b: number; samples: number; source: string } | null;
  rows: { far_until_row: number; close_from_row: number } | null;
};

export type Light = { id: number; box: Box; cx: number; cy: number; area: number };
export type Region = { uid: string; kind: "include" | "exclude"; name: string | null; polygon: [number, number][] };

// SAM 3.1: finds and outlines things named in words (vehicles, roads), on the CPU, offline.
export type Sam3Status = {
  installed: boolean; runtime: boolean; loaded: boolean; reason: string | null; memory_note?: string | null;
  pack: { name: string; version: string; license?: string } | null;
  suggestions?: Record<"include" | "exclude", string[]>;
  vehicle_prompts?: Record<string, string[]>;
};
export type AreaProposal = { prompt: string; kind: "include" | "exclude"; polygon: [number, number][]; area_share: number; score: number };
export type AreaTask = {
  id: string; view_id: number; version_id: number; prompts: string[]; kind: "include" | "exclude";
  state: "running" | "finished" | "failed"; stage: string; error: string | null; elapsed_s: number;
  proposals: AreaProposal[]; size: [number, number] | [];
};

export type Item = {
  id: number;
  file: string;
  sha256: string;
  capture_utc: string | null;
  capture_precision: string | null;
  local: { time: string; abbreviation: string; dst: boolean } | null;
  zone: string;
  width: number | null;
  height: number | null;
  view: { id: number | null; name: string | null; regions_applied: boolean | null };
  view_assignment_history: { view_id: number | null; method: string; author: string; reason: string; created_at: string }[];
  condition: { condition: string; flags: string } | null;
  processing_state: string;
  error: string | null;
  review_status: string;
  empty_background: boolean;
  objects: Obj[];
  candidates: Candidate[];
  lights: Light[];
  bands: NightBands | null;
  regions: Region[];
  position: { index: number | null; total: number; prev: number | null; next: number | null };
};

export type Job = {
  kind: string;
  state: string;
  stage: string;
  done: number;
  total: number;
  elapsed_s: number;
  eta_s: number | null;
  eta_basis: string;
  resource_usage: { process_cpu_percent: number; process_rss_bytes: number; system_memory_available_bytes: number; system_memory_total_bytes: number } | null;
  error: string | null;
  result?: { search_run?: number };
};

export function fmtDuration(s: number | null | undefined): string {
  if (s == null) return "–";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h} h ${m} min` : m ? `${m} min ${sec} s` : `${sec} s`;
}

export const REVIEW_LABEL: Record<string, string> = {
  unreviewed: "Unreviewed",
  approved: "Approved",
  bulk_approved_unreviewed: "Bulk approved (not reviewed)",
};

export type BackgroundCondition = {
  images: number[];
  background_id: number | null;
  built_from: number[];
  up_to_date: boolean;
  empty_check: { source_version_id: number; changes: number }[];
  area: { source: "regions" | "sam" | "whole_frame"; share: number } | null;
};
export type Backgrounds = {
  view_id: number;
  conditions: Record<"day" | "dusk" | "night", BackgroundCondition>;
  fallback: Record<string, string[]>;
  suggestions: { source_version_id: number; condition: string; vehicles: number; night_candidates: number; marked: boolean }[];
  busy: boolean;
};
