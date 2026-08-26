/**
 * Active TTS engine preference (qwen3 | kokoro), persisted under AURA_DATA_DIR.
 */
import fs from "fs";
import path from "path";

export type TtsEngineId = "qwen3" | "kokoro";
/** Kokoro ONNX Runtime target: force CPU, or prefer GPU EP when available. */
export type KokoroDeviceId = "cpu" | "gpu";
/** Kokoro implementation. ONNX is offered alongside MLX on macOS. */
export type KokoroBackendId = "mlx" | "onnx";

const ENGINE_FILE = "tts-engine.json";

type EngineFile = {
  engine?: string;
  kokoroDevice?: string;
  kokoroBackend?: string;
  updatedAt?: string;
};

export function isTtsEngineId(value: unknown): value is TtsEngineId {
  return value === "qwen3" || value === "kokoro";
}

export function isKokoroDeviceId(value: unknown): value is KokoroDeviceId {
  return value === "cpu" || value === "gpu";
}

export function isKokoroBackendId(value: unknown): value is KokoroBackendId {
  return value === "mlx" || value === "onnx";
}

export function defaultKokoroBackend(platform = process.platform): KokoroBackendId {
  return platform === "darwin" ? "mlx" : "onnx";
}

export function ttsEnginePath(auraDataDir: string): string {
  return path.join(auraDataDir, ENGINE_FILE);
}

function readEngineFile(auraDataDir: string): EngineFile {
  const file = ttsEnginePath(auraDataDir);
  try {
    if (fs.existsSync(file)) {
      return JSON.parse(fs.readFileSync(file, "utf8")) as EngineFile;
    }
  } catch {
    // ignore corrupt file
  }
  return {};
}

function writeEngineFile(auraDataDir: string, patch: EngineFile): void {
  fs.mkdirSync(auraDataDir, { recursive: true });
  const prev = readEngineFile(auraDataDir);
  const next: EngineFile = {
    ...prev,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  if (!isTtsEngineId(next.engine)) next.engine = "qwen3";
  if (!isKokoroDeviceId(next.kokoroDevice)) next.kokoroDevice = "gpu";
  if (!isKokoroBackendId(next.kokoroBackend)) {
    next.kokoroBackend = defaultKokoroBackend();
  }
  fs.writeFileSync(ttsEnginePath(auraDataDir), JSON.stringify(next, null, 2));
}

export function readTtsEngine(auraDataDir: string): TtsEngineId {
  const raw = readEngineFile(auraDataDir);
  return isTtsEngineId(raw.engine) ? raw.engine : "qwen3";
}

export function writeTtsEngine(auraDataDir: string, engine: TtsEngineId): void {
  writeEngineFile(auraDataDir, { engine });
}

export function readKokoroDevice(auraDataDir: string): KokoroDeviceId {
  const raw = readEngineFile(auraDataDir);
  return isKokoroDeviceId(raw.kokoroDevice) ? raw.kokoroDevice : "gpu";
}

export function writeKokoroDevice(auraDataDir: string, device: KokoroDeviceId): void {
  writeEngineFile(auraDataDir, { kokoroDevice: device });
}

export function readKokoroBackend(auraDataDir: string): KokoroBackendId {
  const raw = readEngineFile(auraDataDir);
  return isKokoroBackendId(raw.kokoroBackend)
    ? raw.kokoroBackend
    : defaultKokoroBackend();
}

export function writeKokoroBackend(
  auraDataDir: string,
  backend: KokoroBackendId
): void {
  writeEngineFile(auraDataDir, { kokoroBackend: backend });
}

export type VoiceLocale = "en-us" | "en-gb" | "pt-br";

export type VoiceCatalogEntry = {
  id: string;
  name: string;
  gender: "Feminino" | "Masculino";
  description: string;
  icon: string;
  /** Overall grade from hexgrad VOICES.md (training data quality/quantity). */
  grade?: string;
  /** Voice language family for UI prioritization. */
  locale?: VoiceLocale;
};

/** Sort key: A best → F worst; missing grade last. */
export function voiceGradeSortKey(grade: string | undefined): number {
  if (!grade) return 1000;
  const letter = grade.charAt(0).toUpperCase();
  const base = { A: 0, B: 10, C: 20, D: 30, E: 40, F: 50 }[letter];
  if (base == null) return 900;
  if (grade.endsWith("+")) return base - 1;
  if (grade.endsWith("-")) return base + 1;
  return base;
}

/** Friendly Kokoro catalog: curated EN + PT-BR with official Overall Grades. */
export const KOKORO_VOICES: VoiceCatalogEntry[] = [
  {
    id: "af_heart",
    name: "Heart",
    gender: "Feminino",
    description: "Voz feminina clara e natural (EN-US).",
    icon: "👩",
    grade: "A",
    locale: "en-us",
  },
  {
    id: "af_bella",
    name: "Bella",
    gender: "Feminino",
    description: "Presença expressiva e envolvente (EN-US).",
    icon: "🧑",
    grade: "A-",
    locale: "en-us",
  },
  {
    id: "bf_emma",
    name: "Emma",
    gender: "Feminino",
    description: "Britânica, boa dicção para leituras longas (EN-GB).",
    icon: "👩‍🦰",
    grade: "B-",
    locale: "en-gb",
  },
  {
    id: "af_nicole",
    name: "Nicole",
    gender: "Feminino",
    description: "Dicção limpa, boa para diálogos (EN-US).",
    icon: "👩‍🎤",
    grade: "B-",
    locale: "en-us",
  },
  {
    id: "af_sarah",
    name: "Sarah",
    gender: "Feminino",
    description: "Timbre suave para leituras longas (EN-US).",
    icon: "👩‍🦰",
    grade: "C+",
    locale: "en-us",
  },
  {
    id: "pf_dora",
    name: "Dora",
    gender: "Feminino",
    description: "Português brasileiro — narração natural (PT-BR).",
    icon: "👩",
    locale: "pt-br",
  },
  {
    id: "am_fenrir",
    name: "Fenrir",
    gender: "Masculino",
    description: "Presença mais grave e marcada (EN-US).",
    icon: "🧑‍💼",
    grade: "C+",
    locale: "en-us",
  },
  {
    id: "am_michael",
    name: "Michael",
    gender: "Masculino",
    description: "Narração sólida e pausada (EN-US).",
    icon: "🧔",
    grade: "C+",
    locale: "en-us",
  },
  {
    id: "am_puck",
    name: "Puck",
    gender: "Masculino",
    description: "Tom mais leve e animado (EN-US).",
    icon: "👨‍🏫",
    grade: "C+",
    locale: "en-us",
  },
  {
    id: "pm_alex",
    name: "Alex",
    gender: "Masculino",
    description: "Português brasileiro — tom claro (PT-BR).",
    icon: "👨",
    locale: "pt-br",
  },
  {
    id: "pm_santa",
    name: "Santa",
    gender: "Masculino",
    description: "Português brasileiro — presença madura (PT-BR).",
    icon: "🧔",
    locale: "pt-br",
  },
  {
    id: "am_adam",
    name: "Adam",
    gender: "Masculino",
    description: "EN-US — dados de treino fracos (nota baixa).",
    icon: "👨",
    grade: "F+",
    locale: "en-us",
  },
];

export const QWEN_VOICES: VoiceCatalogEntry[] = [
  {
    id: "Vivian",
    name: "Vivian",
    gender: "Feminino",
    description: "Narração clara e calorosa — boa para romances e não-ficção.",
    icon: "👩",
  },
  {
    id: "Serena",
    name: "Serena",
    gender: "Feminino",
    description: "Timbre suave, adequada para leituras longas.",
    icon: "👩‍🦰",
  },
  {
    id: "Sohee",
    name: "Sohee",
    gender: "Feminino",
    description: "Voz expressiva e natural.",
    icon: "🧑",
  },
  {
    id: "Ono_Anna",
    name: "Ono Anna",
    gender: "Feminino",
    description: "Dicção limpa, boa para diálogos.",
    icon: "👩‍🎤",
  },
  {
    id: "Ryan",
    name: "Ryan",
    gender: "Masculino",
    description: "Tom sereno e estável para capítulos longos.",
    icon: "👨",
  },
  {
    id: "Aiden",
    name: "Aiden",
    gender: "Masculino",
    description: "Presença mais animada e envolvente.",
    icon: "🧑‍💼",
  },
  {
    id: "Eric",
    name: "Eric",
    gender: "Masculino",
    description: "Dicção formal, boa para textos técnicos.",
    icon: "👨‍🏫",
  },
  {
    id: "Dylan",
    name: "Dylan",
    gender: "Masculino",
    description: "Tom sólido para narrativa geral.",
    icon: "🧔",
  },
  {
    id: "Uncle_Fu",
    name: "Uncle Fu",
    gender: "Masculino",
    description: "Timbre maduro e pausado.",
    icon: "🧓",
  },
];

/** Prefer voices matching narration locale (e.g. pt-br), then by grade. */
export function sortVoicesForLanguage(
  voices: VoiceCatalogEntry[],
  languageId: string
): VoiceCatalogEntry[] {
  const preferPt = languageId === "pt-br" || languageId.startsWith("pt");
  return [...voices].sort((a, b) => {
    const aMatch = preferPt ? a.locale === "pt-br" : a.locale !== "pt-br";
    const bMatch = preferPt ? b.locale === "pt-br" : b.locale !== "pt-br";
    if (aMatch !== bMatch) return aMatch ? -1 : 1;
    const g = voiceGradeSortKey(a.grade) - voiceGradeSortKey(b.grade);
    if (g !== 0) return g;
    return a.name.localeCompare(b.name, "pt");
  });
}

export function voicesForEngine(engine: TtsEngineId) {
  return engine === "kokoro" ? KOKORO_VOICES : QWEN_VOICES;
}
