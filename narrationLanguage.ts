/**
 * Per-document narration locale. UI stores BCP-47 ids; TTS backends get mapped codes.
 */

export type NarrationLanguageId =
  | "auto"
  | "pt-BR"
  | "en-US"
  | "en-GB"
  | "es-ES"
  | "fr-FR"
  | "it-IT"
  | "de-DE"
  | "ja-JP"
  | "ko-KR"
  | "zh-CN"
  | "ru-RU"
  | "hi-IN";

export type TtsEngineKind = "qwen3" | "kokoro";
export type KokoroBackendKind = "mlx" | "onnx";

export interface NarrationLanguage {
  id: NarrationLanguageId;
  /** Emoji flag (or globe for auto). */
  flag: string;
  /** pt-BR label for the UI. */
  label: string;
  /** Qwen3-TTS `lang_code` / `language` display name. */
  qwenLang: string;
  /** mlx-audio Kokoro single-letter `lang_code`. */
  kokoroMlx: string;
  /** kokoro-onnx `lang` (e.g. en-us, pt-br). */
  kokoroOnnx: string;
  /** False when Kokoro has no dedicated G2P for this locale. */
  kokoroNative: boolean;
}

export const NARRATION_LANGUAGES: NarrationLanguage[] = [
  {
    id: "auto",
    flag: "🌐",
    label: "Automático",
    qwenLang: "Auto",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
  },
  {
    id: "pt-BR",
    flag: "🇧🇷",
    label: "Português (Brasil)",
    qwenLang: "Portuguese",
    kokoroMlx: "p",
    kokoroOnnx: "pt-br",
    kokoroNative: true,
  },
  {
    id: "en-US",
    flag: "🇺🇸",
    label: "Inglês (EUA)",
    qwenLang: "English",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: true,
  },
  {
    id: "en-GB",
    flag: "🇬🇧",
    label: "Inglês (Reino Unido)",
    qwenLang: "English",
    kokoroMlx: "b",
    kokoroOnnx: "en-gb",
    kokoroNative: true,
  },
  {
    id: "es-ES",
    flag: "🇪🇸",
    label: "Espanhol",
    qwenLang: "Spanish",
    kokoroMlx: "e",
    kokoroOnnx: "es",
    kokoroNative: true,
  },
  {
    id: "fr-FR",
    flag: "🇫🇷",
    label: "Francês",
    qwenLang: "French",
    kokoroMlx: "f",
    kokoroOnnx: "fr-fr",
    kokoroNative: true,
  },
  {
    id: "it-IT",
    flag: "🇮🇹",
    label: "Italiano",
    qwenLang: "Italian",
    kokoroMlx: "i",
    kokoroOnnx: "it",
    kokoroNative: true,
  },
  {
    id: "de-DE",
    flag: "🇩🇪",
    label: "Alemão",
    qwenLang: "German",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
  },
  {
    id: "ja-JP",
    flag: "🇯🇵",
    label: "Japonês",
    qwenLang: "Japanese",
    kokoroMlx: "j",
    kokoroOnnx: "ja",
    kokoroNative: true,
  },
  {
    id: "ko-KR",
    flag: "🇰🇷",
    label: "Coreano",
    qwenLang: "Korean",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
  },
  {
    id: "zh-CN",
    flag: "🇨🇳",
    label: "Chinês",
    qwenLang: "Chinese",
    kokoroMlx: "z",
    kokoroOnnx: "zh",
    kokoroNative: true,
  },
  {
    id: "ru-RU",
    flag: "🇷🇺",
    label: "Russo",
    qwenLang: "Russian",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
  },
  {
    id: "hi-IN",
    flag: "🇮🇳",
    label: "Hindi",
    qwenLang: "Auto",
    kokoroMlx: "h",
    kokoroOnnx: "hi",
    kokoroNative: true,
  },
];

const LANGUAGE_BY_ID = new Map(
  NARRATION_LANGUAGES.map((lang) => [lang.id, lang] as const)
);

const LOCALE_PREFIX: Record<string, NarrationLanguageId> = {
  pt: "pt-BR",
  en: "en-US",
  es: "es-ES",
  fr: "fr-FR",
  it: "it-IT",
  de: "de-DE",
  ja: "ja-JP",
  ko: "ko-KR",
  zh: "zh-CN",
  ru: "ru-RU",
  hi: "hi-IN",
};

export function isNarrationLanguageId(value: string): value is NarrationLanguageId {
  return LANGUAGE_BY_ID.has(value as NarrationLanguageId);
}

export function getNarrationLanguage(id: string): NarrationLanguage {
  const resolved = resolveNarrationLanguageId(id);
  const lang = LANGUAGE_BY_ID.get(resolved);

  if (!lang) {
    return LANGUAGE_BY_ID.get("auto") as NarrationLanguage;
  }

  return lang;
}

export function resolveNarrationLanguageId(raw: string | null | undefined): NarrationLanguageId {
  const trimmed = (raw || "").trim();

  if (!trimmed) {
    return "auto";
  }

  if (trimmed.toLowerCase() === "auto") {
    return "auto";
  }

  const normalized = trimmed.replace(/_/g, "-");

  if (isNarrationLanguageId(normalized)) {
    return normalized;
  }

  const lower = normalized.toLowerCase();

  if (lower === "en-gb" || lower === "en-ie") {
    return "en-GB";
  }

  const exact = NARRATION_LANGUAGES.find((lang) => lang.id.toLowerCase() === lower);

  if (exact) {
    return exact.id;
  }

  const byQwen = NARRATION_LANGUAGES.find(
    (lang) => lang.qwenLang.toLowerCase() === lower
  );

  if (byQwen) {
    return byQwen.id;
  }

  const prefix = lower.split("-")[0] || "";
  const mapped = LOCALE_PREFIX[prefix];

  if (mapped) {
    return mapped;
  }

  return "auto";
}

/** Browser/OS locale → catalog id. Falls back to `auto` when unknown. */
export function detectSystemNarrationLanguage(): NarrationLanguageId {
  if (typeof navigator === "undefined") {
    return "auto";
  }

  const locale =
    (Array.isArray(navigator.languages) && navigator.languages[0]) ||
    navigator.language ||
    "";

  return resolveNarrationLanguageId(locale);
}

export function languagePayloadForTts(
  languageId: string,
  engine: TtsEngineKind,
  kokoroBackend: KokoroBackendKind = "mlx"
): string {
  const lang = getNarrationLanguage(languageId);

  if (engine === "qwen3") {
    return lang.qwenLang;
  }

  if (kokoroBackend === "onnx") {
    return lang.kokoroOnnx;
  }

  return lang.kokoroMlx;
}
