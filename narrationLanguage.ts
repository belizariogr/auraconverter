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
  /** Spoken sample for UI preview and Qwen ICL ref_text. */
  previewText: string;
  /**
   * Extra Qwen `instruct` (accent / dialect). Appended to the narrator prompt.
   * Empty when the locale has no extra steering.
   */
  qwenInstruct: string;
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
    previewText:
      "Esta é uma prévia da minha voz para narração de livros. " +
      "Estou falando em tom normal, claro e com um pouco de emoção, " +
      "em português do Brasil — com você, o celular e o ônibus no caminho pela cidade.",
    qwenInstruct: "",
  },
  {
    id: "pt-BR",
    flag: "🇧🇷",
    label: "Português (Brasil)",
    qwenLang: "Portuguese",
    kokoroMlx: "p",
    kokoroOnnx: "pt-br",
    kokoroNative: true,
    // Lexical BR markers (você/celular/ônibus) steer CustomVoice away from European PT.
    previewText:
      "Esta é uma prévia da minha voz para narração de livros. " +
      "Estou falando em tom normal, claro e com um pouco de emoção, " +
      "em português do Brasil — com você, o celular e o ônibus no caminho pela cidade.",
    qwenInstruct:
      "Speak Brazilian Portuguese from Brazil only (português brasileiro / sotaque do Brasil). " +
      "Use Brazilian pronunciation and vocabulary (você, celular, ônibus). " +
      "Do not use a European Portuguese accent from Portugal (tu/telemóvel/autocarro). " +
      "Normal speaking volume; warm audiobook narrator with light emotion — not a whisper.",
  },
  {
    id: "en-US",
    flag: "🇺🇸",
    label: "Inglês (EUA)",
    qwenLang: "English",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: true,
    previewText:
      "This is a preview of my narration voice. " +
      "I am reading at a normal volume, clearly and with a little warmth.",
    qwenInstruct: "",
  },
  {
    id: "en-GB",
    flag: "🇬🇧",
    label: "Inglês (Reino Unido)",
    qwenLang: "English",
    kokoroMlx: "b",
    kokoroOnnx: "en-gb",
    kokoroNative: true,
    previewText:
      "This is a preview of my narration voice. " +
      "I am reading at a normal volume, clearly and with a little warmth.",
    qwenInstruct:
      "Speak British English (UK), not American English. " +
      "Normal speaking volume; warm audiobook narrator with light emotion — not a whisper.",
  },
  {
    id: "es-ES",
    flag: "🇪🇸",
    label: "Espanhol",
    qwenLang: "Spanish",
    kokoroMlx: "e",
    kokoroOnnx: "es",
    kokoroNative: true,
    previewText:
      "Hola. Esta es una muestra de mi voz para narración. " +
      "Leo en un tono normal, claro y con un poco de emoción.",
    qwenInstruct: "",
  },
  {
    id: "fr-FR",
    flag: "🇫🇷",
    label: "Francês",
    qwenLang: "French",
    kokoroMlx: "f",
    kokoroOnnx: "fr-fr",
    kokoroNative: true,
    previewText:
      "Bonjour. Voici un aperçu de ma voix de narration. " +
      "Je lis à volume normal, clairement et avec un peu de chaleur.",
    qwenInstruct: "",
  },
  {
    id: "it-IT",
    flag: "🇮🇹",
    label: "Italiano",
    qwenLang: "Italian",
    kokoroMlx: "i",
    kokoroOnnx: "it",
    kokoroNative: true,
    previewText:
      "Ciao. Questa è un'anteprima della mia voce, che legge con un tono calmo e chiaro.",
    qwenInstruct: "",
  },
  {
    id: "de-DE",
    flag: "🇩🇪",
    label: "Alemão",
    qwenLang: "German",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
    previewText:
      "Hallo. Das ist eine Vorschau meiner Stimme, die in ruhigem und klarem Ton liest.",
    qwenInstruct: "",
  },
  {
    id: "ja-JP",
    flag: "🇯🇵",
    label: "Japonês",
    qwenLang: "Japanese",
    kokoroMlx: "j",
    kokoroOnnx: "ja",
    kokoroNative: true,
    previewText:
      "こんにちは。これは私の声のプレビューです。落ち着いた、はっきりした声で読んでいます。",
    qwenInstruct: "",
  },
  {
    id: "ko-KR",
    flag: "🇰🇷",
    label: "Coreano",
    qwenLang: "Korean",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
    previewText:
      "안녕하세요. 이것은 제 목소리 미리듣기입니다. 차분하고 또렷한 톤으로 읽고 있습니다.",
    qwenInstruct: "",
  },
  {
    id: "zh-CN",
    flag: "🇨🇳",
    label: "Chinês",
    qwenLang: "Chinese",
    kokoroMlx: "z",
    kokoroOnnx: "zh",
    kokoroNative: true,
    previewText: "你好。这是我的声音预览，用平静清晰的语调朗读。",
    qwenInstruct: "",
  },
  {
    id: "ru-RU",
    flag: "🇷🇺",
    label: "Russo",
    qwenLang: "Russian",
    kokoroMlx: "a",
    kokoroOnnx: "en-us",
    kokoroNative: false,
    previewText:
      "Здравствуйте. Это образец моего голоса: я читаю спокойным и ясным тоном.",
    qwenInstruct: "",
  },
  {
    id: "hi-IN",
    flag: "🇮🇳",
    label: "Hindi",
    qwenLang: "Auto",
    kokoroMlx: "h",
    kokoroOnnx: "hi",
    kokoroNative: true,
    previewText:
      "नमस्ते। यह मेरी आवाज़ का पूर्वावलोकन है, शांत और स्पष्ट स्वर में।",
    qwenInstruct: "",
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

/** Spoken sample for UI preview and Qwen ICL (`ref_text` must match the WAV). */
export function previewTextForLanguage(languageId: string): string {
  return getNarrationLanguage(languageId).previewText;
}

/** Append locale accent steering to the stable Qwen narrator instruct. */
export function mergeQwenInstruct(languageId: string, baseInstruct: string): string {
  const extra = getNarrationLanguage(languageId).qwenInstruct.trim();

  if (!extra) {
    return baseInstruct;
  }

  return `${baseInstruct} ${extra}`;
}

/** Disk/memory cache tag: `pt-br`, `en-us`, `auto`, … */
export function voicePreviewCacheTag(languageId: string): string {
  return resolveNarrationLanguageId(languageId).toLowerCase();
}
