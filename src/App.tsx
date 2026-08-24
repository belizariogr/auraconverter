import React, { useState, useRef, useEffect, useCallback } from "react";
import { 
  FileText, 
  Play, 
  Pause, 
  Download, 
  Volume2, 
  VolumeX, 
  Sparkles, 
  Trash2, 
  BookOpen, 
  RotateCcw, 
  RotateCw, 
  Music, 
  AlertCircle,
  Clock,
  ArrowLeft,
  CheckCircle2,
  FileAudio,
  Loader2,
  Book,
  Square,
  Plus,
  Image as ImageIcon,
  ArrowRightLeft,
  Mic2,
  Languages,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import {
  ExtractCoverPanel,
  Mp3ToM4bPanel,
  M4bToMp3Panel,
} from "./ModePanels";
import {
  NARRATION_LANGUAGES,
  detectSystemNarrationLanguage,
  getNarrationLanguage,
  resolveNarrationLanguageId,
  type NarrationLanguageId,
} from "../narrationLanguage.ts";

type AppMode = "narrate" | "extract-cover" | "mp3-to-m4b" | "m4b-to-mp3";
type OutputFormat = "mp3" | "m4b";

// Voice catalog loaded from active TTS engine
interface Voice {
  id: string;
  name: string;
  gender: "Feminino" | "Masculino";
  description: string;
  icon: string;
}

interface EpubChapter {
  index: number;
  id: string;
  title: string;
}

interface ChapterPreview {
  title: string;
  text: string;
  index: number;
  total: number;
}

interface DocItem {
  id: string;
  file: File;
  fileBase64: string;
  fileType: "pdf" | "epub";
  startPage: number;
  endPage: string;
  /** PDF cover page (1-based). Ignored when exportCover is false. */
  coverPage: string;
  /** When false, skip cover JPEG / M4B artwork. */
  exportCover: boolean;
  pdfPageCount: number | null;
  epubChapters: EpubChapter[];
  docInfoMessage: string;
  docInfoLoading: boolean;
  editableText: string;
  extractedText: string;
  pagesNarrated: string;
  audioBase64: string | null;
  audioUrl: string | null;
  audioFormat: OutputFormat;
  coverPreviewUrl: string | null;
  startPreviewUrl: string | null;
  endPreviewUrl: string | null;
  /** Persisted JPEG base64 for faster restore / re-render. */
  coverPreviewBase64: string | null;
  startPreviewBase64: string | null;
  endPreviewBase64: string | null;
  /** EPUB section previews (text cards; no canvas required). */
  startChapterPreview: ChapterPreview | null;
  endChapterPreview: ChapterPreview | null;
  /** Cached narration block progress (resume after cancel). */
  narrationProgress: { completed: number; total: number } | null;
  /** Hash of the text that produced the cached chunks — invalidate on edit. */
  narrationTextHash: string | null;
  /** BCP-47 (or `auto`) for TTS; defaults to the OS locale. */
  narrationLanguage: NarrationLanguageId;
}

type PagePreviewKey = "start" | "end" | "cover";

function createDocId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

/** Stable hash of narration text — used to invalidate chunk cache on edits. */
function hashNarrationText(text: string, language = ""): string {
  const body = text.replace(/\r\n/g, "\n").trim();
  const normalized = language ? `${language}\n${body}` : body;
  let h = 2166136261;
  for (let i = 0; i < normalized.length; i++) {
    h ^= normalized.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }

  return `${normalized.length.toString(36)}_${(h >>> 0).toString(16).padStart(8, "0")}`;
}

function narrationHashesMatch(
  stored: string | null | undefined,
  text: string,
  language: string
): boolean {
  if (!stored) {
    return false;
  }

  if (stored === hashNarrationText(text, language)) {
    return true;
  }

  return stored === hashNarrationText(text, "");
}

function isPdfOrEpub(file: File): boolean {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return (
    file.type === "application/pdf" ||
    extension === "pdf" ||
    extension === "epub" ||
    file.type === "application/epub+zip"
  );
}

function detectFileType(file: File): "pdf" | "epub" {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return extension === "epub" || file.type === "application/epub+zip" ? "epub" : "pdf";
}

const FALLBACK_QWEN_VOICES: Voice[] = [
  { id: "Vivian", name: "Vivian", gender: "Feminino", description: "Narração clara e calorosa — boa para romances e não-ficção.", icon: "👩" },
  { id: "Serena", name: "Serena", gender: "Feminino", description: "Timbre suave, adequada para leituras longas.", icon: "👩‍🦰" },
  { id: "Sohee", name: "Sohee", gender: "Feminino", description: "Voz expressiva e natural.", icon: "🧑" },
  { id: "Ono_Anna", name: "Ono Anna", gender: "Feminino", description: "Dicção limpa, boa para diálogos.", icon: "👩‍🎤" },
  { id: "Ryan", name: "Ryan", gender: "Masculino", description: "Tom sereno e estável para capítulos longos.", icon: "👨" },
  { id: "Aiden", name: "Aiden", gender: "Masculino", description: "Presença mais animada e envolvente.", icon: "🧑‍💼" },
  { id: "Eric", name: "Eric", gender: "Masculino", description: "Dicção formal, boa para textos técnicos.", icon: "👨‍🏫" },
  { id: "Dylan", name: "Dylan", gender: "Masculino", description: "Tom sólido para narrativa geral.", icon: "🧔" },
  { id: "Uncle_Fu", name: "Uncle Fu", gender: "Masculino", description: "Timbre maduro e pausado.", icon: "🧓" },
];

const VOICE_STORAGE_KEY = "aura-reader-voice";
const ENGINE_VOICE_STORAGE_KEY = "aura-reader-voice-by-engine";
const DOCS_DB_NAME = "aura-reader";
const DOCS_DB_VERSION = 1;
const DOCS_STORE = "session";
const DOCS_STORAGE_KEY = "documents";

interface PersistedDoc {
  id: string;
  fileName: string;
  fileSize: number;
  fileMime: string;
  fileBase64: string;
  fileType: "pdf" | "epub";
  startPage: number;
  endPage: string;
  coverPage: string;
  exportCover: boolean;
  pdfPageCount: number | null;
  epubChapters: EpubChapter[];
  docInfoMessage: string;
  editableText: string;
  extractedText: string;
  pagesNarrated: string;
  audioFormat: OutputFormat;
  coverPreviewBase64: string | null;
  startPreviewBase64: string | null;
  endPreviewBase64: string | null;
  startChapterPreview: ChapterPreview | null;
  endChapterPreview: ChapterPreview | null;
  narrationProgress: { completed: number; total: number } | null;
  narrationTextHash: string | null;
  narrationLanguage?: string;
}

interface PersistedDocumentsState {
  activeDocId: string | null;
  outputFormat?: OutputFormat;
  docs: PersistedDoc[];
}

function loadSavedVoice(voices: Voice[], engine: string): string {
  try {
    const byEngine = JSON.parse(localStorage.getItem(ENGINE_VOICE_STORAGE_KEY) || "{}") as Record<
      string,
      string
    >;
    if (byEngine[engine] && voices.some((v) => v.id === byEngine[engine])) {
      return byEngine[engine];
    }
    const saved = localStorage.getItem(VOICE_STORAGE_KEY);
    if (saved && voices.some((v) => v.id === saved)) return saved;
    if (saved === "Ethan") return voices.find((v) => v.id === "Eric")?.id || voices[0]?.id || "Vivian";
    if (saved === "Chelsie") return voices.find((v) => v.id === "Sohee")?.id || voices[0]?.id || "Vivian";
  } catch {
    // ignore
  }
  return voices[0]?.id || (engine === "kokoro" ? "af_heart" : "Vivian");
}

function persistVoice(engine: string, voiceId: string) {
  try {
    localStorage.setItem(VOICE_STORAGE_KEY, voiceId);
    const byEngine = JSON.parse(localStorage.getItem(ENGINE_VOICE_STORAGE_KEY) || "{}") as Record<
      string,
      string
    >;
    byEngine[engine] = voiceId;
    localStorage.setItem(ENGINE_VOICE_STORAGE_KEY, JSON.stringify(byEngine));
  } catch {
    // ignore
  }
}

function stripBreakTagsForQwen(
  text: string,
  engine: "qwen3" | "kokoro"
): string {
  if (engine !== "qwen3" || !text) {
    return text;
  }

  return text
    .replace(/<break\s+time="[\d.]+s"\s*\/?\s*>/gi, "\n\n\n")
    .replace(/\n{4,}/g, "\n\n\n");
}

function openDocsDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DOCS_DB_NAME, DOCS_DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(DOCS_STORE)) {
        db.createObjectStore(DOCS_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("Falha ao abrir IndexedDB"));
  });
}

async function loadPersistedDocuments(): Promise<PersistedDocumentsState | null> {
  try {
    const db = await openDocsDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(DOCS_STORE, "readonly");
      const req = tx.objectStore(DOCS_STORE).get(DOCS_STORAGE_KEY);
      req.onsuccess = () => {
        const value = req.result as PersistedDocumentsState | undefined;
        resolve(value?.docs?.length ? value : null);
      };
      req.onerror = () => reject(req.error);
      tx.oncomplete = () => db.close();
    });
  } catch {
    return null;
  }
}

async function persistDocumentsState(state: PersistedDocumentsState): Promise<void> {
  try {
    const db = await openDocsDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(DOCS_STORE, "readwrite");
      tx.objectStore(DOCS_STORE).put(state, DOCS_STORAGE_KEY);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // ignore quota / private mode failures
  }
}

async function clearPersistedDocuments(): Promise<void> {
  try {
    const db = await openDocsDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(DOCS_STORE, "readwrite");
      tx.objectStore(DOCS_STORE).delete(DOCS_STORAGE_KEY);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // ignore
  }
}

function base64ToFile(base64: string, fileName: string, mime: string): Promise<File> {
  return fetch(`data:${mime || "application/octet-stream"};base64,${base64}`)
    .then((res) => res.blob())
    .then((blob) => new File([blob], fileName, { type: mime || "application/octet-stream" }));
}

function base64JpegToObjectUrl(base64: string | null | undefined): string | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
  } catch {
    return null;
  }
}

function serializeDoc(doc: DocItem): PersistedDoc | null {
  if (!doc.fileBase64) return null;
  return {
    id: doc.id,
    fileName: doc.file.name,
    fileSize: doc.file.size,
    fileMime: doc.file.type,
    fileBase64: doc.fileBase64,
    fileType: doc.fileType,
    startPage: doc.startPage,
    endPage: doc.endPage,
    coverPage: doc.coverPage,
    exportCover: doc.exportCover,
    pdfPageCount: doc.pdfPageCount,
    epubChapters: doc.epubChapters,
    docInfoMessage: doc.docInfoMessage,
    editableText: doc.editableText,
    extractedText: doc.extractedText,
    pagesNarrated: doc.pagesNarrated,
    audioFormat: doc.audioFormat,
    coverPreviewBase64: doc.coverPreviewBase64,
    startPreviewBase64: doc.startPreviewBase64,
    endPreviewBase64: doc.endPreviewBase64,
    startChapterPreview: doc.startChapterPreview,
    endChapterPreview: doc.endChapterPreview,
    narrationProgress: doc.narrationProgress,
    narrationTextHash: doc.narrationTextHash,
    narrationLanguage: doc.narrationLanguage,
  };
}

async function restoreDoc(saved: PersistedDoc): Promise<DocItem> {
  const mime =
    saved.fileMime ||
    (saved.fileType === "epub" ? "application/epub+zip" : "application/pdf");
  const file = await base64ToFile(saved.fileBase64, saved.fileName, mime);
  return {
    id: saved.id,
    file,
    fileBase64: saved.fileBase64,
    fileType: saved.fileType,
    startPage: saved.startPage,
    endPage: saved.endPage,
    coverPage: saved.coverPage,
    exportCover: saved.exportCover,
    pdfPageCount: saved.pdfPageCount,
    epubChapters: saved.epubChapters ?? [],
    docInfoMessage: saved.docInfoMessage || "",
    docInfoLoading: false,
    editableText: saved.editableText || "",
    extractedText: saved.extractedText || "",
    pagesNarrated: saved.pagesNarrated || "",
    audioBase64: null,
    audioUrl: null,
    audioFormat: saved.audioFormat || "mp3",
    coverPreviewBase64: saved.coverPreviewBase64 ?? null,
    startPreviewBase64: saved.startPreviewBase64 ?? null,
    endPreviewBase64: saved.endPreviewBase64 ?? null,
    coverPreviewUrl: base64JpegToObjectUrl(saved.coverPreviewBase64),
    startPreviewUrl: base64JpegToObjectUrl(saved.startPreviewBase64),
    endPreviewUrl: base64JpegToObjectUrl(saved.endPreviewBase64),
    startChapterPreview: saved.startChapterPreview ?? null,
    endChapterPreview: saved.endChapterPreview ?? null,
    narrationProgress: saved.narrationProgress ?? null,
    narrationTextHash: saved.narrationTextHash ?? null,
    narrationLanguage: resolveNarrationLanguageId(
      saved.narrationLanguage || detectSystemNarrationLanguage()
    ),
  };
}

async function fetchChunkCacheStatus(
  docId: string
): Promise<{ completed: number; total: number } | null> {
  try {
    const res = await fetch(`/api/chunk-cache/${encodeURIComponent(docId)}`);
    if (!res.ok) return null;
    const body = await res.json();
    if (!body?.exists) return null;
    const completed = Number(body.completed) || 0;
    const total = Number(body.total) || 0;
    if (completed <= 0 || total <= 0) return null;
    return { completed, total };
  } catch {
    return null;
  }
}

async function clearChunkCacheOnServer(docId: string): Promise<void> {
  try {
    await fetch(`/api/chunk-cache/${encodeURIComponent(docId)}`, { method: "DELETE" });
  } catch {
    // ignore
  }
}

export default function App({ onManageModels }: { onManageModels?: () => void }) {
  const [appMode, setAppMode] = useState<AppMode>("narrate");
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("mp3");

  // Multi-document state
  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [reviewDocId, setReviewDocId] = useState<string | null>(null);
  const [resultDocId, setResultDocId] = useState<string | null>(null);
  const [docsHydrated, setDocsHydrated] = useState(false);
  const [isRestoringSession, setIsRestoringSession] = useState(false);
  const [restorePercent, setRestorePercent] = useState(0);
  const [batchDone, setBatchDone] = useState(false);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [previewLoadingKeys, setPreviewLoadingKeys] = useState<
    Partial<Record<PagePreviewKey, boolean>>
  >({});
  const [coverPreviewMessage, setCoverPreviewMessage] = useState<string | null>(null);
  const previewDepsRef = useRef<{
    id: string;
    fileBase64: string;
    fileType: "pdf" | "epub";
    startPage: number;
    endPage: string;
    coverPage: string;
    exportCover: boolean;
  } | null>(null);

  // Shared configuration
  const [voices, setVoices] = useState<Voice[]>(FALLBACK_QWEN_VOICES);
  const [ttsEngine, setTtsEngine] = useState<"qwen3" | "kokoro">("qwen3");
  const [kokoroDevice, setKokoroDevice] = useState<"cpu" | "gpu">("gpu");
  const [kokoroBackend, setKokoroBackend] = useState<"mlx" | "onnx">("mlx");
  const [selectedVoice, setSelectedVoice] = useState<string>("Vivian");

  const completedDocs = documents.filter((d) => d.audioUrl);
  const activeDoc = documents.find((d) => d.id === activeDocId) ?? documents[0] ?? null;
  const previewLanguageId =
    activeDoc?.narrationLanguage ?? detectSystemNarrationLanguage();
  const previewLanguage = getNarrationLanguage(previewLanguageId);
  const reviewDoc = documents.find((d) => d.id === reviewDocId) ?? documents[0] ?? null;
  const resultDoc =
    documents.find((d) => d.id === resultDocId) ?? completedDocs[0] ?? null;
  const anyDocInfoLoading = documents.some((d) => d.docInfoLoading);
  const allDocsReady = documents.length > 0 && documents.every((d) => d.fileBase64 && !d.docInfoLoading);
  const hasResults = batchDone || completedDocs.length > 0;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/tts-engine");
        const body = await res.json();
        if (cancelled || !res.ok) return;
        const engine = body.engine === "kokoro" ? "kokoro" : "qwen3";
        const list = Array.isArray(body.voices) && body.voices.length
          ? (body.voices as Voice[])
          : FALLBACK_QWEN_VOICES;
        setTtsEngine(engine);
        if (body.kokoroDevice === "cpu" || body.kokoroDevice === "gpu") {
          setKokoroDevice(body.kokoroDevice);
        }
        if (body.kokoroBackend === "mlx" || body.kokoroBackend === "onnx") {
          setKokoroBackend(body.kokoroBackend);
        }
        setVoices(list);
        setSelectedVoice((prev) => {
          if (list.some((v) => v.id === prev)) return prev;
          return loadSavedVoice(list, engine);
        });
      } catch {
        // keep fallbacks
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    persistVoice(ttsEngine, selectedVoice);
  }, [selectedVoice, ttsEngine]);

  // Restore document queue from IndexedDB on launch
  useEffect(() => {
    let cancelled = false;
    let finishTimer: number | undefined;
    (async () => {
      const saved = await loadPersistedDocuments();
      if (cancelled) return;

      if (!saved?.docs?.length) {
        setDocsHydrated(true);
        return;
      }

      setIsRestoringSession(true);
      setRestorePercent(4);

      try {
        const total = saved.docs.length;
        const restored: DocItem[] = [];

        for (let i = 0; i < saved.docs.length; i++) {
          if (cancelled) return;
          const doc = await restoreDoc(saved.docs[i]);
          // Reading file bytes is the heavy step — map 8% → 78%
          setRestorePercent(8 + Math.round(((i + 1) / total) * 70));
          restored.push(doc);
        }

        if (cancelled) return;
        setRestorePercent(82);

        const withProgress: DocItem[] = [];
        for (let i = 0; i < restored.length; i++) {
          if (cancelled) return;
          const doc = restored[i];
          const textHash = hashNarrationText(
            doc.editableText || "",
            doc.narrationLanguage
          );
          const hashMatches = narrationHashesMatch(
            doc.narrationTextHash,
            doc.editableText || "",
            doc.narrationLanguage
          );

          if (doc.narrationProgress && !hashMatches) {
            void clearChunkCacheOnServer(doc.id);
            withProgress.push({
              ...doc,
              narrationProgress: null,
              narrationTextHash: null,
            });
          } else {
            const progress = await fetchChunkCacheStatus(doc.id);
            if (!progress) {
              withProgress.push({
                ...doc,
                narrationProgress: null,
                narrationTextHash: hashMatches ? doc.narrationTextHash : null,
              });
            } else if (!hashMatches) {
              void clearChunkCacheOnServer(doc.id);
              withProgress.push({
                ...doc,
                narrationProgress: null,
                narrationTextHash: null,
              });
            } else {
              withProgress.push({
                ...doc,
                narrationProgress: progress,
                narrationTextHash: textHash,
              });
            }
          }
          // Sync chunk status — map 82% → 98%
          setRestorePercent(82 + Math.round(((i + 1) / total) * 16));
        }

        if (cancelled) return;
        setDocuments(withProgress);
        const activeStillExists = withProgress.some((d) => d.id === saved.activeDocId);
        setActiveDocId(activeStillExists ? saved.activeDocId : withProgress[0]?.id ?? null);
        if (saved.outputFormat === "mp3" || saved.outputFormat === "m4b") {
          setOutputFormat(saved.outputFormat);
        }
        setRestorePercent(100);
      } catch (err) {
        console.error("Falha ao restaurar documentos salvos:", err);
        void clearPersistedDocuments();
      } finally {
        if (!cancelled) {
          // Brief beat at 100% so the circle can finish visually
          finishTimer = window.setTimeout(() => {
            if (cancelled) return;
            setIsRestoringSession(false);
            setDocsHydrated(true);
            setRestorePercent(0);
          }, 220);
        }
      }
    })();
    return () => {
      cancelled = true;
      if (finishTimer) window.clearTimeout(finishTimer);
    };
  }, []);

  // Persist document queue whenever it settles
  useEffect(() => {
    if (!docsHydrated) return;

    if (documents.length === 0) {
      void clearPersistedDocuments();
      return;
    }

    const readyDocs = documents.filter((d) => d.fileBase64 && !d.docInfoLoading);
    if (readyDocs.length === 0) return;

    const docs = readyDocs
      .map(serializeDoc)
      .filter((d): d is PersistedDoc => d != null);

    const t = window.setTimeout(() => {
      void persistDocumentsState({ activeDocId, outputFormat, docs });
    }, 400);
    return () => window.clearTimeout(t);
  }, [documents, activeDocId, outputFormat, docsHydrated]);

  // Processing & Result State
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMode, setLoadingMode] = useState<"extract" | "narrate">("extract");
  const [showTextReview, setShowTextReview] = useState<boolean>(false);
  const [progressStep, setProgressStep] = useState<string>("init"); // 'init', 'extraction', 'pre_tts', 'chunks', 'tts', 'encoding', 'done'
  const [encodePercent, setEncodePercent] = useState<number | null>(null);
  const [currentChunk, setCurrentChunk] = useState<number>(0);
  const [totalChunks, setTotalChunks] = useState<number>(0);
  /** Live (non-cached) TTS parts finished this run — used for ETA when resuming mid-narration. */
  const [liveTtsCompleted, setLiveTtsCompleted] = useState(0);
  const liveTtsCompletedRef = useRef(0);
  const pendingLiveTtsRef = useRef(false);
  const lastLivePartRef = useRef<number | null>(null);
  const etaLiveStartedAtRef = useRef<number | null>(null);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState<boolean>(false);
  const [processingFileIndex, setProcessingFileIndex] = useState<number>(0);
  const [processingFileTotal, setProcessingFileTotal] = useState<number>(0);
  const [processingFileName, setProcessingFileName] = useState<string>("");
  const [processingPreviewText, setProcessingPreviewText] = useState<string>("");
  const [lastSavedFileName, setLastSavedFileName] = useState<string>("");
  const [savedFilesCount, setSavedFilesCount] = useState<number>(0);
  
  const [error, setError] = useState<string | null>(null);

  const buildDownloadFileName = (doc: DocItem, format?: OutputFormat) => {
    const safeName = doc.file.name.replace(/\.[^/.]+$/, "") || "narracao";
    const ext = format || doc.audioFormat || outputFormat || "mp3";
    return `${safeName}.${ext}`;
  };

  const saveAudioToDownloads = async (
    opts: { audioId?: string; audioData?: string; fileName: string; saveCover?: boolean }
  ): Promise<{ fileName: string; path: string; coverFileName?: string | null } | null> => {
    try {
      const res = await fetch("/api/save-to-downloads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audioId: opts.audioId,
          audioData: opts.audioData,
          fileName: opts.fileName,
          saveCover: opts.saveCover !== false,
        }),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.error || "Falha ao salvar em Downloads.");
      }
      return {
        fileName: payload.fileName as string,
        path: payload.path as string,
        coverFileName: (payload.coverFileName as string | null) ?? null,
      };
    } catch (err) {
      console.error("Erro ao salvar áudio em Downloads:", err);
      return null;
    }
  };

  // Custom Audio Player State
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [volume, setVolume] = useState<number>(0.8);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const isStoppingRef = useRef(false);
  const stopBatchRef = useRef(false);

  // Voice preview samples
  const [previewLoadingVoice, setPreviewLoadingVoice] = useState<string | null>(null);
  const [previewDeletingVoice, setPreviewDeletingVoice] = useState<string | null>(null);
  const [previewPlayingVoice, setPreviewPlayingVoice] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const previewCacheRef = useRef<Record<string, string>>({});
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const previewRequestIdRef = useRef(0);

  // Persist voice selection
  useEffect(() => {
    try {
      localStorage.setItem(VOICE_STORAGE_KEY, selectedVoice);
    } catch {
      // ignore
    }
  }, [selectedVoice]);

  // Clean up voice preview audio on unmount
  useEffect(() => {
    return () => {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current = null;
      }

      for (const url of Object.values(previewCacheRef.current) as string[]) {
        URL.revokeObjectURL(url);
      }
      previewCacheRef.current = {};
    };
  }, []);

  // Clean up Object URLs on unmount
  useEffect(() => {
    return () => {
      for (const doc of documents) {
        if (doc.audioUrl) URL.revokeObjectURL(doc.audioUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only on unmount
  }, []);

  const updateDoc = (id: string, patch: Partial<DocItem>) => {
    setDocuments((docs) => docs.map((d) => (d.id === id ? { ...d, ...patch } : d)));
  };

  const stripQwenBreaksInQueue = () => {
    if (ttsEngine !== "qwen3") {
      return;
    }

    setDocuments((docs) =>
      docs.map((d) => {
        const extractedText = stripBreakTagsForQwen(d.extractedText, "qwen3");
        const editableText = stripBreakTagsForQwen(d.editableText, "qwen3");

        if (extractedText === d.extractedText && editableText === d.editableText) {
          return d;
        }

        return { ...d, extractedText, editableText };
      })
    );
  };

  // Page / cover previews for active document (refresh only the slots that changed)
  const fetchPageJpeg = async (
    fileBase64: string,
    fileType: "pdf" | "epub",
    coverPage?: string | number
  ): Promise<{ url: string; base64: string } | null> => {
    const res = await fetch("/api/cover-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fileData: fileBase64,
        fileType,
        coverPage,
      }),
    });
    const payload = await res.json();
    if (!res.ok || !payload.found || !payload.imageData) return null;
    const base64 = String(payload.imageData);
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return {
      url: URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" })),
      base64,
    };
  };

  const fetchChapterPreview = async (
    fileBase64: string,
    chapterIndex: number
  ): Promise<ChapterPreview | null> => {
    const res = await fetch("/api/chapter-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fileData: fileBase64, chapterIndex }),
    });
    const payload = await res.json();
    if (!res.ok || !payload.found) return null;
    return {
      title: String(payload.title || `Seção ${chapterIndex}`),
      text: String(payload.text || ""),
      index: Number(payload.index) || chapterIndex,
      total: Number(payload.total) || 0,
    };
  };

  const revokePreviewUrl = (url: string | null) => {
    if (url) URL.revokeObjectURL(url);
  };

  const resolvePdfEndPage = (doc: DocItem) => {
    const startPage = Math.max(1, doc.startPage || 1);
    const endNum = doc.endPage.trim() ? parseInt(doc.endPage, 10) : startPage;
    const endPage =
      Number.isFinite(endNum) && endNum >= startPage ? endNum : startPage;
    return { startPage, endPage };
  };

  const refreshPagePreviews = useCallback(
    async (doc: DocItem, keys: PagePreviewKey[]) => {
      if (!doc.fileBase64 || keys.length === 0) return;

      const uniqueKeys = [...new Set(keys)];
      setPreviewLoadingKeys((prev) => {
        const next = { ...prev };
        for (const k of uniqueKeys) next[k] = true;
        return next;
      });
      if (uniqueKeys.includes("cover")) setCoverPreviewMessage(null);

      const patch: Partial<DocItem> = {};

      try {
        const { startPage, endPage } =
          doc.fileType === "pdf"
            ? resolvePdfEndPage(doc)
            : {
                startPage: Math.max(1, doc.startPage || 1),
                endPage: (() => {
                  const start = Math.max(1, doc.startPage || 1);
                  const endNum = doc.endPage.trim()
                    ? parseInt(doc.endPage, 10)
                    : start;
                  return Number.isFinite(endNum) && endNum >= start
                    ? endNum
                    : start;
                })(),
              };

        if (doc.fileType === "epub") {
          if (uniqueKeys.includes("start")) {
            revokePreviewUrl(doc.startPreviewUrl);
            patch.startPreviewUrl = null;
            patch.startPreviewBase64 = null;
            patch.startChapterPreview = await fetchChapterPreview(
              doc.fileBase64,
              startPage
            );
          }
          if (uniqueKeys.includes("end")) {
            if (doc.endPreviewUrl && doc.endPreviewUrl !== doc.startPreviewUrl) {
              revokePreviewUrl(doc.endPreviewUrl);
            }
            patch.endPreviewUrl = null;
            patch.endPreviewBase64 = null;
            if (endPage === startPage) {
              patch.endChapterPreview = null;
            } else {
              patch.endChapterPreview = await fetchChapterPreview(
                doc.fileBase64,
                endPage
              );
            }
          }
        } else {
          if (uniqueKeys.includes("start")) {
            revokePreviewUrl(doc.startPreviewUrl);
            const startImg = await fetchPageJpeg(
              doc.fileBase64,
              "pdf",
              startPage
            );
            patch.startPreviewUrl = startImg?.url ?? null;
            patch.startPreviewBase64 = startImg?.base64 ?? null;
            patch.startChapterPreview = null;
          }

          if (uniqueKeys.includes("end")) {
            if (doc.endPreviewUrl && doc.endPreviewUrl !== doc.startPreviewUrl) {
              revokePreviewUrl(doc.endPreviewUrl);
            }
            if (endPage === startPage) {
              patch.endPreviewUrl = null;
              patch.endPreviewBase64 = null;
            } else {
              const endImg = await fetchPageJpeg(
                doc.fileBase64,
                "pdf",
                endPage
              );
              patch.endPreviewUrl = endImg?.url ?? null;
              patch.endPreviewBase64 = endImg?.base64 ?? null;
            }
            patch.endChapterPreview = null;
          }
        }

        if (uniqueKeys.includes("cover")) {
          revokePreviewUrl(doc.coverPreviewUrl);
          if (doc.fileType === "epub") {
            if (!doc.exportCover) {
              patch.coverPreviewUrl = null;
              patch.coverPreviewBase64 = null;
              setCoverPreviewMessage(null);
            } else {
              const coverImg = await fetchPageJpeg(doc.fileBase64, "epub");
              patch.coverPreviewUrl = coverImg?.url ?? null;
              patch.coverPreviewBase64 = coverImg?.base64 ?? null;
              setCoverPreviewMessage(coverImg ? null : "Capa não encontrada");
            }
          } else {
            const coverEmpty = doc.coverPage.trim() === "";
            const wantCover = doc.exportCover && !coverEmpty;
            if (!wantCover) {
              patch.coverPreviewUrl = null;
              patch.coverPreviewBase64 = null;
              setCoverPreviewMessage(null);
            } else {
              const coverPage = Math.max(1, parseInt(doc.coverPage, 10) || 1);
              const coverImg = await fetchPageJpeg(
                doc.fileBase64,
                "pdf",
                coverPage
              );
              patch.coverPreviewUrl = coverImg?.url ?? null;
              patch.coverPreviewBase64 = coverImg?.base64 ?? null;
              setCoverPreviewMessage(coverImg ? null : "Capa não encontrada");
            }
          }
        }

        updateDoc(doc.id, patch);
      } catch (err: any) {
        const failPatch: Partial<DocItem> = {};
        for (const k of uniqueKeys) {
          if (k === "start") {
            failPatch.startPreviewUrl = null;
            failPatch.startPreviewBase64 = null;
            failPatch.startChapterPreview = null;
          }
          if (k === "end") {
            failPatch.endPreviewUrl = null;
            failPatch.endPreviewBase64 = null;
            failPatch.endChapterPreview = null;
          }
          if (k === "cover") {
            failPatch.coverPreviewUrl = null;
            failPatch.coverPreviewBase64 = null;
          }
        }
        updateDoc(doc.id, failPatch);
        if (uniqueKeys.includes("cover")) {
          setCoverPreviewMessage(err?.message || "Falha no preview das páginas");
        }
      } finally {
        setPreviewLoadingKeys((prev) => {
          const next = { ...prev };
          for (const k of uniqueKeys) next[k] = false;
          return next;
        });
      }
    },
    []
  );

  useEffect(() => {
    if (!activeDoc?.fileBase64 || activeDoc.docInfoLoading) return;

    const next = {
      id: activeDoc.id,
      fileBase64: activeDoc.fileBase64,
      fileType: activeDoc.fileType,
      startPage: activeDoc.startPage,
      endPage: activeDoc.endPage,
      coverPage: activeDoc.coverPage,
      exportCover: activeDoc.exportCover,
    };
    const prev = previewDepsRef.current;

    const isDocSwitch =
      !prev ||
      prev.id !== next.id ||
      prev.fileBase64 !== next.fileBase64 ||
      prev.fileType !== next.fileType;

    let keys: PagePreviewKey[];
    if (isDocSwitch) {
      keys = ["start", "end", "cover"];
    } else {
      keys = [];
      if (prev.startPage !== next.startPage) keys.push("start");
      if (prev.endPage !== next.endPage) keys.push("end");
      if (
        prev.coverPage !== next.coverPage ||
        prev.exportCover !== next.exportCover
      ) {
        keys.push("cover");
      }
    }

    previewDepsRef.current = next;
    if (keys.length === 0) return;

    // Reuse persisted previews only when switching/restoring a doc.
    // When start/end/cover change, always regenerate and save the new version.
    const needed = isDocSwitch
      ? keys.filter((key) => {
          if (key === "start") {
            if (activeDoc.fileType === "epub") return !activeDoc.startChapterPreview;
            return !(activeDoc.startPreviewUrl || activeDoc.startPreviewBase64);
          }
          if (key === "end") {
            if (activeDoc.fileType === "epub") {
              const endNum = parseInt(activeDoc.endPage, 10) || activeDoc.startPage;
              if (endNum === activeDoc.startPage) return false;
              return !activeDoc.endChapterPreview;
            }
            const endNum = parseInt(activeDoc.endPage, 10) || activeDoc.startPage;
            if (endNum === activeDoc.startPage) return false;
            return !(activeDoc.endPreviewUrl || activeDoc.endPreviewBase64);
          }
          if (!activeDoc.exportCover) return false;
          return !(activeDoc.coverPreviewUrl || activeDoc.coverPreviewBase64);
        })
      : keys;
    if (needed.length === 0) return;

    const t = window.setTimeout(() => {
      void refreshPagePreviews(activeDoc, needed);
    }, 400);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refresh only changed preview slots
  }, [
    activeDoc?.id,
    activeDoc?.fileBase64,
    activeDoc?.coverPage,
    activeDoc?.exportCover,
    activeDoc?.startPage,
    activeDoc?.endPage,
    activeDoc?.fileType,
    activeDoc?.docInfoLoading,
    refreshPagePreviews,
  ]);

  // Drag-and-drop handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
      e.target.value = "";
    }
  };

  const loadDocumentInfo = async (docId: string, base64: string, type: "pdf" | "epub") => {
    updateDoc(docId, { docInfoLoading: true });
    try {
      const res = await fetch("/api/document-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileData: base64, fileType: type }),
      });
      const info = await res.json();
      if (!res.ok) throw new Error(info.error || "Falha ao ler o documento.");

      if (type === "pdf") {
        const pageCount = typeof info.pageCount === "number" ? info.pageCount : null;
        updateDoc(docId, {
          docInfoMessage: info.message || "",
          pdfPageCount: pageCount,
          startPage: 1,
          endPage: pageCount != null ? String(pageCount) : "",
          docInfoLoading: false,
        });
      } else {
        const chapters = Array.isArray(info.chapters) ? info.chapters : [];
        updateDoc(docId, {
          docInfoMessage: info.message || "",
          epubChapters: chapters,
          startPage: 1,
          endPage: chapters.length > 0 ? String(chapters.length) : "",
          docInfoLoading: false,
        });
      }
    } catch (err: any) {
      console.error(err);
      updateDoc(docId, { docInfoLoading: false });
      setError(err.message || "Não foi possível inspecionar o documento.");
    }
  };

  const processFiles = (files: File[]) => {
    setError(null);
    const valid = files.filter(isPdfOrEpub);
    if (valid.length === 0) {
      setError("Por favor, adicione apenas arquivos PDF ou EPUB válidos.");
      return;
    }
    if (valid.length < files.length) {
      setError("Alguns arquivos foram ignorados. Apenas PDF e EPUB são aceitos.");
    }

    const newDocs: DocItem[] = valid.map((file) => ({
      id: createDocId(),
      file,
      fileBase64: "",
      fileType: detectFileType(file),
      startPage: 1,
      endPage: "",
      coverPage: "1",
      exportCover: true,
      pdfPageCount: null,
      epubChapters: [],
      docInfoMessage: "",
      docInfoLoading: true,
      editableText: "",
      extractedText: "",
      pagesNarrated: "",
      audioBase64: null,
      audioUrl: null,
      audioFormat: outputFormat,
      coverPreviewUrl: null,
      startPreviewUrl: null,
      endPreviewUrl: null,
      coverPreviewBase64: null,
      startPreviewBase64: null,
      endPreviewBase64: null,
      startChapterPreview: null,
      endChapterPreview: null,
      narrationProgress: null,
      narrationTextHash: null,
      narrationLanguage: detectSystemNarrationLanguage(),
    }));

    setDocuments((prev) => [...prev, ...newDocs]);
    if (!activeDocId && newDocs.length > 0) {
      setActiveDocId(newDocs[0].id);
    }

    for (const doc of newDocs) {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(",")[1];
        updateDoc(doc.id, { fileBase64: base64 });
        void loadDocumentInfo(doc.id, base64, doc.fileType);
      };
      reader.onerror = () => {
        updateDoc(doc.id, { docInfoLoading: false });
        setError(`Erro ao ler o arquivo ${doc.fileType.toUpperCase()}: ${doc.file.name}`);
      };
      reader.readAsDataURL(doc.file);
    }
  };

  const revokeDocUrls = (doc: DocItem) => {
    if (doc.audioUrl) URL.revokeObjectURL(doc.audioUrl);
    if (doc.coverPreviewUrl) URL.revokeObjectURL(doc.coverPreviewUrl);
    if (doc.startPreviewUrl) URL.revokeObjectURL(doc.startPreviewUrl);
    if (doc.endPreviewUrl) URL.revokeObjectURL(doc.endPreviewUrl);
  };

  const removeDocFromState = (docId: string, opts?: { clearChunkCache?: boolean }) => {
    if (opts?.clearChunkCache !== false) {
      void clearChunkCacheOnServer(docId);
    }
    setDocuments((prev) => {
      const target = prev.find((d) => d.id === docId);
      if (target) revokeDocUrls(target);
      const next = prev.filter((d) => d.id !== docId);
      setActiveDocId((current) => {
        if (current !== docId) return current;
        return next[0]?.id ?? null;
      });
      setReviewDocId((current) => {
        if (current !== docId) return current;
        return next[0]?.id ?? null;
      });
      setResultDocId((current) => {
        if (current !== docId) return current;
        return next.find((d) => d.audioUrl)?.id ?? null;
      });
      if (next.length === 0) {
        setShowTextReview(false);
      }
      return next;
    });
    if (previewDepsRef.current?.id === docId) {
      previewDepsRef.current = null;
    }
  };

  const handleRemoveFile = (docId: string) => {
    removeDocFromState(docId, { clearChunkCache: true });
  };

  const handleClearNarrationProgress = (docId: string) => {
    void clearChunkCacheOnServer(docId);
    updateDoc(docId, { narrationProgress: null, narrationTextHash: null });
  };

  const updateEditableText = (docId: string, value: string) => {
    setDocuments((docs) =>
      docs.map((d) => {
        if (d.id !== docId)
          return d;

        const shouldInvalidate =
          !!d.narrationProgress &&
          !!d.narrationTextHash &&
          !narrationHashesMatch(d.narrationTextHash, value, d.narrationLanguage);

        if (shouldInvalidate) {
          void clearChunkCacheOnServer(d.id);
          return {
            ...d,
            editableText: value,
            narrationProgress: null,
            narrationTextHash: null,
          };
        }

        return { ...d, editableText: value };
      })
    );
  };

  const setDocNarrationLanguage = (docId: string, language: NarrationLanguageId) => {
    setDocuments((docs) =>
      docs.map((d) => {
        if (d.id !== docId)
          return d;

        if (d.narrationLanguage === language)
          return d;

        const shouldInvalidate =
          !!d.narrationProgress &&
          !!d.narrationTextHash &&
          !narrationHashesMatch(d.narrationTextHash, d.editableText, language);

        if (shouldInvalidate) {
          void clearChunkCacheOnServer(d.id);
          return {
            ...d,
            narrationLanguage: language,
            narrationProgress: null,
            narrationTextHash: null,
          };
        }

        return { ...d, narrationLanguage: language };
      })
    );
  };

  const handleClearAllFiles = () => {
    setDocuments((prev) => {
      for (const doc of prev) {
        revokeDocUrls(doc);
        void clearChunkCacheOnServer(doc.id);
      }
      return [];
    });
    setActiveDocId(null);
    setReviewDocId(null);
    setResultDocId(null);
    setShowTextReview(false);
    setBatchDone(false);
    setError(null);
    previewDepsRef.current = null;
    void clearPersistedDocuments();
  };

  const validatePageRange = (doc: DocItem): { start: number; end: number } | null => {
    const start = Math.max(1, doc.startPage || 1);
    if (doc.fileType === "pdf" && doc.pdfPageCount != null && start > doc.pdfPageCount) {
      setError(`${doc.file.name}: este PDF tem ${doc.pdfPageCount} página(s). A página inicial ${start} está fora do intervalo.`);
      return null;
    }
    if (doc.fileType === "epub" && doc.epubChapters.length > 0 && start > doc.epubChapters.length) {
      setError(`${doc.file.name}: este EPUB tem ${doc.epubChapters.length} seção(ões). O índice ${start} está fora do intervalo.`);
      return null;
    }

    const end = doc.endPage.trim()
      ? parseInt(doc.endPage, 10)
      : start;

    if (!Number.isFinite(end) || end < 1) {
      setError(`${doc.file.name}: a página/seção final é inválida.`);
      return null;
    }
    if (end < start) {
      setError(`${doc.file.name}: a página/seção final (${end}) não pode ser menor que a inicial (${start}).`);
      return null;
    }
    if (doc.fileType === "pdf" && doc.pdfPageCount != null && end > doc.pdfPageCount) {
      setError(`${doc.file.name}: este PDF tem ${doc.pdfPageCount} página(s). A página final ${end} está fora do intervalo.`);
      return null;
    }
    if (doc.fileType === "epub" && doc.epubChapters.length > 0 && end > doc.epubChapters.length) {
      setError(`${doc.file.name}: este EPUB tem ${doc.epubChapters.length} seção(ões). O índice ${end} está fora do intervalo.`);
      return null;
    }
    return { start, end };
  };

  const allDocsHaveExtractedText = documents.length > 0 && documents.every((d) => d.editableText.trim());

  const extractDocuments = async (docsToExtract: DocItem[]): Promise<boolean> => {
    if (docsToExtract.length === 0) return true;

    const ranges: { id: string; start: number; end: number }[] = [];
    for (const doc of docsToExtract) {
      const range = validatePageRange(doc);
      if (!range) return false;
      ranges.push({ id: doc.id, ...range });
    }

    setLoading(true);
    setLoadingMode("extract");
    setError(null);
    setShowTextReview(false);
    setProgressStep("extraction");
    setProcessingFileTotal(docsToExtract.length);
    setProcessingPreviewText("");

    const extractIds = new Set(docsToExtract.map((d) => d.id));

    // Clear previous audio / text for docs being re-extracted
    setDocuments((docs) =>
      docs.map((d) => {
        if (!extractIds.has(d.id)) return d;
        if (d.audioUrl) URL.revokeObjectURL(d.audioUrl);
        if (d.narrationProgress || d.narrationTextHash) {
          void clearChunkCacheOnServer(d.id);
        }
        return {
          ...d,
          editableText: "",
          extractedText: "",
          pagesNarrated: "",
          audioBase64: null,
          audioUrl: null,
          narrationProgress: null,
          narrationTextHash: null,
        };
      })
    );

    try {
      for (let i = 0; i < docsToExtract.length; i++) {
        const doc = docsToExtract[i];
        const range = ranges[i];
        setProcessingFileIndex(i + 1);
        setProcessingFileName(doc.file.name);
        setProgressStep("extraction");
        setCurrentChunk(0);
        setTotalChunks(0);

        const response = await fetch("/api/extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileData: doc.fileBase64,
            fileType: doc.fileType,
            startPage: range.start,
            endPage: range.end,
          }),
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(`${doc.file.name}: ${payload.error || "Falha ao extrair o texto."}`);
        }

        const extractedText = stripBreakTagsForQwen(
          String(payload.extractedText ?? ""),
          ttsEngine
        );
        updateDoc(doc.id, {
          extractedText,
          editableText: extractedText,
          pagesNarrated: payload.pagesNarrated || "",
        });
      }

      return true;
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Não foi possível extrair o texto do documento.");
      return false;
    } finally {
      setLoading(false);
      setProcessingFileIndex(0);
      setProcessingFileTotal(0);
      setProcessingFileName("");
    }
  };

  // Step 1: open review from cache when available; otherwise extract first
  const handleExtractForReview = async () => {
    if (documents.length === 0 || !allDocsReady) {
      return;
    }

    // Text already in session cache — skip re-extraction and open review
    if (allDocsHaveExtractedText) {
      setError(null);
      setReviewDocId(documents[0]?.id ?? null);
      stripQwenBreaksInQueue();
      setShowTextReview(true);
      return;
    }

    const missing = documents.filter((d) => !d.editableText.trim());
    const ok = await extractDocuments(missing.length > 0 ? missing : documents);

    if (!ok) {
      return;
    }

    setReviewDocId(documents[0]?.id ?? null);
    stripQwenBreaksInQueue();
    setShowTextReview(true);
  };

  // Force re-extract the document currently open in review
  const handleReExtract = async () => {
    const doc = reviewDoc ?? documents[0];
    if (!doc || !allDocsReady) {
      return;
    }

    const ok = await extractDocuments([doc]);

    if (!ok) {
      return;
    }

    setReviewDocId(doc.id);
    stripQwenBreaksInQueue();
    setShowTextReview(true);
  };

  const stopVoicePreview = () => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current.onended = null;
      previewAudioRef.current.onerror = null;
      previewAudioRef.current = null;
    }

    setPreviewPlayingVoice(null);
  };

  useEffect(() => {
    previewRequestIdRef.current += 1;
    stopVoicePreview();
  }, [previewLanguageId]);

  const playVoicePreview = async (
    voiceId: string,
    opts?: { forceRegenerate?: boolean }
  ) => {
    setPreviewError(null);

    if (previewPlayingVoice === voiceId && !opts?.forceRegenerate) {
      stopVoicePreview();

      return;
    }

    stopVoicePreview();

    const cacheKey = `${voiceId}::${previewLanguageId}`;
    const requestId = ++previewRequestIdRef.current;
    let url = opts?.forceRegenerate ? undefined : previewCacheRef.current[cacheKey];
    if (!url) {
      setPreviewLoadingVoice(voiceId);
      try {
        const response = await fetch("/api/voice-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ voiceName: voiceId, language: previewLanguageId }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Falha ao gerar prévia da voz.");
        }

        const binary = atob(payload.audioData as string);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }

        const blob = new Blob([bytes], { type: payload.mimeType || "audio/mpeg" });
        url = URL.createObjectURL(blob);
        previewCacheRef.current[cacheKey] = url;
      } catch (err: unknown) {
        console.error(err);
        setPreviewError(
          err instanceof Error ? err.message : "Não foi possível ouvir esta voz agora."
        );
        setPreviewLoadingVoice(null);

        return;
      } finally {
        setPreviewLoadingVoice((current) => (current === voiceId ? null : current));
      }
    }

    if (requestId !== previewRequestIdRef.current) {
      return;
    }

    const audio = new Audio(url);
    previewAudioRef.current = audio;
    setPreviewPlayingVoice(voiceId);
    audio.onended = () => {
      if (previewAudioRef.current === audio) {
        previewAudioRef.current = null;
        setPreviewPlayingVoice(null);
      }
    };
    audio.onerror = () => {
      setPreviewError("Erro ao reproduzir a prévia.");
      stopVoicePreview();
    };
    try {
      await audio.play();
    } catch (err: unknown) {
      setPreviewError(
        err instanceof Error ? err.message : "Não foi possível reproduzir o áudio."
      );
      stopVoicePreview();
    }
  };

  const deleteAndRegenerateVoicePreview = async (voiceId: string) => {
    setPreviewError(null);
    stopVoicePreview();

    const cacheKey = `${voiceId}::${previewLanguageId}`;
    setPreviewDeletingVoice(voiceId);

    try {
      const response = await fetch("/api/voice-preview", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voiceName: voiceId, language: previewLanguageId }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Falha ao excluir a prévia da voz.");
      }

      const oldUrl = previewCacheRef.current[cacheKey];
      if (oldUrl) {
        URL.revokeObjectURL(oldUrl);
        delete previewCacheRef.current[cacheKey];
      }

      await playVoicePreview(voiceId, { forceRegenerate: true });
    } catch (err: unknown) {
      console.error(err);
      setPreviewError(
        err instanceof Error ? err.message : "Não foi possível regenerar a prévia."
      );
    } finally {
      setPreviewDeletingVoice((current) => (current === voiceId ? null : current));
    }
  };

  // Step 2: narrate one document (reuses disk chunk cache keyed by doc UUID)
  type NarrateResult =
    | { status: "done"; fileName: string }
    | { status: "cancelled"; completed: number; total: number };

  const narrateDocument = async (doc: DocItem): Promise<NarrateResult> => {
    const text = stripBreakTagsForQwen(doc.editableText.trim(), ttsEngine);
    if (!text) {
      throw new Error(`${doc.file.name}: edite ou confirme um texto antes de narrar.`);
    }

    const textHash = hashNarrationText(text, doc.narrationLanguage);
    // If text or language changed since last cached run, drop old progress (server also clears by fingerprint)
    if (doc.narrationTextHash && !narrationHashesMatch(doc.narrationTextHash, text, doc.narrationLanguage)) {
      void clearChunkCacheOnServer(doc.id);
      updateDoc(doc.id, { narrationProgress: null, narrationTextHash: null });
    } else {
      updateDoc(doc.id, { narrationTextHash: textHash });
    }

    setProcessingPreviewText(text);
    setProgressStep("pre_tts");
    setEncodePercent(null);
    setCurrentChunk(0);
    setTotalChunks(0);
    liveTtsCompletedRef.current = 0;
    setLiveTtsCompleted(0);
    pendingLiveTtsRef.current = false;
    lastLivePartRef.current = null;
    etaLiveStartedAtRef.current = null;
    setIsStopping(false);
    isStoppingRef.current = false;

    const taskId = Math.random().toString(36).substring(7);
    setCurrentTaskId(taskId);

    if (doc.audioUrl) {
      URL.revokeObjectURL(doc.audioUrl);
      updateDoc(doc.id, { audioUrl: null, audioBase64: null });
    }

    const response = await fetch("/api/narrate-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        voiceName: selectedVoice,
        taskId,
        docId: doc.id,
        pagesNarrated: doc.pagesNarrated,
        fileData: doc.fileBase64,
        fileType: doc.fileType,
        outputFormat,
        coverPage:
          doc.fileType === "pdf"
            ? doc.exportCover && doc.coverPage.trim() !== ""
              ? doc.coverPage
              : null
            : undefined,
        includeCover:
          doc.exportCover &&
          (doc.fileType === "epub" ||
            (doc.fileType === "pdf" && doc.coverPage.trim() !== "")),
        sourceFileName: doc.file.name,
        language: doc.narrationLanguage,
      }),
    });

    if (!response.ok) {
      throw new Error(`${doc.file.name}: erro de conexão ou falha no servidor ao iniciar a narração.`);
    }

    if (!response.body) {
      throw new Error(`${doc.file.name}: o servidor não retornou um fluxo de dados válido.`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let gotDone = false;
    let cancelledResult: NarrateResult | null = null;
    let savedFileName = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine || !trimmedLine.startsWith("data: ")) continue;

        const dataStr = trimmedLine.slice(6).trim();
        if (!dataStr) continue;

        try {
          const payload = JSON.parse(dataStr);

          if (payload.type === "status") {
            if (isStoppingRef.current && payload.step === "tts") {
              continue;
            }
            if (payload.step) setProgressStep(payload.step);
            if (payload.current !== undefined) setCurrentChunk(payload.current);
            if (payload.total !== undefined) setTotalChunks(payload.total);
            if (typeof payload.percent === "number") {
              setEncodePercent(payload.percent);
            } else if (payload.step && payload.step !== "encoding") {
              setEncodePercent(null);
            }
            // Track only real TTS work for ETA (cache replay must not dilute avg/part).
            if (payload.step === "tts") {
              if (payload.cached) {
                if (pendingLiveTtsRef.current) {
                  liveTtsCompletedRef.current += 1;
                  setLiveTtsCompleted(liveTtsCompletedRef.current);
                  pendingLiveTtsRef.current = false;
                  lastLivePartRef.current = null;
                }
              } else if (typeof payload.current === "number") {
                if (
                  pendingLiveTtsRef.current &&
                  lastLivePartRef.current !== null &&
                  lastLivePartRef.current !== payload.current
                ) {
                  liveTtsCompletedRef.current += 1;
                  setLiveTtsCompleted(liveTtsCompletedRef.current);
                }
                if (etaLiveStartedAtRef.current === null) {
                  etaLiveStartedAtRef.current = Date.now();
                }
                lastLivePartRef.current = payload.current;
                pendingLiveTtsRef.current = true;
              }
            } else if (
              (payload.step === "encoding" || payload.step === "done") &&
              pendingLiveTtsRef.current
            ) {
              liveTtsCompletedRef.current += 1;
              setLiveTtsCompleted(liveTtsCompletedRef.current);
              pendingLiveTtsRef.current = false;
            }
            if (
              typeof payload.current === "number" &&
              typeof payload.total === "number" &&
              payload.total > 0
            ) {
              const completed =
                payload.step === "tts"
                  ? payload.cached
                    ? payload.current
                    : Math.max(0, payload.current - 1)
                  : payload.current;
              updateDoc(doc.id, {
                narrationProgress: {
                  completed: Math.min(payload.total, completed),
                  total: payload.total,
                },
                narrationTextHash: textHash,
              });
            }
            if (
              !text &&
              typeof payload.extractedText === "string" &&
              payload.extractedText.length > 0
            ) {
              const extractedText = stripBreakTagsForQwen(
                payload.extractedText,
                ttsEngine
              );
              setProcessingPreviewText(extractedText);
              updateDoc(doc.id, { extractedText });
            }
          } else if (payload.type === "cancelled") {
            const completed = Number(payload.completed) || 0;
            const total = Number(payload.total) || 0;
            updateDoc(doc.id, {
              narrationProgress:
                total > 0 ? { completed, total } : null,
              narrationTextHash: total > 0 && completed > 0 ? textHash : null,
            });
            cancelledResult = { status: "cancelled", completed, total };
          } else if (payload.type === "done") {
            setProgressStep("done");
            gotDone = true;

            const pagesNarrated = payload.pagesNarrated || doc.pagesNarrated;
            const fmt: OutputFormat = payload.format === "m4b" ? "m4b" : "mp3";
            let url: string | null = null;

            if (typeof payload.audioUrl === "string" && payload.audioUrl) {
              const audioRes = await fetch(payload.audioUrl);
              if (!audioRes.ok) {
                throw new Error(`${doc.file.name}: falha ao baixar o áudio gerado.`);
              }
              const blob = await audioRes.blob();
              url = URL.createObjectURL(blob);
            } else if (typeof payload.audioData === "string" && payload.audioData) {
              const binary = atob(payload.audioData);
              const bytes = new Uint8Array(binary.length);
              for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
              }
              const blob = new Blob([bytes], { type: "audio/mp3" });
              url = URL.createObjectURL(blob);
            } else {
              throw new Error(`${doc.file.name}: resposta sem áudio.`);
            }

            const downloadName = buildDownloadFileName(doc, fmt);
            const saved = await saveAudioToDownloads({
              audioId: typeof payload.audioId === "string" ? payload.audioId : undefined,
              audioData: typeof payload.audioData === "string" ? payload.audioData : undefined,
              fileName: downloadName,
              // M4B already embeds artwork — no separate cover JPEG.
              saveCover: doc.exportCover && fmt !== "m4b",
            });
            if (saved) {
              const label = saved.coverFileName
                ? `${saved.fileName} + ${saved.coverFileName}`
                : saved.fileName;
              savedFileName = label;
              setLastSavedFileName(label);
              setSavedFilesCount((n) => n + 1);
            } else {
              savedFileName = downloadName;
            }

            // Keep audio briefly only if we somehow stay on results with this doc;
            // normally the item is removed from the queue after success.
            if (url) URL.revokeObjectURL(url);
            updateDoc(doc.id, {
              audioBase64: null,
              extractedText: stripBreakTagsForQwen(text, ttsEngine),
              editableText: stripBreakTagsForQwen(text, ttsEngine),
              pagesNarrated,
              audioUrl: null,
              audioFormat: fmt,
              narrationProgress: null,
              narrationTextHash: null,
            });
          } else if (payload.type === "error") {
            throw new Error(`${doc.file.name}: ${payload.error || "Ocorreu um erro durante o processamento."}`);
          }
        } catch (jsonErr: any) {
          if (jsonErr.message && !jsonErr.message.startsWith("Unexpected token")) {
            throw jsonErr;
          }
          console.error("Erro ao analisar dados do fluxo:", jsonErr);
        }
      }
    }

    if (cancelledResult) {
      return cancelledResult;
    }

    if (!gotDone) {
      throw new Error(`${doc.file.name}: a narração terminou sem áudio gerado.`);
    }

    return { status: "done", fileName: savedFileName || doc.file.name };
  };

  const handleConfirmNarration = async () => {
    const docsSnapshot = documents.map((d) => ({ ...d }));
    const empty = docsSnapshot.find((d) => !d.editableText.trim());
    if (empty) {
      setError(`${empty.file.name}: edite ou confirme um texto antes de narrar.`);
      setReviewDocId(empty.id);
      return;
    }

    setShowTextReview(false);
    setLoading(true);
    setLoadingMode("narrate");
    setError(null);
    setBatchDone(false);
    setProcessingFileTotal(docsSnapshot.length);
    setIsPlaying(false);
    setCurrentTime(0);
    setLastSavedFileName("");
    setSavedFilesCount(0);
    stopBatchRef.current = false;

    let savedCount = 0;
    let lastName = "";
    let wasCancelled = false;

    try {
      for (let i = 0; i < docsSnapshot.length; i++) {
        const doc = docsSnapshot[i];
        setProcessingFileIndex(i + 1);
        setProcessingFileName(doc.file.name);
        const result = await narrateDocument(doc);

        if (result.status === "cancelled") {
          wasCancelled = true;
          break;
        }

        savedCount += 1;
        lastName = result.fileName || doc.file.name;
        // Completed: drop from queue + clear preview/chunk caches (server already cleared chunks)
        removeDocFromState(doc.id, { clearChunkCache: true });

        if (stopBatchRef.current) break;

        setIsStopping(false);
        isStoppingRef.current = false;
        setCurrentTaskId(null);
      }

      if (wasCancelled) {
        // Keep cancelled item in state with progress; return to setup
        setShowTextReview(false);
      } else if (savedCount > 0) {
        setBatchDone(true);
        setSavedFilesCount(savedCount);
        if (lastName) setLastSavedFileName(lastName);
      }
      setIsPlaying(false);
      setCurrentTime(0);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Não foi possível completar a narração. Verifique sua conexão e tente novamente.");
      if (savedCount > 0) {
        setBatchDone(true);
        setSavedFilesCount(savedCount);
        if (lastName) setLastSavedFileName(lastName);
      } else {
        stripQwenBreaksInQueue();
        setShowTextReview(true);
      }
    } finally {
      setLoading(false);
      setIsStopping(false);
      isStoppingRef.current = false;
      stopBatchRef.current = false;
      setCurrentTaskId(null);
      setProcessingFileIndex(0);
      setProcessingFileTotal(0);
      setProcessingFileName("");
      setProcessingPreviewText("");
    }
  };

  // Stop current narration — true cancel (no partial audio); keep item + chunk cache
  const handleStopNarration = async () => {
    if (!currentTaskId || isStopping) return;
    isStoppingRef.current = true;
    stopBatchRef.current = true;
    setIsStopping(true);
    setProgressStep("chunks");

    try {
      await fetch("/api/narrate-stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taskId: currentTaskId })
      });
    } catch (err) {
      console.error("Erro ao enviar comando de parada:", err);
    }
  };

  // Custom Audio Controls
  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play()
        .then(() => setIsPlaying(true))
        .catch(err => {
          console.error("Audio playback error:", err);
          setError("Erro ao reproduzir áudio. Verifique as permissões de áudio do seu navegador.");
        });
    }
  };

  const handleTimeUpdate = () => {
    if (!audioRef.current) return;
    setCurrentTime(audioRef.current.currentTime);
  };

  const handleLoadedMetadata = () => {
    if (!audioRef.current) return;
    setDuration(audioRef.current.duration);
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const time = parseFloat(e.target.value);
    audioRef.current.currentTime = time;
    setCurrentTime(time);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const vol = parseFloat(e.target.value);
    audioRef.current.volume = vol;
    setVolume(vol);
    if (vol > 0) setIsMuted(false);
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    if (isMuted) {
      audioRef.current.volume = volume;
      setIsMuted(false);
    } else {
      audioRef.current.volume = 0;
      setIsMuted(true);
    }
  };

  const skipTime = (amount: number) => {
    if (!audioRef.current) return;
    let newTime = audioRef.current.currentTime + amount;
    if (newTime < 0) newTime = 0;
    if (newTime > duration) newTime = duration;
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time: number) => {
    if (isNaN(time)) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;
  };

  // Force physical browser download of the audio
  const downloadMp3 = (doc: DocItem | null = resultDoc) => {
    if (!doc?.audioUrl) return;
    const a = document.createElement("a");
    a.href = doc.audioUrl;
    a.download = buildDownloadFileName(doc);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const resetAll = () => {
    setShowTextReview(false);
    setResultDocId(null);
    setBatchDone(false);
    setSavedFilesCount(0);
    setLastSavedFileName("");
    setDocuments((docs) =>
      docs.map((d) => {
        if (d.audioUrl) URL.revokeObjectURL(d.audioUrl);
        return {
          ...d,
          editableText: "",
          extractedText: "",
          pagesNarrated: "",
          audioBase64: null,
          audioUrl: null,
          audioFormat: outputFormat,
        };
      })
    );
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  };

  const selectResultDoc = (docId: string) => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setResultDocId(docId);
  };

  const [stageBarPercent, setStageBarPercent] = useState(0);
  const [etaTick, setEtaTick] = useState(0);
  const [etaLabel, setEtaLabel] = useState<string | null>(null);
  const stageStartedAtRef = useRef<number>(Date.now());
  // Stable ETA: recalculate only on new evidence, countdown between ticks
  const etaSecondsRef = useRef<number | null>(null);
  const etaFinishedRef = useRef(0);
  const progressStageKey = `${loadingMode}:${progressStep}`;

  // Each stage owns the full 0–100% bar; timer fills stages without discrete units
  useEffect(() => {
    if (!loading) {
      setStageBarPercent(0);
      return;
    }

    setStageBarPercent(0);

    if (progressStep === "done") {
      setStageBarPercent(100);
      return;
    }

    // TTS uses real chunk progress — no timer fill
    if (progressStep === "tts" && totalChunks > 0) {
      return;
    }

    // Encoding uses real encoder percent when available
    if (progressStep === "encoding") {
      return;
    }

    const started = Date.now();
    const tau =
      loadingMode === "extract" || progressStep === "extraction"
        ? 14
        : 3.5;

    const id = window.setInterval(() => {
      const elapsedSec = (Date.now() - started) / 1000;
      // Approaches ~92% while the stage runs; hits 100% only when the next stage starts/ends
      const p = Math.min(92, Math.round((1 - Math.exp(-elapsedSec / tau)) * 100));
      setStageBarPercent(p);
    }, 150);

    return () => window.clearInterval(id);
  }, [loading, progressStageKey, progressStep, loadingMode, totalChunks]);

  // Reset ETA clock whenever the stage changes
  useEffect(() => {
    if (!loading) {
      setEtaLabel(null);
      etaSecondsRef.current = null;
      etaFinishedRef.current = 0;
      return;
    }
    stageStartedAtRef.current = Date.now();
    etaSecondsRef.current = null;
    etaFinishedRef.current = 0;
    setEtaLabel(null);
  }, [loading, progressStageKey]);

  // Tick every second to refresh remaining-time estimate
  useEffect(() => {
    if (!loading || progressStep === "done") return;
    const id = window.setInterval(() => setEtaTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, [loading, progressStep]);

  const formatEta = (seconds: number): string => {
    const s = Math.max(0, Math.round(seconds));
    if (s <= 0) return "quase pronto";
    if (s < 60) return `~${s}s restantes`;
    const m = Math.floor(s / 60);
    const rem = s % 60;
    if (m < 60) return rem > 0 ? `~${m}m ${rem}s restantes` : `~${m}m restantes`;
    const h = Math.floor(m / 60);
    const m2 = m % 60;
    return m2 > 0 ? `~${h}h ${m2}m restantes` : `~${h}h restantes`;
  };

  // Progress bar: reset per stage; 100% of the bar = progress within the current stage only
  const getProgressPercentage = () => {
    if (!loading) return 0;
    if (progressStep === "done") return 100;

    if (progressStep === "tts") {
      if (totalChunks <= 0) return stageBarPercent;
      // Only completed parts — starts at 0% while the 1st part is running
      const finished = Math.max(0, currentChunk - 1);
      return Math.min(99, Math.round((finished / totalChunks) * 100));
    }

    if (progressStep === "encoding" && encodePercent != null) {
      return Math.min(100, Math.max(0, Math.round(encodePercent)));
    }

    return stageBarPercent;
  };

  // ETA: for TTS, average only live (non-cached) parts so resume mid-narration stays accurate.
  useEffect(() => {
    if (!loading || progressStep === "done") {
      setEtaLabel(null);
      return;
    }

    if (progressStep === "tts" && totalChunks > 0) {
      const finished = Math.max(0, currentChunk - 1);
      const remainingParts = Math.max(0, totalChunks - finished);

      if (!etaLiveStartedAtRef.current) {
        // Still replaying cache — no live synth yet
        etaSecondsRef.current = null;
        etaFinishedRef.current = 0;
        setEtaLabel(null);
        return;
      }

      if (liveTtsCompleted < 1) {
        etaSecondsRef.current = null;
        etaFinishedRef.current = 0;
        setEtaLabel("Estimando após a 1ª parte…");
        return;
      }

      const liveElapsed = (Date.now() - etaLiveStartedAtRef.current) / 1000;
      if (liveElapsed < 1.5) {
        setEtaLabel("Estimando após a 1ª parte…");
        return;
      }

      if (etaSecondsRef.current === null || liveTtsCompleted !== etaFinishedRef.current) {
        etaFinishedRef.current = liveTtsCompleted;
        const avgPerPart = liveElapsed / liveTtsCompleted;
        etaSecondsRef.current = avgPerPart * remainingParts;
      } else {
        etaSecondsRef.current = Math.max(0, etaSecondsRef.current - 1);
      }

      setEtaLabel(formatEta(etaSecondsRef.current));
      return;
    }

    const elapsed = (Date.now() - stageStartedAtRef.current) / 1000;
    if (elapsed < 1.5) {
      setEtaLabel(null);
      return;
    }

    const pct = getProgressPercentage();
    if (pct >= 8 && pct < 99) {
      const rawEta = elapsed * ((100 - pct) / pct);
      if (etaSecondsRef.current === null) {
        etaSecondsRef.current = rawEta;
      } else {
        // Count down; never climb within a stage (asymptotic bar made raw ETA rise)
        etaSecondsRef.current = Math.max(0, Math.min(etaSecondsRef.current - 1, rawEta));
      }
      setEtaLabel(formatEta(etaSecondsRef.current));
      return;
    }

    setEtaLabel(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- recompute on tick / chunk / bar changes
  }, [loading, progressStep, currentChunk, totalChunks, liveTtsCompleted, stageBarPercent, etaTick, progressStageKey, encodePercent]);

  const getStageProgressLabel = () => {
    if (loadingMode === "extract") return "Extração";
    if (progressStep === "pre_tts" || progressStep === "chunks") return "Preparação";
    if (progressStep === "tts") return "Narração";
    if (progressStep === "encoding") return "Codificação";
    if (progressStep === "done") return "Concluído";
    return "Etapa atual";
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-blue-500/30 selection:text-white relative flex flex-col">
      {/* Background Decor — clipped so negative offsets don't force page scrollbars */}
      <div className="pointer-events-none absolute inset-0 overflow-clip" aria-hidden>
        <div className="absolute top-[-100px] left-[-100px] w-[500px] h-[500px] bg-blue-600/15 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-100px] right-[-100px] w-[600px] h-[600px] bg-purple-600/15 rounded-full blur-[150px]" />
        <div className="absolute top-[20%] right-[10%] w-[300px] h-[300px] bg-emerald-500/8 rounded-full blur-[100px]" />
      </div>

      <AnimatePresence>
        {isRestoringSession && (
          <motion.div
            key="session-restore"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/70 backdrop-blur-md"
            role="status"
            aria-live="polite"
            aria-label="Restaurando sessão"
          >
            <div className="flex flex-col items-center gap-4 rounded-3xl border border-white/10 bg-slate-900/80 px-8 py-7 shadow-2xl">
              <div className="relative h-20 w-20">
                <svg className="h-20 w-20 -rotate-90" viewBox="0 0 80 80" aria-hidden>
                  <circle
                    cx="40"
                    cy="40"
                    r="34"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="6"
                    className="text-white/10"
                  />
                  <circle
                    cx="40"
                    cy="40"
                    r="34"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="6"
                    strokeLinecap="round"
                    className="text-blue-400 transition-[stroke-dashoffset] duration-200 ease-out"
                    strokeDasharray={2 * Math.PI * 34}
                    strokeDashoffset={2 * Math.PI * 34 * (1 - Math.min(100, Math.max(0, restorePercent)) / 100)}
                  />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-sm font-bold tabular-nums text-white">
                  {Math.min(100, Math.max(0, Math.round(restorePercent)))}%
                </span>
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-white">Restaurando sessão</p>
                <p className="mt-1 text-xs text-slate-400">
                  Carregando documentos e progresso salvos…
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hidden audio tag linked with React state */}
      {resultDoc?.audioUrl && (
        <audio
          key={resultDoc.id}
          ref={audioRef}
          src={resultDoc.audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleAudioEnded}
        />
      )}

      {/* Modern minimalist Header */}
      <header className="relative z-10 backdrop-blur-md bg-slate-950/40 border-b border-white/10 px-6 py-4">
        <div className="w-full flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 text-white p-2.5 rounded-xl shadow-lg shadow-blue-500/10">
              <Sparkles className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-300">
                Aura Converter
              </h1>
              <p className="text-xs text-slate-400">PDF e EPUB em narração natural com IA</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => onManageModels?.()}
            title="Gerenciar modelos TTS"
            className="text-xs font-mono text-slate-300 bg-white/5 border border-white/10 px-3 py-1.5 rounded-full hover:bg-white/10 hover:text-white hover:border-white/20 transition-colors"
          >
            {ttsEngine === "kokoro"
              ? `Kokoro · ${kokoroBackend === "mlx" ? "MLX" : "ONNX fp32"} · ${
                  kokoroDevice === "gpu" ? "GPU" : "CPU"
                }`
              : "Qwen3 TTS · local"}
          </button>
        </div>
      </header>

      <div className="relative z-10 flex-1 flex min-h-0">
        {/* Side navigation */}
        <aside className="w-56 shrink-0 border-r border-white/10 bg-slate-950/50 backdrop-blur-md px-3 py-6 flex flex-col gap-1">
          {(
            [
              { id: "narrate" as const, label: "Narrar", icon: Mic2 },
              { id: "extract-cover" as const, label: "Extrair capa", icon: ImageIcon },
              { id: "mp3-to-m4b" as const, label: "MP3 → M4B", icon: ArrowRightLeft },
              { id: "m4b-to-mp3" as const, label: "M4B → MP3", icon: FileAudio },
            ] as const
          ).map((item) => {
            const Icon = item.icon;
            const active = appMode === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setAppMode(item.id);
                  setError(null);
                }}
                className={`flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors text-left ${
                  active
                    ? "bg-blue-500/15 text-white border border-blue-500/30"
                    : "text-slate-400 hover:text-white hover:bg-white/5 border border-transparent"
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                {item.label}
              </button>
            );
          })}
        </aside>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-10 relative z-10 overflow-y-auto">
        {appMode === "extract-cover" ? (
          <ExtractCoverPanel />
        ) : appMode === "mp3-to-m4b" ? (
          <Mp3ToM4bPanel />
        ) : appMode === "m4b-to-mp3" ? (
          <M4bToMp3Panel />
        ) : (
        <AnimatePresence mode="wait">
          {/* Main Error Banner */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-8 p-4 bg-rose-950/45 border border-rose-500/30 backdrop-blur-xl rounded-2xl flex items-start gap-3 text-rose-200"
            >
              <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
              <div className="flex-1 text-sm">
                <span className="font-semibold text-rose-100">Ops! Algo deu errado: </span>
                {error}
              </div>
              <button 
                onClick={() => setError(null)} 
                className="text-xs font-medium text-rose-300 hover:text-rose-100 px-2 py-1 rounded-md hover:bg-rose-500/10 transition-colors"
              >
                Dispensar
              </button>
            </motion.div>
          )}

          {/* Loading Serene Screen */}
          {loading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className={`${loadingMode === "narrate" && processingPreviewText ? "max-w-5xl" : "max-w-2xl"} mx-auto bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl shadow-2xl p-8 sm:p-12 text-center`}
            >
              <div className={`grid ${loadingMode === "narrate" && processingPreviewText ? "lg:grid-cols-2 gap-8 items-start text-left" : ""}`}>
                <div>
                  <div className="relative w-24 h-24 mx-auto mb-8 flex items-center justify-center">
                    <div className="absolute inset-0 border-4 border-blue-500/10 rounded-full animate-pulse" />
                    <div className="absolute inset-0 border-4 border-t-blue-500 rounded-full animate-spin [animation-duration:1.2s]" />
                    <FileAudio className="w-10 h-10 text-blue-400 animate-bounce" />
                  </div>

                  <h3 className="text-xl font-bold text-white mb-2 text-center">
                    {loadingMode === "extract"
                      ? (processingFileTotal > 1 ? "Extraindo textos" : "Extraindo texto")
                      : (processingFileTotal > 1 ? "Narrando seus textos" : "Narrando seu texto")}
                  </h3>

                  {processingFileName && (
                    <div className="max-w-md mx-auto mb-4 rounded-2xl border border-blue-500/25 bg-blue-500/10 px-4 py-3 text-center">
                      {processingFileTotal > 1 && (
                        <p className="text-[11px] font-semibold uppercase tracking-wider text-blue-300/90 mb-1">
                          Arquivo {processingFileIndex} de {processingFileTotal}
                        </p>
                      )}
                      <p className="text-sm font-semibold text-white truncate" title={processingFileName}>
                        {processingFileName}
                      </p>
                    </div>
                  )}

                  <p className="text-slate-300 text-sm max-w-md mx-auto mb-8 text-center">
                    {loadingMode === "extract"
                      ? "Em seguida você poderá revisar e editar o texto de cada arquivo antes da narração."
                      : "Cada áudio concluído é salvo automaticamente na pasta Downloads."}
                  </p>

                  {/* Progress bar: 0–100% within the current stage only */}
                  <div className="max-w-md mx-auto mb-8">
                    <div className="flex justify-between text-xs text-slate-400 mb-2">
                      <span>Progresso — {getStageProgressLabel()}</span>
                      <span className="font-mono font-bold text-blue-400">{getProgressPercentage()}%</span>
                    </div>
                    <div className="w-full bg-slate-900 border border-white/10 rounded-full h-2 overflow-hidden">
                      <motion.div
                        key={progressStageKey}
                        className="bg-gradient-to-r from-blue-500 to-indigo-500 h-full rounded-full shadow-lg shadow-blue-500/20"
                        initial={{ width: "0%" }}
                        animate={{ width: `${getProgressPercentage()}%` }}
                        transition={{ ease: "easeOut", duration: 0.25 }}
                      />
                    </div>
                    {etaLabel && (
                      <div className="mt-2 flex items-center justify-center gap-1.5 text-[11px] text-slate-400">
                        <Clock className="w-3 h-3 text-blue-400/80" />
                        <span className="font-medium text-slate-300">{etaLabel}</span>
                      </div>
                    )}
                  </div>

                  {/* Steps stay on the left only when there is no right column */}
                  {!(loadingMode === "narrate" && processingPreviewText) && (
                    <div className="bg-slate-900/40 border border-white/10 rounded-2xl p-5 max-w-md mx-auto text-left space-y-4 mb-8">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono text-slate-400">Etapas do Processo</span>
                        {processingFileTotal > 1 && (
                          <span className="text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2 py-0.5 rounded-full">
                            {processingFileIndex}/{processingFileTotal}
                          </span>
                        )}
                      </div>
                      <hr className="border-white/5" />
                      {loadingMode === "extract" ? (
                        <div className="flex items-center gap-3 text-sm">
                          <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
                          <span className="text-white font-medium animate-pulse">
                            Extraindo texto do arquivo atual
                          </span>
                        </div>
                      ) : null}
                    </div>
                  )}

                  {/* Stop Button */}
                  {loadingMode === "narrate" && progressStep === "tts" && (
                    <div className="max-w-md mx-auto mb-6">
                      <button
                        onClick={handleStopNarration}
                        disabled={isStopping}
                        className="w-full bg-red-500/10 hover:bg-red-500/20 active:bg-red-500/30 border border-red-500/30 hover:border-red-500/50 disabled:opacity-50 disabled:cursor-not-allowed text-red-400 py-3 px-6 rounded-2xl text-sm font-semibold transition-all flex items-center justify-center gap-2 shadow-lg shadow-red-500/5 cursor-pointer"
                      >
                        {isStopping ? (
                          <>
                            <Loader2 className="w-4.5 h-4.5 animate-spin" />
                            <span>Cancelando…</span>
                          </>
                        ) : (
                          <>
                            <Square className="w-4 h-4 fill-current" />
                            <span>Cancelar narração</span>
                          </>
                        )}
                      </button>
                      <p className="text-[11px] text-slate-400 mt-2 text-center">
                        Cancela sem gerar áudio parcial. Os blocos já narrados ficam em cache para retomar depois.
                        {processingFileTotal > 1
                          ? " Os arquivos seguintes não serão narrados."
                          : ""}
                      </p>
                    </div>
                  )}

                  {!(loadingMode === "narrate" && processingPreviewText) && (
                    <p className="text-xs italic text-slate-400 mt-6 text-center">
                      "A leitura nos dá um lugar para ir quando temos que ficar onde estamos."
                    </p>
                  )}
                </div>

                {loadingMode === "narrate" && processingPreviewText && (
                  <div className="space-y-4">
                    <div className="bg-slate-900/40 border border-white/10 rounded-2xl p-5 text-left space-y-4">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-mono text-slate-400">Etapas do Processo</span>
                        <div className="flex items-center gap-2 shrink-0">
                          {processingFileTotal > 1 && (
                            <span className="text-xs font-semibold bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 px-2 py-0.5 rounded-full">
                              Arquivo {processingFileIndex}/{processingFileTotal}
                            </span>
                          )}
                          {(progressStep === "tts" || (progressStep === "encoding" && totalChunks > 0)) && totalChunks > 0 && (
                            <span className="text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2 py-0.5 rounded-full animate-pulse">
                              {progressStep === "encoding"
                                ? `${currentChunk} de ${totalChunks} partes`
                                : `Parte ${currentChunk} de ${totalChunks}`}
                            </span>
                          )}
                        </div>
                      </div>
                      <hr className="border-white/5" />

                      {/* Step 1 skipped — text already reviewed */}
                      <div className="flex items-center gap-3 text-sm">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        <span className="text-slate-400">Texto revisado e confirmado</span>
                      </div>

                      {/* Step 2: Content preparation */}
                      <div className="flex items-center gap-3 text-sm">
                        {["pre_tts", "chunks"].includes(progressStep) ? (
                          <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
                        ) : ["tts", "encoding", "done"].includes(progressStep) ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-white/20 shrink-0" />
                        )}
                        <span className={["pre_tts", "chunks"].includes(progressStep) ? "text-white font-medium animate-pulse" : "text-slate-400"}>
                          Estruturando e dividindo conteúdo
                        </span>
                      </div>

                      {/* Step 3: Synthesis */}
                      <div className="flex items-center gap-3 text-sm">
                        {["pre_tts", "chunks"].includes(progressStep) ? (
                          <div className="w-4 h-4 rounded-full border border-white/20 shrink-0" />
                        ) : progressStep === "tts" ? (
                          <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
                        ) : (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        )}
                        <span className={progressStep === "tts" ? "text-white font-medium animate-pulse" : ["pre_tts", "chunks"].includes(progressStep) ? "text-slate-500" : "text-slate-400"}>
                          Narrando com Inteligência Artificial (
                            {ttsEngine === "kokoro" ? "Kokoro" : "Qwen3 TTS"}
                          )
                        </span>
                      </div>

                      {/* Step 4: Encoding */}
                      <div className="flex items-center gap-3 text-sm">
                        {["pre_tts", "chunks", "tts"].includes(progressStep) ? (
                          <div className="w-4 h-4 rounded-full border border-white/20 shrink-0" />
                        ) : progressStep === "encoding" ? (
                          <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
                        ) : (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        )}
                        <span className={progressStep === "encoding" ? "text-white font-medium animate-pulse" : ["pre_tts", "chunks", "tts"].includes(progressStep) ? "text-slate-500" : "text-slate-400"}>
                          Codificando áudio ({outputFormat === "m4b" ? "M4B" : "MP3"})
                          {progressStep === "encoding" && encodePercent != null
                            ? ` — ${Math.round(encodePercent)}%`
                            : ""}
                        </span>
                      </div>
                    </div>
                    {savedFilesCount > 0 && lastSavedFileName && (
                      <div className="rounded-2xl border border-emerald-500/25 bg-emerald-500/10 px-4 py-2.5 text-center">
                        <p className="text-[11px] font-semibold text-emerald-300 flex items-center justify-center gap-1.5">
                          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                          Salvo em Downloads
                          {savedFilesCount > 1 ? ` (${savedFilesCount})` : ""}
                        </p>
                        <p className="text-xs text-emerald-100/90 truncate mt-0.5" title={lastSavedFileName}>
                          {lastSavedFileName}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ) : showTextReview ? (
            /* Editable text review before TTS */
            <motion.div
              key="review"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="max-w-4xl mx-auto bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl shadow-2xl p-6 sm:p-8"
            >
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-5">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-blue-400" />
                    {documents.length > 1
                      ? "Revisar textos antes de narrar"
                      : "Revisar texto antes de narrar"}
                  </h3>
                  <p className="text-sm text-slate-400 mt-1">
                    {documents.length > 1
                      ? "Valide o texto de cada arquivo. A narração usará exatamente o conteúdo revisado."
                      : "Edite o conteúdo como quiser. A narração usará exatamente este texto."}
                    {reviewDoc?.pagesNarrated ? (
                      <span className="block mt-1 text-xs text-slate-500">
                        Intervalo: {reviewDoc.pagesNarrated}
                      </span>
                    ) : null}
                  </p>
                </div>
                <button
                  onClick={() => {
                    setShowTextReview(false);
                  }}
                  className="inline-flex items-center gap-1.5 text-xs font-bold text-slate-400 hover:text-white transition-colors cursor-pointer"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  Voltar
                </button>
              </div>

              {documents.length > 1 && (
                <div className="flex gap-2 overflow-x-auto pb-3 mb-4 scrollbar-thin scrollbar-thumb-white/10">
                  {documents.map((doc, index) => {
                    const isActive = (reviewDocId ?? documents[0]?.id) === doc.id;
                    const hasText = Boolean(doc.editableText.trim());
                    return (
                      <button
                        key={doc.id}
                        type="button"
                        onClick={() => setReviewDocId(doc.id)}
                        className={`shrink-0 max-w-[220px] rounded-xl border px-3 py-2 text-left transition-all cursor-pointer ${
                          isActive
                            ? "border-blue-500/60 bg-blue-500/15 text-white"
                            : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/10"
                        }`}
                        title={doc.file.name}
                      >
                        <span className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                          Arquivo {index + 1}
                          {!hasText ? " · vazio" : ""}
                        </span>
                        <span className="block text-xs font-semibold truncate">{doc.file.name}</span>
                      </button>
                    );
                  })}
                </div>
              )}

              <textarea
                value={reviewDoc?.editableText ?? ""}
                onChange={(e) => {
                  const id = reviewDocId ?? documents[0]?.id;
                  if (!id) return;
                  updateEditableText(id, e.target.value);
                }}
                rows={18}
                className="w-full min-h-[320px] bg-slate-950/60 border border-white/10 focus:border-blue-500/60 rounded-2xl p-5 text-sm text-slate-200 leading-relaxed font-serif whitespace-pre-wrap outline-none resize-y scrollbar-thin scrollbar-thumb-white/10"
                placeholder="Texto extraído..."
              />

              <div className="mt-5 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => {
                    setShowTextReview(false);
                  }}
                  className="sm:flex-1 bg-white/5 hover:bg-white/10 border border-white/15 text-slate-200 py-3.5 rounded-2xl text-sm font-semibold transition-all cursor-pointer"
                >
                  Cancelar
                </button>
                <button
                  onClick={handleReExtract}
                  disabled={!allDocsReady || anyDocInfoLoading}
                  className="sm:flex-1 bg-white/5 hover:bg-white/10 border border-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 py-3.5 rounded-2xl text-sm font-semibold flex items-center justify-center gap-2 transition-all cursor-pointer"
                >
                  <RotateCcw className="w-4 h-4" />
                  Extrair novamente
                </button>
                <button
                  onClick={handleConfirmNarration}
                  disabled={documents.some((d) => !d.editableText.trim())}
                  className="sm:flex-[2] bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white py-3.5 rounded-2xl font-bold text-sm shadow-lg shadow-blue-900/35 flex items-center justify-center gap-2.5 transition-all cursor-pointer"
                >
                  <Sparkles className="w-5 h-5" />
                  <span>
                    {documents.length > 1
                      ? `Narrar todos (${documents.length} arquivos)`
                      : "Narrar este texto"}
                  </span>
                </button>
              </div>
            </motion.div>
          ) : !hasResults ? (
            /* Upload & Configuration Panel */
            <motion.div
              key="setup"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start"
            >
              {/* Left Column: Upload or Document Detail */}
              <div className="lg:col-span-5 space-y-6">
                <div className="bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl p-6 shadow-xl">
                  <div className="flex items-center justify-between gap-3 mb-4">
                    <h2 className="text-base font-bold text-white flex items-center gap-2">
                      <FileText className="w-4.5 h-4.5 text-blue-400" />
                      <span>1. Documentos PDF ou EPUB</span>
                    </h2>
                    {documents.length > 0 && (
                      <button
                        type="button"
                        onClick={handleClearAllFiles}
                        className="shrink-0 inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 hover:bg-rose-500/15 hover:border-rose-500/30 px-2.5 py-1 text-[11px] font-semibold text-slate-300 hover:text-rose-300 transition-colors cursor-pointer"
                        title="Remover todos os arquivos"
                      >
                        <Trash2 className="w-3 h-3" />
                        Limpar todos
                      </button>
                    )}
                  </div>

                  {documents.length === 0 ? (
                    // Drag and Drop Zone
                    <div
                      onDragEnter={handleDrag}
                      onDragOver={handleDrag}
                      onDragLeave={handleDrag}
                      onDrop={handleDrop}
                      className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
                        dragActive 
                          ? "border-blue-500 bg-blue-500/10" 
                          : "border-white/10 hover:border-blue-500/50 hover:bg-white/5"
                      }`}
                    >
                      <input
                        type="file"
                        id="file-upload"
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        accept="application/pdf,application/epub+zip,.epub,.pdf"
                        multiple
                        onChange={handleFileChange}
                      />
                      <div className="bg-blue-500/10 text-blue-400 w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <FileText className="w-6 h-6" />
                      </div>
                      <p className="text-sm font-semibold text-slate-200 mb-1">
                        Arraste PDFs ou EPUBs aqui
                      </p>
                      <p className="text-xs text-slate-400 mb-4">
                        um ou vários arquivos — clique para selecionar
                      </p>
                      <span className="inline-block bg-white/10 hover:bg-white/25 border border-white/15 rounded-lg px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors">
                        Escolher Arquivos
                      </span>
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      <div className="max-h-[15.75rem] space-y-1.5 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
                        {documents.map((doc, index) => {
                          const isActive = activeDocId === doc.id;
                          return (
                            <div
                              key={doc.id}
                              role="button"
                              tabIndex={0}
                              onClick={() => setActiveDocId(doc.id)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") {
                                  e.preventDefault();
                                  setActiveDocId(doc.id);
                                }
                              }}
                              className={`border rounded-xl px-2.5 py-1.5 cursor-pointer transition-all ${
                                isActive
                                  ? "border-blue-500/50 bg-blue-500/10"
                                  : "border-white/15 bg-white/5 hover:border-white/25"
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex items-center gap-2 overflow-hidden min-w-0">
                                  <div className="bg-gradient-to-br from-red-500/20 to-red-600/20 border border-red-500/30 text-red-400 p-1.5 rounded-lg shrink-0">
                                    {doc.fileType === "epub" ? (
                                      <Book className="w-3.5 h-3.5 text-indigo-400" />
                                    ) : (
                                      <FileText className="w-3.5 h-3.5" />
                                    )}
                                  </div>
                                  <div className="overflow-hidden min-w-0">
                                    <h4 className="text-xs font-semibold text-white truncate leading-tight" title={doc.file.name}>
                                      <span className="text-slate-500 font-medium mr-1">{index + 1}.</span>
                                      {doc.file.name}
                                    </h4>
                                    <p className="text-[11px] text-slate-400 leading-tight mt-0.5">
                                      {(doc.file.size / (1024 * 1024)).toFixed(2)} MB • {doc.fileType.toUpperCase()}
                                      {doc.docInfoLoading ? " • lendo…" : ""}
                                    </p>
                                  </div>
                                </div>
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleRemoveFile(doc.id);
                                  }}
                                  className="bg-white/5 hover:bg-rose-500/20 text-slate-300 hover:text-rose-400 border border-white/10 rounded-lg p-1.5 transition-all shadow-sm shrink-0 cursor-pointer"
                                  title="Remover arquivo"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </div>
                              {doc.narrationProgress &&
                                doc.narrationProgress.completed > 0 &&
                                doc.narrationProgress.total > 0 && (
                                <div className="mt-1.5 flex items-center gap-2">
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between gap-2 mb-0.5">
                                      <span className="text-[10px] font-semibold text-amber-200/90">
                                        Progresso {doc.narrationProgress.completed}/{doc.narrationProgress.total}
                                      </span>
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleClearNarrationProgress(doc.id);
                                        }}
                                        className="text-[10px] font-semibold text-slate-400 hover:text-rose-300 transition-colors cursor-pointer"
                                        title="Limpar blocos narrados em cache"
                                      >
                                        Limpar progresso
                                      </button>
                                    </div>
                                    <div className="h-1 rounded-full bg-white/10 overflow-hidden">
                                      <div
                                        className="h-full rounded-full bg-amber-400/80 transition-all"
                                        style={{
                                          width: `${Math.min(
                                            100,
                                            Math.round(
                                              (doc.narrationProgress.completed /
                                                doc.narrationProgress.total) *
                                                100
                                            )
                                          )}%`,
                                        }}
                                      />
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>

                      <label className="relative flex items-center justify-center gap-2 rounded-xl border border-dashed border-white/15 hover:border-blue-500/40 hover:bg-white/5 px-3 py-2 text-xs font-semibold text-slate-300 hover:text-white transition-all cursor-pointer">
                        <Plus className="w-3.5 h-3.5" />
                        Adicionar mais arquivos
                        <input
                          type="file"
                          className="absolute inset-0 opacity-0 cursor-pointer"
                          accept="application/pdf,application/epub+zip,.epub,.pdf"
                          multiple
                          onChange={handleFileChange}
                        />
                      </label>
                    </div>
                  )}
                </div>

                {/* 2. Voice */}
                <div className="bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl p-6 shadow-xl">
                  <h2 className="text-base font-bold text-white mb-1 flex items-center gap-2">
                    <Music className="w-4.5 h-4.5 text-blue-400" />
                    <span>2. Voz do Narrador</span>
                  </h2>
                  <p className="text-xs text-slate-400 mb-4">
                    {ttsEngine === "kokoro"
                      ? `Kokoro (${kokoroBackend === "mlx" ? "MLX" : "ONNX fp32"} · ${
                          kokoroDevice === "gpu" ? "GPU" : "CPU"
                        }). A prévia fala no idioma do livro (${previewLanguage.label}). Passe o mouse na voz e use ↺ para excluir a prévia salva e gerar outra.`
                      : `Motor Qwen3. A prévia em ${previewLanguage.label} vira a referência de voz da narração. Passe o mouse na voz e use ↺ para excluir a prévia salva e gerar outra.`}
                  </p>

                  {previewError && (
                    <div className="mb-4 flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                      <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                      <span>{previewError}</span>
                    </div>
                  )}

                  <div className="space-y-4">
                    {(["Feminino", "Masculino"] as const).map((gender) => {
                      const genderVoices = voices.filter((v) => v.gender === gender);
                      return (
                        <div key={gender}>
                          <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500 mb-2 px-0.5">
                            {gender}
                          </p>
                          <div
                            className="grid grid-cols-1 sm:grid-cols-2 gap-1.5"
                            role="radiogroup"
                            aria-label={`Vozes ${gender.toLowerCase()}s`}
                          >
                            {genderVoices.map((voice) => {
                              const isSelected = selectedVoice === voice.id;
                              const isLoadingPreview = previewLoadingVoice === voice.id;
                              const isDeletingPreview = previewDeletingVoice === voice.id;
                              const isPlayingPreview = previewPlayingVoice === voice.id;
                              const previewBusy =
                                isLoadingPreview ||
                                isDeletingPreview ||
                                (!!previewLoadingVoice && previewLoadingVoice !== voice.id) ||
                                (!!previewDeletingVoice && previewDeletingVoice !== voice.id);
                              return (
                                <div
                                  key={voice.id}
                                  role="radio"
                                  aria-checked={isSelected}
                                  tabIndex={0}
                                  title={voice.description}
                                  onClick={() => setSelectedVoice(voice.id)}
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter" || e.key === " ") {
                                      e.preventDefault();
                                      setSelectedVoice(voice.id);
                                    }
                                  }}
                                  className={`group relative flex items-center gap-2.5 rounded-xl border px-2.5 py-2 cursor-pointer transition-all ${
                                    isSelected
                                      ? "border-blue-500/60 bg-blue-500/10 shadow-[inset_0_0_0_1px_rgba(59,130,246,0.15)]"
                                      : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                                  }`}
                                >
                                  <span className="text-base leading-none shrink-0 w-6 text-center" aria-hidden>
                                    {voice.icon}
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <span className={`block text-[13px] font-semibold truncate ${isSelected ? "text-white" : "text-slate-200"}`}>
                                      {voice.name}
                                    </span>
                                    <span className="block text-[10px] text-slate-500 truncate leading-tight mt-0.5">
                                      {voice.description}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-0.5 shrink-0">
                                    <button
                                      type="button"
                                      title={
                                        isDeletingPreview
                                          ? "Regenerando prévia..."
                                          : `Excluir prévia salva de ${voice.name} e gerar outra`
                                      }
                                      aria-label={
                                        isDeletingPreview
                                          ? "Regenerando prévia"
                                          : `Excluir e regenerar prévia de ${voice.name}`
                                      }
                                      disabled={previewBusy}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        void deleteAndRegenerateVoicePreview(voice.id);
                                      }}
                                      className={`w-7 h-7 rounded-lg flex items-center justify-center border transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer bg-transparent border-transparent text-slate-500 hover:!opacity-100 group-hover:bg-white/5 group-hover:border-white/10 hover:text-rose-300 ${
                                        isSelected ? "opacity-70" : "opacity-0 group-hover:opacity-70"
                                      }`}
                                    >
                                      {isDeletingPreview ? (
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                      ) : (
                                        <RotateCcw className="w-3 h-3" />
                                      )}
                                    </button>
                                    <button
                                      type="button"
                                      title={
                                        isPlayingPreview
                                          ? "Parar prévia"
                                          : `Ouvir ${voice.name} em ${previewLanguage.label}`
                                      }
                                      aria-label={
                                        isPlayingPreview
                                          ? "Parar prévia"
                                          : `Ouvir exemplo de ${voice.name} em ${previewLanguage.label}`
                                      }
                                      disabled={previewBusy}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        void playVoicePreview(voice.id);
                                      }}
                                      className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 border transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer ${
                                        isPlayingPreview
                                          ? "bg-blue-500/20 border-blue-400/40 text-blue-300"
                                          : "bg-transparent border-transparent text-slate-500 opacity-70 group-hover:opacity-100 group-hover:bg-white/5 group-hover:border-white/10 hover:text-white"
                                      }`}
                                    >
                                      {isLoadingPreview ? (
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                      ) : isPlayingPreview ? (
                                        <Square className="w-2.5 h-2.5 fill-current" />
                                      ) : (
                                        <Play className="w-3 h-3 fill-current" />
                                      )}
                                    </button>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Right Column: Pages & Extract */}
              <div className="lg:col-span-7 space-y-6">
                {/* Page Controls Panel (Only shows if file loaded) */}
                {activeDoc && (
                  <motion.div
                    key={activeDoc.id}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl p-6 shadow-xl space-y-4"
                  >
                    <h2 className="text-base font-bold text-white mb-2 flex items-center gap-2">
                      <Clock className="w-4.5 h-4.5 text-blue-400" />
                      <span>
                        3.{" "}
                        {activeDoc.fileType === "pdf" ? "Páginas do PDF" : "Capítulos do EPUB"}
                      </span>
                    </h2>
                    {documents.length > 1 && (
                      <p className="text-xs text-slate-400 -mt-2 truncate" title={activeDoc.file.name}>
                        Configurando: <span className="text-slate-200 font-medium">{activeDoc.file.name}</span>
                      </p>
                    )}

                    <div className="space-y-1.5">
                      <label className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                        <Languages className="w-3.5 h-3.5 text-blue-400" />
                        Idioma do livro
                      </label>
                      <select
                        value={activeDoc.narrationLanguage}
                        onChange={(e) =>
                          setDocNarrationLanguage(
                            activeDoc.id,
                            resolveNarrationLanguageId(e.target.value)
                          )
                        }
                        className="w-full bg-slate-900/50 border border-white/15 text-white focus:border-blue-500 rounded-xl px-4 py-3 text-sm font-semibold outline-none"
                        aria-label="Idioma do livro e da narração"
                      >
                        {NARRATION_LANGUAGES.map((lang) => (
                          <option key={lang.id} value={lang.id}>
                            {lang.flag} {lang.label}
                          </option>
                        ))}
                      </select>
                      <p className="text-[11px] text-slate-400 leading-normal">
                        {ttsEngine === "kokoro" &&
                        !getNarrationLanguage(activeDoc.narrationLanguage).kokoroNative
                          ? "O padrão segue o idioma do sistema. Kokoro não tem voz nativa nesta língua — a pronúncia pode ficar em inglês."
                          : "O padrão segue o idioma do sistema. A narração usa este idioma no Qwen e no Kokoro."}
                      </p>
                    </div>

                    {activeDoc.docInfoLoading ? (
                      <div className="flex items-center gap-2 text-sm text-slate-400">
                        <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                        Lendo estrutura do documento...
                      </div>
                    ) : (
                      <>
                        {activeDoc.docInfoMessage && (
                          <p className="text-[11px] text-amber-200/90 bg-amber-500/10 border border-amber-500/20 rounded-xl px-3 py-2 leading-relaxed">
                            {activeDoc.docInfoMessage}
                            {activeDoc.fileType === "pdf" && activeDoc.pdfPageCount != null
                              ? ` Use números de 1 a ${activeDoc.pdfPageCount}.`
                              : ""}
                          </p>
                        )}

                        {activeDoc.fileType === "epub" && activeDoc.epubChapters.length > 0 ? (
                          <div className="space-y-4">
                            <div className="grid grid-cols-1 gap-4">
                              <div className="space-y-1.5">
                                <label className="text-xs font-bold text-slate-300">Capítulo / seção inicial</label>
                                <select
                                  value={activeDoc.startPage}
                                  onChange={(e) => {
                                    const next = parseInt(e.target.value, 10) || 1;
                                    const currentEnd = activeDoc.endPage
                                      ? parseInt(activeDoc.endPage, 10)
                                      : next;
                                    updateDoc(activeDoc.id, {
                                      startPage: next,
                                      endPage:
                                        !activeDoc.endPage || currentEnd < next
                                          ? String(next)
                                          : activeDoc.endPage,
                                    });
                                  }}
                                  className="w-full bg-slate-900/50 border border-white/15 text-white focus:border-blue-500 rounded-xl px-4 py-3 text-sm font-semibold outline-none"
                                >
                                  {activeDoc.epubChapters.map((ch) => (
                                    <option key={ch.id || ch.index} value={ch.index}>
                                      {ch.index}. {ch.title}
                                    </option>
                                  ))}
                                </select>
                              </div>
                              <div className="space-y-1.5">
                                <label className="text-xs font-bold text-slate-300">Capítulo / seção final</label>
                                <select
                                  value={activeDoc.endPage || String(activeDoc.startPage)}
                                  onChange={(e) =>
                                    updateDoc(activeDoc.id, { endPage: e.target.value })
                                  }
                                  className="w-full bg-slate-900/50 border border-white/15 text-white focus:border-blue-500 rounded-xl px-4 py-3 text-sm font-semibold outline-none"
                                >
                                  {activeDoc.epubChapters
                                    .filter((ch) => ch.index >= activeDoc.startPage)
                                    .map((ch) => (
                                      <option key={`end-${ch.id || ch.index}`} value={ch.index}>
                                        {ch.index}. {ch.title}
                                      </option>
                                    ))}
                                </select>
                              </div>
                            </div>
                            <p className="text-[11px] text-slate-400 leading-normal">
                              Em EPUB, “11” é o 11º capítulo da estrutura — não a página impressa 11. Um único capítulo pode ter o equivalente a muitas páginas.
                            </p>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div className="space-y-1.5">
                                <label className="text-xs font-bold text-slate-300">Página inicial</label>
                                <input
                                  type="number"
                                  min={1}
                                  max={activeDoc.pdfPageCount ?? undefined}
                                  value={activeDoc.startPage}
                                  onChange={(e) =>
                                    updateDoc(activeDoc.id, {
                                      startPage: Math.max(1, parseInt(e.target.value) || 1),
                                    })
                                  }
                                  className="w-full bg-slate-900/50 border border-white/15 text-white focus:border-blue-500 focus:bg-slate-900 rounded-xl px-4 py-3 text-sm font-semibold outline-none transition-all"
                                />
                              </div>
                              <div className="space-y-1.5">
                                <label className="text-xs font-bold text-slate-300">
                                  Página final <span className="text-slate-400 font-normal">(vazio = só a inicial)</span>
                                </label>
                                <input
                                  type="number"
                                  placeholder={String(activeDoc.startPage)}
                                  min={activeDoc.startPage}
                                  max={activeDoc.pdfPageCount ?? undefined}
                                  value={activeDoc.endPage}
                                  onChange={(e) =>
                                    updateDoc(activeDoc.id, { endPage: e.target.value })
                                  }
                                  className="w-full bg-slate-900/50 border border-white/15 text-white focus:border-blue-500 focus:bg-slate-900 rounded-xl px-4 py-3 text-sm font-semibold outline-none transition-all placeholder:text-slate-500"
                                />
                              </div>
                            </div>
                            <p className="text-[11px] text-slate-400 leading-normal">
                              São páginas do arquivo PDF (como no leitor). Ex.: 11 a 12 extrai só essas duas folhas.
                            </p>
                          </div>
                        )}

                        {/* Cover page + preview */}
                        <div className="pt-2 border-t border-white/10 space-y-3">
                          <div className="flex items-center justify-between gap-3">
                            <h3 className="text-sm font-bold text-white flex items-center gap-2">
                              <ImageIcon className="w-4 h-4 text-blue-400" />
                              Capa do livro
                            </h3>
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                              <input
                                type="checkbox"
                                checked={activeDoc.exportCover}
                                onChange={(e) =>
                                  updateDoc(activeDoc.id, {
                                    exportCover: e.target.checked,
                                  })
                                }
                                className="rounded border-white/20 bg-slate-900 text-blue-500 focus:ring-blue-500/40"
                              />
                              Exportar capa
                            </label>
                          </div>
                          {!activeDoc.exportCover ? (
                            <p className="text-[11px] text-slate-400">
                              Sem JPEG da capa e sem artwork no M4B.
                            </p>
                          ) : activeDoc.fileType === "pdf" ? (
                            <div className="space-y-1.5">
                              <label className="text-xs font-bold text-slate-300">
                                Página da capa
                              </label>
                              <input
                                type="number"
                                min={1}
                                max={activeDoc.pdfPageCount ?? undefined}
                                value={activeDoc.coverPage}
                                placeholder="1"
                                onChange={(e) =>
                                  updateDoc(activeDoc.id, { coverPage: e.target.value })
                                }
                                className="w-full bg-slate-900/50 border border-white/15 text-white focus:border-blue-500 rounded-xl px-4 py-3 text-sm font-semibold outline-none"
                              />
                            </div>
                          ) : (
                            <p className="text-[11px] text-slate-400">
                              A capa é detectada automaticamente no EPUB (metadata OPF).
                            </p>
                          )}
                          <div className="grid grid-cols-3 gap-2">
                            {(
                              [
                                {
                                  key: "start" as const,
                                  label:
                                    activeDoc.fileType === "pdf"
                                      ? `Inicial (${activeDoc.startPage})`
                                      : `Início (§${activeDoc.startPage})`,
                                  url: activeDoc.startPreviewUrl,
                                  chapter:
                                    activeDoc.startChapterPreview ||
                                    (activeDoc.fileType === "epub"
                                      ? {
                                          title:
                                            activeDoc.epubChapters.find(
                                              (ch) => ch.index === activeDoc.startPage
                                            )?.title || `Seção ${activeDoc.startPage}`,
                                          text: "",
                                          index: activeDoc.startPage,
                                          total: activeDoc.epubChapters.length,
                                        }
                                      : null),
                                },
                                {
                                  key: "end" as const,
                                  label:
                                    activeDoc.fileType === "pdf"
                                      ? `Final (${activeDoc.endPage.trim() || activeDoc.startPage})`
                                      : `Fim (§${activeDoc.endPage.trim() || activeDoc.startPage})`,
                                  url: activeDoc.endPreviewUrl || activeDoc.startPreviewUrl,
                                  chapter:
                                    activeDoc.endChapterPreview ||
                                    activeDoc.startChapterPreview ||
                                    (activeDoc.fileType === "epub"
                                      ? {
                                          title:
                                            activeDoc.epubChapters.find(
                                              (ch) =>
                                                ch.index ===
                                                (parseInt(activeDoc.endPage, 10) ||
                                                  activeDoc.startPage)
                                            )?.title ||
                                            `Seção ${activeDoc.endPage.trim() || activeDoc.startPage}`,
                                          text: "",
                                          index:
                                            parseInt(activeDoc.endPage, 10) ||
                                            activeDoc.startPage,
                                          total: activeDoc.epubChapters.length,
                                        }
                                      : null),
                                },
                                {
                                  key: "cover" as const,
                                  label: "Capa",
                                  url: activeDoc.exportCover
                                    ? activeDoc.coverPreviewUrl
                                    : null,
                                  chapter: null as ChapterPreview | null,
                                },
                              ] as const
                            ).map((item) => (
                              <div
                                key={item.key}
                                className="rounded-xl border border-white/10 bg-black/20 p-2 flex flex-col items-center gap-1.5 min-h-[120px]"
                              >
                                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                                  {item.label}
                                </span>
                                {previewLoadingKeys[item.key] ? (
                                  <Loader2 className="w-5 h-5 text-blue-400 animate-spin my-auto" />
                                ) : item.url ? (
                                  <img
                                    src={item.url}
                                    alt={item.label}
                                    className="max-h-28 w-full object-contain rounded-md"
                                  />
                                ) : item.chapter ? (
                                  <div className="w-full flex-1 rounded-md bg-[#f7f4ef] text-[#1c1917] px-2 py-1.5 overflow-hidden">
                                    <p className="text-[9px] font-semibold text-[#8a847a] uppercase tracking-wide mb-0.5">
                                      Seção {item.chapter.index}
                                      {item.chapter.total
                                        ? ` / ${item.chapter.total}`
                                        : ""}
                                    </p>
                                    <p className="text-[11px] font-bold leading-snug line-clamp-2 mb-1">
                                      {item.chapter.title}
                                    </p>
                                    {item.chapter.text ? (
                                      <p className="text-[10px] leading-snug text-[#44403c] line-clamp-4">
                                        {item.chapter.text}
                                      </p>
                                    ) : (
                                      <p className="text-[10px] text-[#8a847a] italic">
                                        Carregando trecho…
                                      </p>
                                    )}
                                  </div>
                                ) : (
                                  <p className="text-[10px] text-slate-500 text-center px-1 my-auto leading-snug">
                                    {item.key === "cover"
                                      ? !activeDoc.exportCover
                                        ? "Não exportar"
                                        : coverPreviewMessage || "Sem capa"
                                      : "Sem preview"}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Output format */}
                        <div className="pt-2 border-t border-white/10 space-y-2">
                          <label className="text-xs font-bold text-slate-300">Formato de saída</label>
                          <div className="grid grid-cols-2 gap-2">
                            <button
                              type="button"
                              onClick={() => setOutputFormat("mp3")}
                              className={`rounded-xl border px-3 py-2.5 text-sm font-semibold transition-colors ${
                                outputFormat === "mp3"
                                  ? "border-blue-500/50 bg-blue-500/15 text-white"
                                  : "border-white/10 bg-white/5 text-slate-400 hover:text-white"
                              }`}
                            >
                              MP3
                            </button>
                            <button
                              type="button"
                              onClick={() => setOutputFormat("m4b")}
                              className={`rounded-xl border px-3 py-2.5 text-sm font-semibold transition-colors ${
                                outputFormat === "m4b"
                                  ? "border-violet-500/50 bg-violet-500/15 text-white"
                                  : "border-white/10 bg-white/5 text-slate-400 hover:text-white"
                              }`}
                            >
                              M4B
                            </button>
                          </div>
                          <p className="text-[11px] text-slate-500">
                            {outputFormat === "m4b"
                              ? activeDoc.exportCover
                                ? "M4B inclui a capa (PDF) ou as imagens do EPUB como artwork do audiobook."
                                : "M4B sem artwork de capa."
                              : activeDoc.exportCover
                                ? "MP3 + JPEG da capa (mesmo nome) na pasta Downloads."
                                : "Somente o MP3 em Downloads (sem JPEG da capa)."}
                          </p>
                        </div>
                      </>
                    )}
                  {documents.length > 0 ? (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-6 pt-6 border-t border-white/10"
                    >
                      <button
                        onClick={handleExtractForReview}
                        disabled={!allDocsReady || anyDocInfoLoading}
                        className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white py-4 rounded-2xl font-bold text-sm shadow-lg shadow-blue-900/35 flex items-center justify-center gap-2.5 transition-all hover:scale-[1.01] active:scale-[0.99] cursor-pointer"
                      >
                        {allDocsHaveExtractedText ? (
                          <BookOpen className="w-5 h-5" />
                        ) : (
                          <Sparkles className="w-5 h-5" />
                        )}
                        <span>
                          {allDocsHaveExtractedText
                            ? documents.length > 1
                              ? `Revisar textos (${documents.length})`
                              : "Revisar texto"
                            : documents.length > 1
                              ? `Extrair Textos para Revisar (${documents.length})`
                              : "Extrair Texto para Revisar"}
                        </span>
                      </button>
                    </motion.div>
                  ) : (
                    <div className="mt-6 p-4 bg-white/5 border border-white/10 rounded-2xl text-center">
                      <p className="text-xs text-slate-400">
                        Adicione um ou mais documentos PDF/EPUB acima para configurar páginas e iniciar a narração.
                      </p>
                    </div>
                  )}
                  </motion.div>
                )}
                {!activeDoc && (
                  <div className="bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl p-6 shadow-xl">
                    <h2 className="text-base font-bold text-white mb-2 flex items-center gap-2">
                      <Clock className="w-4.5 h-4.5 text-blue-400" />
                      <span>3. Páginas e capa</span>
                    </h2>
                    <p className="text-xs text-slate-400">
                      Adicione um PDF ou EPUB na coluna ao lado para escolher o intervalo e ver o preview das páginas.
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            /* Narration Result Panel */
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start"
            >
              <div className="lg:col-span-5 space-y-6">
                <div className="bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl p-6 shadow-xl">
                  <button 
                    onClick={resetAll}
                    className="mb-6 inline-flex items-center gap-1.5 text-xs font-bold text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
                  >
                    <ArrowLeft className="w-3.5 h-3.5" />
                    <span>Narrar Outro Trecho</span>
                  </button>

                  {batchDone && !resultDoc ? (
                    <div className="text-center py-4">
                      <div className="inline-flex bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-3 py-1.5 rounded-full text-xs font-bold gap-1.5 items-center mb-4">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                        <span>
                          {savedFilesCount > 1
                            ? `${savedFilesCount} ÁUDIOS SALVOS`
                            : "ÁUDIO SALVO"}
                        </span>
                      </div>
                      <h2 className="text-lg font-bold text-white px-2">
                        Narração concluída
                      </h2>
                      <p className="text-xs text-slate-400 mt-2 px-2">
                        O arquivo foi salvo em Downloads
                        {lastSavedFileName ? (
                          <>
                            :{" "}
                            <strong className="text-slate-200" title={lastSavedFileName}>
                              {lastSavedFileName}
                            </strong>
                          </>
                        ) : (
                          "."
                        )}
                      </p>
                      <p className="text-[11px] text-slate-500 mt-4">
                        Os itens concluídos foram removidos da fila. O cache de blocos e previews foi limpo.
                      </p>
                    </div>
                  ) : (
                    <>
                  {completedDocs.length > 1 && (
                    <div className="flex gap-2 overflow-x-auto pb-3 mb-4 scrollbar-thin scrollbar-thumb-white/10">
                      {completedDocs.map((doc, index) => {
                        const isActive = resultDoc?.id === doc.id;
                        return (
                          <button
                            key={doc.id}
                            type="button"
                            onClick={() => selectResultDoc(doc.id)}
                            className={`shrink-0 max-w-[200px] rounded-xl border px-3 py-2 text-left transition-all cursor-pointer ${
                              isActive
                                ? "border-emerald-500/50 bg-emerald-500/10 text-white"
                                : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20"
                            }`}
                            title={doc.file.name}
                          >
                            <span className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                              Áudio {index + 1}
                            </span>
                            <span className="block text-xs font-semibold truncate">{doc.file.name}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  <div className="text-center mb-6">
                    <div className="inline-flex bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-3 py-1.5 rounded-full text-xs font-bold gap-1.5 items-center mb-4">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      <span>
                        {completedDocs.length > 1
                          ? `${completedDocs.length} ÁUDIOS PRONTOS`
                          : `ÁUDIO ${((resultDoc?.audioFormat || outputFormat) === "m4b" ? "M4B" : "MP3")} PRONTO`}
                      </span>
                    </div>

                    <h2 className="text-lg font-bold text-white truncate px-4" title={resultDoc?.file.name}>
                      {resultDoc?.file.name}
                    </h2>
                    <p className="text-xs text-slate-400 mt-1">
                      {resultDoc?.fileType === "pdf" ? "Páginas: " : "Seções: "}
                      <strong className="text-slate-200">{resultDoc?.pagesNarrated}</strong>
                      {" • "}Voz: <strong className="text-blue-400">{selectedVoice}</strong>
                    </p>
                  </div>

                  {/* PREMIUM CUSTOM AUDIO PLAYER */}
                  <div className="bg-slate-950/40 border border-white/10 rounded-2xl p-5 mb-6">
                    {/* Progress Slider */}
                    <div className="space-y-1 mb-4">
                      <input
                        type="range"
                        min={0}
                        max={duration || 100}
                        value={currentTime}
                        onChange={handleSeek}
                        className="w-full h-1.5 bg-white/10 accent-blue-500 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-[11px] font-semibold text-slate-400">
                        <span>{formatTime(currentTime)}</span>
                        <span>{formatTime(duration)}</span>
                      </div>
                    </div>

                    {/* Controls Row */}
                    <div className="flex items-center justify-center gap-5 mb-5">
                      <button
                        onClick={() => skipTime(-10)}
                        className="text-slate-400 hover:text-white transition-colors p-2 rounded-lg hover:bg-white/5 cursor-pointer"
                        title="Voltar 10 segundos"
                      >
                        <RotateCcw className="w-5 h-5" />
                      </button>

                      <button
                        onClick={togglePlay}
                        className="bg-white text-slate-950 w-14 h-14 rounded-full flex items-center justify-center shadow-xl transition-all hover:scale-105 active:scale-95 cursor-pointer"
                        title={isPlaying ? "Pausar" : "Reproduzir"}
                      >
                        {isPlaying ? (
                          <Pause className="w-6 h-6 fill-current text-slate-950" />
                        ) : (
                          <Play className="w-6 h-6 fill-current text-slate-950 translate-x-0.5" />
                        )}
                      </button>

                      <button
                        onClick={() => skipTime(10)}
                        className="text-slate-400 hover:text-white transition-colors p-2 rounded-lg hover:bg-white/5 cursor-pointer"
                        title="Avançar 10 segundos"
                      >
                        <RotateCw className="w-5 h-5" />
                      </button>
                    </div>

                    {/* Volume and Mute row */}
                    <div className="flex items-center gap-3 border-t border-white/5 pt-3">
                      <button
                        onClick={toggleMute}
                        className="text-slate-400 hover:text-white transition-colors p-1.5 rounded-lg cursor-pointer"
                      >
                        {isMuted ? (
                          <VolumeX className="w-4 h-4" />
                        ) : (
                          <Volume2 className="w-4 h-4" />
                        )}
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={isMuted ? 0 : volume}
                        onChange={handleVolumeChange}
                        className="flex-1 h-1 bg-white/10 accent-white rounded-lg appearance-none cursor-pointer"
                      />
                    </div>
                  </div>

                  {/* Physical Download MP3 Action */}
                  <button
                    onClick={() => downloadMp3(resultDoc)}
                    disabled={!resultDoc?.audioUrl}
                    className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-40 disabled:cursor-not-allowed text-white py-4 rounded-2xl font-bold text-sm shadow-lg shadow-emerald-950/20 flex items-center justify-center gap-2.5 transition-all hover:scale-[1.01] active:scale-[0.99] mb-4 cursor-pointer"
                  >
                    <Download className="w-5 h-5" />
                    <span>
                      Baixar Áudio (.
                      {(resultDoc?.audioFormat || outputFormat) === "m4b" ? "m4b" : "mp3"})
                    </span>
                  </button>

                  <p className="text-[10px] text-center text-slate-400">
                    O áudio gerado pode ser reproduzido offline e é compatível com celulares e computadores.
                  </p>
                    </>
                  )}
                </div>
              </div>

              {/* Right Column: Extracted Text Follow-along Panel */}
              <div className="lg:col-span-7">
                <div className="bg-white/5 backdrop-blur-2xl border border-white/15 rounded-3xl p-6 shadow-xl h-[520px] flex flex-col">
                  <h3 className="text-base font-bold text-white mb-1 flex items-center gap-2 shrink-0">
                    <BookOpen className="w-4.5 h-4.5 text-blue-400" />
                    <span>{batchDone && !resultDoc ? "Resumo" : "Texto Narrado"}</span>
                  </h3>
                  <p className="text-xs text-slate-400 mb-4 shrink-0">
                    {batchDone && !resultDoc
                      ? "Os arquivos foram exportados. Você pode iniciar uma nova narração quando quiser."
                      : "Acompanhe a leitura visual do texto que foi narrado."}
                  </p>

                  <div className="flex-1 overflow-y-auto bg-slate-950/40 border border-white/5 rounded-2xl p-5 scrollbar-thin scrollbar-thumb-white/10">
                    {batchDone && !resultDoc ? (
                      <div className="text-slate-300 text-sm leading-relaxed space-y-3">
                        <p>
                          <strong className="text-white">{savedFilesCount}</strong>{" "}
                          {savedFilesCount === 1 ? "arquivo salvo" : "arquivos salvos"} em Downloads.
                        </p>
                        {lastSavedFileName && (
                          <p className="text-slate-400 break-all">
                            Último: {lastSavedFileName}
                          </p>
                        )}
                      </div>
                    ) : (
                      <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap select-text font-serif">
                        {resultDoc?.editableText || resultDoc?.extractedText}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        )}
      </main>
      </div>

    </div>
  );
}
