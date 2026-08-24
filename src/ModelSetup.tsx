import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Download,
  HardDrive,
  Loader2,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react";

type ModelInfo = {
  id: string;
  folder: string;
  label: string;
  present: boolean;
  approxBytes: number;
};

type EngineId = "qwen3" | "kokoro";
type KokoroDeviceId = "cpu" | "gpu";
type KokoroBackendId = "mlx" | "onnx";

type GpuInfo = {
  supported: boolean;
  devices: Array<{ name: string; vendor: string }>;
  primary: "nvidia" | "amd" | "intel" | "unknown" | null;
  recommendedAccel: "cuda" | "rocm" | "cpu" | null;
  packagedAccel: "cuda" | "rocm" | "cpu" | "mlx" | null;
  summary: string;
};

type KokoroAccelInfo = {
  requested: KokoroDeviceId;
  effective: "gpu" | "cpu";
  gpuReady: boolean;
  availableProviders: string[];
  hint: string | null;
};

type EngineInfo = {
  ready: boolean;
  modelsReady?: boolean;
  runtimeReady?: boolean;
  modelsDir: string;
  label: string;
  description: string;
  models: ModelInfo[];
};

type ProgressState = {
  modelId?: string;
  modelLabel?: string;
  file?: string;
  fileIndex?: number;
  fileCount?: number;
  fileDownloadedBytes: number;
  fileTotalBytes: number;
  filePercent: number;
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  bytesPerSecond: number;
  etaLabel?: string | null;
  phase: string;
};

function formatBytes(n: number): string {
  if (!n || n < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatSpeed(bps: number): string {
  if (!bps || bps < 1) return "—";
  return `${formatBytes(bps)}/s`;
}

function vendorBadge(primary: GpuInfo["primary"]): {
  label: string;
  className: string;
} {
  if (primary === "nvidia") {
    return {
      label: "NVIDIA",
      className: "bg-emerald-500/15 text-emerald-300 border-emerald-400/25",
    };
  }
  if (primary === "amd") {
    return {
      label: "AMD",
      className: "bg-rose-500/15 text-rose-300 border-rose-400/25",
    };
  }
  if (primary === "intel") {
    return {
      label: "Intel",
      className: "bg-sky-500/15 text-sky-300 border-sky-400/25",
    };
  }
  return {
    label: "GPU não identificada",
    className: "bg-slate-500/15 text-slate-300 border-white/10",
  };
}

const emptyProgress = (): ProgressState => ({
  fileDownloadedBytes: 0,
  fileTotalBytes: 0,
  filePercent: 0,
  downloadedBytes: 0,
  totalBytes: 0,
  percent: 0,
  bytesPerSecond: 0,
  etaLabel: null,
  phase: "Aguardando",
});

export default function ModelSetup({
  models,
  modelsDir,
  gpu: gpuProp,
  backend,
  engine: engineProp = "qwen3",
  engines: enginesProp,
  kokoroDevice: kokoroDeviceProp = "gpu",
  kokoroBackend: kokoroBackendProp = "mlx",
  kokoroAccel: kokoroAccelProp,
  platform,
  runtimeReady: runtimeReadyProp = true,
  onComplete,
  onStatusChange,
}: {
  models: ModelInfo[];
  modelsDir: string;
  gpu?: GpuInfo | null;
  backend?: "torch" | "mlx" | "onnx";
  engine?: EngineId;
  engines?: { qwen3: EngineInfo; kokoro: EngineInfo };
  kokoroDevice?: KokoroDeviceId;
  kokoroBackend?: KokoroBackendId;
  kokoroAccel?: KokoroAccelInfo | null;
  platform?: string;
  runtimeReady?: boolean;
  onComplete: () => void;
  onStatusChange?: (
    models: ModelInfo[],
    modelsDir: string,
    extra?: {
      gpu?: GpuInfo | null;
      backend?: string;
      engine?: EngineId;
      engines?: { qwen3: EngineInfo; kokoro: EngineInfo };
      kokoroDevice?: KokoroDeviceId;
      kokoroBackend?: KokoroBackendId;
      kokoroAccel?: KokoroAccelInfo | null;
      ready?: boolean;
      modelsReady?: boolean;
      runtimeReady?: boolean;
    }
  ) => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localModels, setLocalModels] = useState(models);
  const [localDir, setLocalDir] = useState(modelsDir);
  const [runtimeReady, setRuntimeReady] = useState(runtimeReadyProp);
  const [gpu, setGpu] = useState<GpuInfo | null | undefined>(gpuProp);
  const [engine, setEngine] = useState<EngineId>(engineProp);
  const [engines, setEngines] = useState(enginesProp);
  const [kokoroDevice, setKokoroDevice] = useState<KokoroDeviceId>(kokoroDeviceProp);
  const [kokoroBackend, setKokoroBackend] =
    useState<KokoroBackendId>(kokoroBackendProp);
  const [kokoroAccel, setKokoroAccel] = useState<KokoroAccelInfo | null | undefined>(
    kokoroAccelProp
  );
  const [switching, setSwitching] = useState(false);
  const [warming, setWarming] = useState(false);
  const [warmup, setWarmup] = useState<{
    running?: boolean;
    done?: boolean;
    current?: number;
    total?: number;
    phase?: string;
    error?: string | null;
    elapsedMs?: number;
  } | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState>(emptyProgress);
  const [overall, setOverall] = useState({ current: 0, total: models.length });

  useEffect(() => {
    setGpu(gpuProp);
  }, [gpuProp]);

  useEffect(() => {
    setEngine(engineProp);
  }, [engineProp]);

  useEffect(() => {
    setEngines(enginesProp);
  }, [enginesProp]);

  useEffect(() => {
    setKokoroDevice(kokoroDeviceProp);
  }, [kokoroDeviceProp]);

  useEffect(() => {
    setKokoroBackend(kokoroBackendProp);
  }, [kokoroBackendProp]);

  useEffect(() => {
    setKokoroAccel(kokoroAccelProp);
  }, [kokoroAccelProp]);

  useEffect(() => {
    setRuntimeReady(runtimeReadyProp);
  }, [runtimeReadyProp]);

  useEffect(() => {
    setLocalModels(models);
    setLocalDir(modelsDir);
  }, [models, modelsDir]);

  const missing = useMemo(() => localModels.filter((m) => !m.present), [localModels]);
  const present = useMemo(() => localModels.filter((m) => m.present), [localModels]);
  const approxTotal = useMemo(
    () => missing.reduce((sum, m) => sum + (m.approxBytes || 0), 0),
    [missing]
  );

  const showGpu = Boolean(gpu?.supported) && engine === "qwen3";
  const isTorch = backend === "torch" || (showGpu && engine === "qwen3");
  const kokoroIsMlx = engine === "kokoro" && kokoroBackend === "mlx";
  const badge = vendorBadge(gpu?.primary ?? null);
  const accelMismatch =
    showGpu &&
    gpu?.packagedAccel &&
    gpu?.recommendedAccel &&
    gpu.packagedAccel !== "mlx" &&
    gpu.packagedAccel !== gpu.recommendedAccel;

  async function refreshStatus() {
    const res = await fetch("/api/models/status");
    const body = await res.json();
    if (body.models) {
      setLocalModels(body.models);
      onStatusChange?.(body.models, body.modelsDir || localDir, {
        gpu: body.gpu ?? null,
        backend: body.backend,
        engine: body.engine,
        engines: body.engines,
        kokoroDevice: body.kokoroDevice,
        kokoroBackend: body.kokoroBackend,
        kokoroAccel: body.kokoroAccel ?? null,
        ready: body.ready,
        modelsReady: body.modelsReady,
        runtimeReady: body.runtimeReady,
      });
    }
    if (body.modelsDir) setLocalDir(body.modelsDir);
    if (body.gpu) setGpu(body.gpu);
    if (body.engine) setEngine(body.engine);
    if (body.engines) setEngines(body.engines);
    if (typeof body.runtimeReady === "boolean") setRuntimeReady(body.runtimeReady);
    if (body.kokoroDevice === "cpu" || body.kokoroDevice === "gpu") {
      setKokoroDevice(body.kokoroDevice);
    }
    if (body.kokoroBackend === "mlx" || body.kokoroBackend === "onnx") {
      setKokoroBackend(body.kokoroBackend);
    }
    if (body.kokoroAccel) setKokoroAccel(body.kokoroAccel);
    return body;
  }

  const modelsReady = localModels.every((m) => m.present);
  const allReady = modelsReady && runtimeReady;

  async function selectEngine(next: EngineId) {
    if (next === engine || downloading || switching) return;
    setSwitching(true);
    setError(null);
    try {
      const res = await fetch("/api/tts-engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: next }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      await refreshStatus();
      setProgress(emptyProgress());
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setSwitching(false);
    }
  }

  async function selectKokoroDevice(next: KokoroDeviceId) {
    if (next === kokoroDevice || downloading || switching) return;
    setSwitching(true);
    setError(null);
    try {
      const res = await fetch("/api/tts-engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kokoroDevice: next }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      setKokoroDevice(next);
      if (body.kokoroAccel) setKokoroAccel(body.kokoroAccel);
      onStatusChange?.(localModels, localDir, {
        gpu: gpu ?? null,
        backend: kokoroBackend,
        engine,
        engines,
        kokoroDevice: next,
        kokoroBackend,
        kokoroAccel: body.kokoroAccel ?? null,
      });
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setSwitching(false);
    }
  }

  async function selectKokoroBackend(next: KokoroBackendId) {
    if (next === kokoroBackend || downloading || switching) return;
    setSwitching(true);
    setError(null);
    try {
      const res = await fetch("/api/tts-engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kokoroBackend: next }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      setKokoroBackend(next);
      await refreshStatus();
      setProgress(emptyProgress());
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setSwitching(false);
    }
  }

  async function startKokoroWarmup() {
    if (warming || downloading || switching) return;
    if (engine !== "kokoro" || kokoroDevice !== "gpu") return;
    setWarming(true);
    setError(null);
    setWarmup({ running: true, done: false, current: 0, total: 6, phase: "Iniciando…" });
    try {
      const res = await fetch("/api/tts/kokoro-warmup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      setWarmup(body);

      // Poll until compile finishes (can take many minutes).
      const started = Date.now();
      while (Date.now() - started < 45 * 60_000) {
        await new Promise((r) => setTimeout(r, 2000));
        const st = await fetch("/api/tts/kokoro-warmup");
        const status = await st.json();
        if (!st.ok) throw new Error(status.error || `HTTP ${st.status}`);
        setWarmup(status);
        if (status.error) throw new Error(status.error);
        if (status.done) break;
        if (!status.running && status.current > 0) break;
      }
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setWarming(false);
    }
  }

  async function cancelDownload() {
    try {
      await fetch("/api/models/download/cancel", { method: "POST" });
      setProgress((p) => ({ ...p, phase: "Cancelando…" }));
    } catch {
      // ignore
    }
  }

  async function removeModels(ids?: string[]) {
    const label = ids?.length
      ? `Excluir ${ids.join(", ")}?`
      : "Excluir todos os modelos baixados?";
    if (!window.confirm(`${label}\n\nSerá necessário baixar de novo para narrar.`)) return;

    setDeleting(ids?.join(",") || "all");
    setError(null);
    try {
      const q = ids?.length ? `?id=${encodeURIComponent(ids.join(","))}` : "";
      const res = await fetch(`/api/models${q}`, { method: "DELETE" });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      if (body.status?.models) {
        setLocalModels(body.status.models);
        onStatusChange?.(body.status.models, body.status.modelsDir || localDir, {
          gpu: body.status.gpu ?? gpu ?? null,
          backend: body.status.backend,
          engine: body.status.engine,
          engines: body.status.engines,
          kokoroDevice: body.status.kokoroDevice,
        });
      }
      if (body.status?.modelsDir) setLocalDir(body.status.modelsDir);
      if (body.status?.gpu) setGpu(body.status.gpu);
      if (body.status?.engine) setEngine(body.status.engine);
      if (body.status?.engines) setEngines(body.status.engines);
      if (body.status?.kokoroDevice === "cpu" || body.status?.kokoroDevice === "gpu") {
        setKokoroDevice(body.status.kokoroDevice);
      }
      setProgress(emptyProgress());
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setDeleting(null);
    }
  }

  async function startDownload() {
    setDownloading(true);
    setError(null);
    setProgress({
      ...emptyProgress(),
      phase:
        engine === "kokoro"
          ? "Preparando Kokoro e baixando modelos…"
          : "Preparando runtime e baixando modelos…",
    });
    setOverall({ current: 0, total: Math.max(1, missing.length || localModels.length) });

    try {
      const res = await fetch("/api/models/download", { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Falha ao iniciar download (HTTP ${res.status})`);
      }
      if (!res.body) throw new Error("Resposta sem stream de progresso.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completedModels = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          let evt: any;
          try {
            evt = JSON.parse(line);
          } catch {
            continue;
          }

          if (
            evt.type === "runtime_start" ||
            evt.type === "runtime_log" ||
            evt.type === "runtime_skip" ||
            evt.type === "runtime_done"
          ) {
            if (evt.type === "runtime_done" || evt.type === "runtime_skip") {
              setRuntimeReady(true);
            }
            setProgress((prev) => ({
              ...prev,
              phase: String(evt.phase || evt.line || "Preparando runtime…"),
              percent: evt.type === "runtime_done" || evt.type === "runtime_skip" ? prev.percent : 0,
            }));
            continue;
          }

          if (evt.type === "model_start") {
            setOverall((o) => ({ ...o, current: completedModels + 1 }));
            setProgress({
              ...emptyProgress(),
              modelId: evt.model,
              modelLabel: evt.label || evt.model,
              totalBytes: evt.totalBytes || 0,
              phase: `Baixando ${evt.label || evt.model}`,
            });
          } else if (evt.type === "model_skip") {
            completedModels += 1;
            setLocalModels((prev) =>
              prev.map((m) => (m.id === evt.model ? { ...m, present: true } : m))
            );
            setOverall((o) => ({ ...o, current: completedModels }));
          } else if (evt.type === "file_start") {
            setProgress((prev) => ({
              ...prev,
              modelId: evt.model,
              modelLabel: evt.label || prev.modelLabel,
              file: evt.file,
              fileIndex: evt.fileIndex,
              fileCount: evt.fileCount,
              fileDownloadedBytes: 0,
              fileTotalBytes: evt.fileBytes || 0,
              filePercent: 0,
              downloadedBytes: evt.downloadedBytes || prev.downloadedBytes,
              totalBytes: evt.totalBytes || prev.totalBytes,
              percent: typeof evt.percent === "number" ? evt.percent : prev.percent,
              phase: "Baixando arquivo",
            }));
          } else if (evt.type === "progress") {
            setProgress((prev) => ({
              ...prev,
              modelId: evt.model,
              modelLabel: evt.label || prev.modelLabel,
              file: evt.file,
              fileIndex: evt.fileIndex,
              fileCount: evt.fileCount,
              fileDownloadedBytes: evt.fileDownloadedBytes || 0,
              fileTotalBytes: evt.fileTotalBytes || 0,
              filePercent: typeof evt.filePercent === "number" ? evt.filePercent : 0,
              downloadedBytes: evt.downloadedBytes || 0,
              totalBytes: evt.totalBytes || 0,
              percent: typeof evt.percent === "number" ? evt.percent : 0,
              bytesPerSecond: evt.bytesPerSecond || 0,
              etaLabel: evt.etaLabel ?? null,
              phase: "Baixando",
            }));
          } else if (evt.type === "file_done") {
            setProgress((prev) => ({
              ...prev,
              file: evt.file,
              fileIndex: evt.fileIndex,
              fileCount: evt.fileCount,
              filePercent: 100,
              downloadedBytes: evt.downloadedBytes || prev.downloadedBytes,
              totalBytes: evt.totalBytes || prev.totalBytes,
              percent: typeof evt.percent === "number" ? evt.percent : prev.percent,
              bytesPerSecond: evt.bytesPerSecond || prev.bytesPerSecond,
              etaLabel: evt.etaLabel ?? prev.etaLabel,
              phase: "Arquivo concluído",
            }));
          } else if (evt.type === "model_done") {
            completedModels += 1;
            setLocalModels((prev) =>
              prev.map((m) => (m.id === evt.model ? { ...m, present: true } : m))
            );
            setOverall((o) => ({ ...o, current: completedModels }));
            setProgress((p) => ({
              ...p,
              percent: 100,
              filePercent: 100,
              phase: `${evt.label || evt.model} pronto`,
            }));
          } else if (evt.type === "done") {
            await refreshStatus();
            setDownloading(false);
            setProgress((p) => ({ ...p, phase: "Download concluído", percent: 100 }));
            return;
          } else if (evt.type === "cancelled") {
            setError(evt.message || "Download cancelado.");
            setDownloading(false);
            setProgress((p) => ({ ...p, phase: "Cancelado" }));
            await refreshStatus().catch(() => undefined);
            return;
          } else if (evt.type === "error") {
            throw new Error(evt.message || "Erro no download");
          }
        }
      }

      const finalStatus = await refreshStatus();
      setDownloading(false);
      if (!finalStatus.ready) {
        throw new Error(
          !finalStatus.runtimeReady
            ? "A preparação do runtime TTS ainda está incompleta."
            : "Download terminou, mas os modelos ainda estão incompletos."
        );
      }
    } catch (err: any) {
      setError(err?.message || String(err));
      setDownloading(false);
      await refreshStatus().catch(() => undefined);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-blue-500/30 relative flex flex-col">
      <div className="pointer-events-none absolute inset-0 overflow-clip" aria-hidden>
        <div className="absolute top-[-100px] left-[-100px] w-[500px] h-[500px] bg-blue-600/15 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-100px] right-[-100px] w-[600px] h-[600px] bg-indigo-600/15 rounded-full blur-[150px]" />
      </div>

      <header className="relative z-10 backdrop-blur-md bg-slate-950/40 border-b border-white/10 px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <div className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white p-2.5 rounded-xl">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">Aura Converter</h1>
            <p className="text-xs text-slate-400">Instalação dos modelos de voz</p>
          </div>
        </div>
      </header>

      <main className="relative z-10 flex-1 flex items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-xl bg-white/5 border border-white/10 backdrop-blur-2xl rounded-3xl p-8 shadow-2xl"
        >
          <div className="flex items-start gap-3 mb-6">
            <div className="p-2 rounded-xl bg-amber-500/15 text-amber-300 border border-amber-400/20">
              <HardDrive className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white mb-1">Motor de voz</h2>
              <p className="text-sm text-slate-300 leading-relaxed">
                Escolha Qwen3 (qualidade / clonagem) ou Kokoro (rápido e leve). A instalação
                prepara o runtime Python e baixa só os arquivos necessários (~
                {formatBytes(
                  approxTotal ||
                    (engine === "kokoro" ? 350_000_000 : 4_000_000_000)
                )}
                ).
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            {(
              [
                {
                  id: "qwen3" as const,
                  title: engines?.qwen3?.label || "Qwen3-TTS",
                  desc:
                    engines?.qwen3?.description ||
                    "Alta qualidade com clonagem de voz (ICL).",
                  ready: engines?.qwen3?.ready,
                },
                {
                  id: "kokoro" as const,
                  title: engines?.kokoro?.label || "Kokoro",
                  desc:
                    engines?.kokoro?.description ||
                    (kokoroBackend === "mlx"
                      ? "Rápido no Apple Silicon (MLX bf16)."
                      : "Qualidade máxima (ONNX fp32) com Core ML/CPU."),
                  ready: engines?.kokoro?.ready,
                },
              ] as const
            ).map((opt) => {
              const selected = engine === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  disabled={downloading || switching}
                  onClick={() => selectEngine(opt.id)}
                  className={`text-left rounded-2xl border px-4 py-3 transition-colors disabled:opacity-50 ${
                    selected
                      ? "border-blue-400/50 bg-blue-500/15"
                      : "border-white/10 bg-slate-900/40 hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-sm font-semibold text-white">{opt.title}</span>
                    {opt.ready ? (
                      <span className="text-[10px] text-emerald-300">instalado</span>
                    ) : (
                      <span className="text-[10px] text-amber-200/80">não instalado</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{opt.desc}</p>
                  {selected && (
                    <p className="text-[11px] text-blue-300 mt-2">Selecionado</p>
                  )}
                </button>
              );
            })}
          </div>

          {engine === "kokoro" && (
            <div className="mb-6 rounded-2xl border border-white/10 bg-slate-900/50 p-4 space-y-3">
              {platform === "darwin" && (
                <>
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-slate-400" />
                    <span className="text-xs text-slate-300 font-medium">
                      Versão do Kokoro
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {(
                      [
                        {
                          id: "mlx" as const,
                          title: "MLX",
                          desc: "bf16 otimizado para Apple Silicon",
                        },
                        {
                          id: "onnx" as const,
                          title: "Não MLX",
                          desc: "ONNX fp32 — modelo de maior qualidade",
                        },
                      ] as const
                    ).map((opt) => {
                      const selected = kokoroBackend === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          disabled={downloading || switching}
                          onClick={() => selectKokoroBackend(opt.id)}
                          className={`text-left rounded-xl border px-3 py-2.5 transition-colors disabled:opacity-50 ${
                            selected
                              ? "border-blue-400/45 bg-blue-500/10"
                              : "border-white/10 bg-slate-950/40 hover:bg-white/5"
                          }`}
                        >
                          <span className="text-sm font-semibold text-white">{opt.title}</span>
                          <p className="text-[11px] text-slate-400 leading-snug mt-0.5">
                            {opt.desc}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                  <div className="border-t border-white/10" />
                </>
              )}
              <div className="flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-xs text-slate-300 font-medium">Aceleração Kokoro</span>
              </div>
              {kokoroIsMlx ? (
                <>
                  <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 px-3 py-2.5">
                    <span className="text-sm font-semibold text-white">Apple GPU (MLX)</span>
                    <p className="text-[11px] text-slate-400 leading-snug mt-0.5">
                      Kokoro 82M bf16 via Metal — mesma qualidade do modelo completo, sem ONNX no app.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {(
                      [
                        {
                          id: "gpu" as const,
                          title: "GPU",
                          desc: "Metal / MLX (recomendado)",
                        },
                        {
                          id: "cpu" as const,
                          title: "CPU",
                          desc: "Mais lento; útil só para diagnóstico",
                        },
                      ] as const
                    ).map((opt) => {
                      const selected = kokoroDevice === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          disabled={downloading || switching}
                          onClick={() => selectKokoroDevice(opt.id)}
                          className={`text-left rounded-xl border px-3 py-2.5 transition-colors disabled:opacity-50 ${
                            selected
                              ? "border-emerald-400/45 bg-emerald-500/10"
                              : "border-white/10 bg-slate-950/40 hover:bg-white/5"
                          }`}
                        >
                          <span className="text-sm font-semibold text-white">{opt.title}</span>
                          <p className="text-[11px] text-slate-400 leading-snug mt-0.5">{opt.desc}</p>
                        </button>
                      );
                    })}
                  </div>
                  {kokoroDevice === "gpu" && kokoroAccel && !kokoroAccel.gpuReady && (
                    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 leading-relaxed">
                      <p className="font-semibold text-amber-50 mb-1">MLX ainda não pronto</p>
                      <p>{kokoroAccel.hint || "Instale as dependências MLX (misaki + mlx-audio)."}</p>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    {(
                      [
                        {
                          id: "gpu" as const,
                          title: "GPU",
                          desc:
                            platform === "darwin"
                              ? "Core ML quando disponível"
                              : "MIGraphX / CUDA / DirectML quando disponível",
                        },
                        {
                          id: "cpu" as const,
                          title: "CPU",
                          desc: "Mais estável; sem dependência de ROCm/CUDA",
                        },
                      ] as const
                    ).map((opt) => {
                      const selected = kokoroDevice === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          disabled={downloading || switching}
                          onClick={() => selectKokoroDevice(opt.id)}
                          className={`text-left rounded-xl border px-3 py-2.5 transition-colors disabled:opacity-50 ${
                            selected
                              ? "border-emerald-400/45 bg-emerald-500/10"
                              : "border-white/10 bg-slate-950/40 hover:bg-white/5"
                          }`}
                        >
                          <span className="text-sm font-semibold text-white">{opt.title}</span>
                          <p className="text-[11px] text-slate-400 leading-snug mt-0.5">{opt.desc}</p>
                        </button>
                      );
                    })}
                  </div>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    {platform === "darwin" ? (
                      <>A versão Não MLX usa o modelo Kokoro v1.0 fp32 completo e cai para CPU se o Core ML não estiver disponível.</>
                    ) : (
                      <>Em AMD, GPU exige o pacote de sistema <span className="font-mono">migraphx</span>. Sem ele, o Kokoro cai automaticamente para CPU.</>
                    )}
                  </p>
                  {kokoroDevice === "gpu" && kokoroAccel && !kokoroAccel.gpuReady && (
                    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 leading-relaxed">
                      <p className="font-semibold text-amber-50 mb-1">GPU marcada, mas ainda no CPU</p>
                      <p>{kokoroAccel.hint || "Dependências de GPU não estão prontas."}</p>
                    </div>
                  )}
                  {platform !== "darwin" &&
                    kokoroDevice === "gpu" &&
                    kokoroAccel?.gpuReady && (
                    <div className="space-y-2">
                      <p className="text-[11px] text-emerald-300/90 leading-relaxed">
                        Runtime GPU pronto (
                        {kokoroAccel.availableProviders
                          .filter((p) => p !== "CPUExecutionProvider")
                          .join(", ") || "GPU"}
                        ).
                      </p>
                      <button
                        type="button"
                        disabled={warming || downloading || switching}
                        onClick={() => startKokoroWarmup()}
                        className="w-full inline-flex items-center justify-center gap-2 rounded-xl border border-emerald-400/30 bg-emerald-500/10 px-3 py-2 text-xs font-medium text-emerald-100 hover:bg-emerald-500/20 disabled:opacity-50 transition-colors"
                      >
                        {warming ? (
                          <>
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            Aquecendo GPU…
                          </>
                        ) : (
                          "Aquecer GPU (pré-compilar)"
                        )}
                      </button>
                      <p className="text-[11px] text-slate-500 leading-relaxed">
                        Compila vários tamanhos de texto com todas as cores da CPU (
                        <span className="font-mono">MIGRAPHX_GPU_COMPILE_PARALLEL</span>
                        ). Pode levar vários minutos na 1ª vez; depois a narração fica bem mais rápida.
                      </p>
                      {(warming || warmup) && (
                        <div className="rounded-xl border border-white/10 bg-slate-950/50 px-3 py-2 text-[11px] text-slate-300 space-y-1">
                          <p>{warmup?.phase || "…"}</p>
                          {typeof warmup?.current === "number" && typeof warmup?.total === "number" && warmup.total > 0 && (
                            <p className="font-mono text-slate-400">
                              {warmup.current}/{warmup.total}
                              {warmup.elapsedMs
                                ? ` · ${(warmup.elapsedMs / 1000).toFixed(0)}s`
                                : ""}
                              {warmup.done ? " · pronto" : ""}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                    )}
                </>
              )}
            </div>
          )}

          {showGpu && (            <div className="mb-6 rounded-2xl border border-white/10 bg-slate-900/50 p-4 space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="inline-flex items-center gap-1.5 text-xs text-slate-400">
                  <Cpu className="w-3.5 h-3.5" />
                  Placa de vídeo
                </span>
                <span
                  className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${badge.className}`}
                >
                  {badge.label}
                </span>
                {gpu?.recommendedAccel && (
                  <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-[11px] text-slate-300 font-mono">
                    sugerido: {gpu.recommendedAccel.toUpperCase()}
                  </span>
                )}
                {gpu?.packagedAccel && gpu.packagedAccel !== "mlx" && (
                  <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-[11px] text-slate-400 font-mono">
                    este app: {gpu.packagedAccel.toUpperCase()}
                  </span>
                )}
              </div>
              {gpu?.devices?.length ? (
                <ul className="space-y-1">
                  {gpu.devices.map((d) => (
                    <li key={d.name} className="text-xs text-slate-300 font-mono truncate">
                      {d.name}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-400">Nenhuma GPU detectada.</p>
              )}
              {gpu?.summary && (
                <p className="text-xs text-slate-400 leading-relaxed">{gpu.summary}</p>
              )}
              {accelMismatch && (
                <p className="text-xs text-amber-200/90 leading-relaxed">
                  A GPU sugere o build {gpu!.recommendedAccel!.toUpperCase()}, mas este pacote é{" "}
                  {gpu!.packagedAccel!.toUpperCase()}. A narração ainda funciona (com fallback para
                  CPU se a aceleração não bater), mas o desempenho pode ficar abaixo do ideal.
                </p>
              )}
            </div>
          )}

          <ul className="space-y-3 mb-6">
            {localModels.map((m) => (
              <li
                key={m.id}
                className="flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-slate-900/40 px-4 py-3"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium text-white">{m.label}</p>
                  <p className="text-xs text-slate-400 font-mono mt-0.5 truncate">{m.folder}</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {m.present ? (
                    <>
                      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-300">
                        <CheckCircle2 className="w-4 h-4" /> Pronto
                      </span>
                      <button
                        type="button"
                        disabled={downloading || deleting !== null}
                        onClick={() => removeModels([m.id])}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-300 hover:bg-rose-500/10 disabled:opacity-40"
                        title="Excluir este modelo"
                      >
                        {deleting === m.id ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Trash2 className="w-4 h-4" />
                        )}
                      </button>
                    </>
                  ) : (
                    <span className="text-xs text-amber-200/90">Pendente</span>
                  )}
                </div>
              </li>
            ))}
          </ul>

          <div className="mb-5 rounded-2xl border border-white/10 bg-slate-900/40 px-4 py-3 space-y-1.5">
            <p className="text-xs font-medium text-slate-300">
              Arquivos salvos em
            </p>
            <p className="text-[11px] text-slate-400 font-mono break-all leading-relaxed select-all">
              {localDir}
            </p>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              Pasta dos pesos do motor selecionado. Para liberar espaço, use Excluir
              acima ou apague esta pasta manualmente no Finder / Explorador de arquivos.
            </p>
          </div>

          {downloading && (
            <div className="mb-6 rounded-2xl border border-white/10 bg-slate-900/50 p-4 space-y-4">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span>
                    Modelo {Math.min(overall.current, overall.total)} de {overall.total}
                    {progress.modelLabel ? ` · ${progress.modelLabel}` : ""}
                  </span>
                  <span className="font-mono text-blue-300">{progress.percent.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 border border-white/10 rounded-full h-2 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-indigo-500"
                    initial={false}
                    animate={{ width: `${Math.min(100, Math.max(0, progress.percent))}%` }}
                    transition={{ duration: 0.15 }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span className="truncate mr-2">
                    Arquivo
                    {progress.fileIndex && progress.fileCount
                      ? ` ${progress.fileIndex}/${progress.fileCount}`
                      : ""}
                    {progress.file ? ` · ${progress.file}` : ""}
                  </span>
                  <span className="font-mono text-indigo-300 shrink-0">
                    {progress.filePercent.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-950 border border-white/10 rounded-full h-1.5 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-cyan-400 to-blue-400"
                    initial={false}
                    animate={{ width: `${Math.min(100, Math.max(0, progress.filePercent))}%` }}
                    transition={{ duration: 0.12 }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400 font-mono">
                <p>
                  Modelo: {formatBytes(progress.downloadedBytes)} / {formatBytes(progress.totalBytes)}
                </p>
                <p className="text-right">
                  Arquivo: {formatBytes(progress.fileDownloadedBytes)} /{" "}
                  {formatBytes(progress.fileTotalBytes)}
                </p>
                <p>Velocidade: {formatSpeed(progress.bytesPerSecond)}</p>
                <p className="text-right">Restante: {progress.etaLabel || "—"}</p>
              </div>
            </div>
          )}

          {error && (
            <div className="mb-5 p-3 rounded-xl border border-rose-500/30 bg-rose-950/40 text-rose-100 text-sm flex gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-300" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex flex-col gap-3">
            {downloading ? (
              <button
                type="button"
                onClick={cancelDownload}
                className="w-full inline-flex items-center justify-center gap-2.5 rounded-2xl border border-rose-400/30 bg-rose-500/10 hover:bg-rose-500/20 text-rose-100 font-semibold px-5 py-3.5 transition-colors"
              >
                <Square className="w-4 h-4 fill-current shrink-0" />
                Cancelar instalação
              </button>
            ) : (
              <div className="flex flex-col sm:flex-row gap-3">
                {allReady ? (
                  <button
                    type="button"
                    onClick={onComplete}
                    className="w-full inline-flex items-center justify-center gap-2.5 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold px-5 py-3.5 transition-colors"
                  >
                    <CheckCircle2 className="w-5 h-5 shrink-0" strokeWidth={2.25} />
                    Continuar para o app
                  </button>
                ) : (
                  <button
                    type="button"
                    disabled={deleting !== null}
                    onClick={startDownload}
                    className="w-full inline-flex items-center justify-center gap-2.5 rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold px-5 py-3.5 transition-colors"
                  >
                    <Download className="w-5 h-5 shrink-0" strokeWidth={2.25} />
                    {modelsReady && !runtimeReady
                      ? "Preparar runtime agora"
                      : "Baixar e preparar agora"}
                  </button>
                )}

                {present.length > 0 && (
                  <button
                    type="button"
                    disabled={deleting !== null}
                    onClick={() => removeModels()}
                    className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 text-slate-200 font-medium px-5 py-3.5 transition-colors disabled:opacity-50"
                  >
                    {deleting === "all" ? (
                      <Loader2 className="w-5 h-5 shrink-0 animate-spin" strokeWidth={2.25} />
                    ) : (
                      <Trash2 className="w-5 h-5 shrink-0" strokeWidth={2.25} />
                    )}
                    Excluir todos
                  </button>
                )}
              </div>
            )}
          </div>

          <p className="text-center text-[11px] text-slate-500 mt-4">
            Na instalação o app prepara o runtime Python e baixa os pesos (precisa de internet).
            {engine === "kokoro"
              ? kokoroIsMlx
                ? " Kokoro usa MLX no Apple Silicon."
                : " Kokoro usa o modelo ONNX fp32 completo."
              : isTorch
                ? " Windows/Linux: o setup escolhe CUDA (NVIDIA), ROCm (AMD) ou CPU automaticamente."
                : " Apple Silicon recomendado."}
            {engines?.qwen3?.ready && engines?.kokoro?.ready
              ? " Ambos os motores estão instalados — troque acima a qualquer momento."
              : ""}
          </p>
        </motion.div>
      </main>
    </div>
  );
}
