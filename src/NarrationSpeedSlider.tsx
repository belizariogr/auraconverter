import { Gauge } from "lucide-react";
import {
  NARRATION_SPEED_DEFAULT,
  NARRATION_SPEED_MAX,
  NARRATION_SPEED_MIN,
  NARRATION_SPEED_STEP,
  clampNarrationSpeed,
} from "../narrationSpeed.ts";

export {
  NARRATION_SPEED_DEFAULT,
  NARRATION_SPEED_MAX,
  NARRATION_SPEED_MIN,
  NARRATION_SPEED_STEP,
  clampNarrationSpeed,
};

function formatSpeedLabel(speed: number): string {
  if (Math.abs(speed - 1) < 0.001) {
    return "1,00× (normal)";
  }

  return `${speed.toFixed(2).replace(".", ",")}×`;
}

export default function NarrationSpeedSlider({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (speed: number) => void;
  disabled?: boolean;
}) {
  const speed = clampNarrationSpeed(value);
  const percent =
    ((speed - NARRATION_SPEED_MIN) /
      (NARRATION_SPEED_MAX - NARRATION_SPEED_MIN)) *
    100;

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/40 px-4 py-3 space-y-2.5">
      <div className="flex items-center justify-between gap-3">
        <label
          htmlFor="narration-speed"
          className="text-xs font-bold text-slate-300 flex items-center gap-1.5"
        >
          <Gauge className="w-3.5 h-3.5 text-blue-400" />
          Velocidade da narração
        </label>
        <span className="text-xs font-mono font-semibold text-blue-300 tabular-nums">
          {formatSpeedLabel(speed)}
        </span>
      </div>

      <input
        id="narration-speed"
        type="range"
        min={NARRATION_SPEED_MIN}
        max={NARRATION_SPEED_MAX}
        step={NARRATION_SPEED_STEP}
        value={speed}
        disabled={disabled}
        onChange={(e) => onChange(clampNarrationSpeed(Number(e.target.value)))}
        aria-valuemin={NARRATION_SPEED_MIN}
        aria-valuemax={NARRATION_SPEED_MAX}
        aria-valuenow={speed}
        aria-label="Velocidade da narração"
        className="w-full h-2 rounded-full appearance-none cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed accent-blue-500"
        style={{
          background: `linear-gradient(to right, rgb(59 130 246 / 0.85) ${percent}%, rgb(15 23 42 / 0.9) ${percent}%)`,
        }}
      />

      <div className="flex justify-between text-[10px] text-slate-500 font-medium">
        <span>Mais lento</span>
        <span>Mais rápido</span>
      </div>
    </div>
  );
}
