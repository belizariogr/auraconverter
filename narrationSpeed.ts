/** Shared narration playback speed (UI preview, ICL anchor, encode / Kokoro). */

export const NARRATION_SPEED_MIN = 0.75;
export const NARRATION_SPEED_MAX = 1.5;
export const NARRATION_SPEED_DEFAULT = 1;
export const NARRATION_SPEED_STEP = 0.05;

/** Clamp and snap to the slider step (avoids float drift). */
export function clampNarrationSpeed(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);

  if (!Number.isFinite(n)) {
    return NARRATION_SPEED_DEFAULT;
  }

  const clamped = Math.min(
    NARRATION_SPEED_MAX,
    Math.max(NARRATION_SPEED_MIN, n)
  );
  const steps = Math.round(
    (clamped - NARRATION_SPEED_MIN) / NARRATION_SPEED_STEP
  );

  return Number(
    (NARRATION_SPEED_MIN + steps * NARRATION_SPEED_STEP).toFixed(2)
  );
}

/** Filename/cache segment for a speed (e.g. 1 → `s100`, 0.75 → `s075`). */
export function narrationSpeedCacheTag(speed: unknown): string {
  const clamped = clampNarrationSpeed(speed);
  const cents = Math.round(clamped * 100)
    .toString()
    .padStart(3, "0");

  return `s${cents}`;
}
