/** How sure the detector must be for a result to be exported. 0 means everything is exported. */
export function ScoreSlider({ value, onChange, what }: { value: number; onChange: (v: number) => void; what: string }) {
  return (
    <label className="row muted small" title={`Leave out the ${what} the detector was less sure of. What you added, corrected or confirmed is always kept.`}>
      Export {what} with confidence
      <input type="range" min={0} max={0.95} step={0.05} value={value} aria-label="Minimum confidence"
        onChange={(e) => onChange(Number(e.target.value))} style={{ width: 110 }} />
      <b style={{ minWidth: 70 }}>{value > 0 ? `${Math.round(value * 100)}% or more` : "any"}</b>
    </label>
  );
}
