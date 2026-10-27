// LanePulse mark: a road corridor crossed by a live traffic pulse.
export default function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} aria-hidden="true">
      <defs>
        <linearGradient id="lanePulseBg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#162943" />
          <stop offset="1" stopColor="#07111f" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="17" fill="url(#lanePulseBg)" />
      <rect x="1" y="1" width="62" height="62" rx="16" fill="none" stroke="#47647e" strokeOpacity=".45" />
      <path d="M17 55C21 42 25 27 28 10M47 55C43 42 39 27 36 10" fill="none" stroke="#d8e7f5" strokeOpacity=".72" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M32 53v-7M32 39v-6M32 27v-5" fill="none" stroke="#ffcc70" strokeWidth="3.2" strokeLinecap="round" />
      <path d="M20 34h7l4-8 5 16 4-8h6" fill="none" stroke="#59e0c2" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="47" cy="34" r="2.3" fill="#59e0c2" />
    </svg>
  );
}
