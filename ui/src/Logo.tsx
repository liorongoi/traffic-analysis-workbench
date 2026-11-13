// LanePulse mark: a road corridor crossed by a live traffic pulse, on a cut-corner tile like the app's panels.
export default function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} aria-hidden="true">
      <defs>
        <linearGradient id="lanePulseBg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#13223a" />
          <stop offset="1" stopColor="#08101b" />
        </linearGradient>
      </defs>
      <path d="M11 1h52v51L52 63H1V12z" fill="url(#lanePulseBg)" stroke="#3a5c86" strokeWidth="1.5" />
      <path d="M17 55C21 42 25 27 28 10M47 55C43 42 39 27 36 10" fill="none" stroke="#dce7f5" strokeOpacity=".72" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M32 53v-7M32 39v-6M32 27v-5" fill="none" stroke="#ffb648" strokeWidth="3.2" strokeLinecap="round" />
      <path d="M20 34h7l4-8 5 16 4-8h6" fill="none" stroke="#4cd7f6" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="47" cy="34" r="2.3" fill="#4cd7f6" />
    </svg>
  );
}
