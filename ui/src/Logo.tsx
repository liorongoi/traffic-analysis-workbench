// The Lanemark mark: two road lanes and a dashed centre line. Same drawing as docs/brand/lanemark-small.svg.
export default function Logo({ size = 24 }: { size?: number }) {
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} aria-hidden="true">
      <rect width="64" height="64" rx="14" fill="#1f5fbf" />
      <path d="M18 52L28 16M46 52L36 16" fill="none" stroke="#fff" strokeWidth="5" strokeLinecap="round" />
      <path d="M32 50v-7M32 37v-5M32 27v-3" fill="none" stroke="#ffb547" strokeWidth="4.5" strokeLinecap="round" />
    </svg>
  );
}
