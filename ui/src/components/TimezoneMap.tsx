import { useEffect, useState } from "react";

type Feature = { zone: string; path: string };
type MapData = { width: number; height: number; features: Feature[] };

export default function TimezoneMap({ selected, onPick }: {
  selected: string;
  onPick: (zone: string, latitude: number, longitude: number) => void;
}) {
  const [map, setMap] = useState<MapData | null>(null);
  useEffect(() => {
    let live = true;
    import("../data/timezones.json").then((module) => {
      if (live) setMap(module.default as MapData);
    });
    return () => { live = false; };
  }, []);

  if (!map) return <div className="timezone-map-loading muted">Loading offline timezone map…</div>;

  const pick = (event: React.MouseEvent<SVGPathElement>, zone: string) => {
    const svg = event.currentTarget.ownerSVGElement;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const mapPoint = point.matrixTransform(matrix.inverse());
    const latitude = Math.max(-90, Math.min(90, 90 - mapPoint.y * 180 / map.height));
    const longitude = Math.max(-180, Math.min(180, mapPoint.x * 360 / map.width - 180));
    onPick(zone, latitude, longitude);
  };

  return (
    <div className="timezone-map-wrap">
      <svg className="timezone-map" viewBox={`0 0 ${map.width} ${map.height}`} role="img" aria-label="Interactive world time-zone map">
        <rect className="timezone-ocean" width={map.width} height={map.height} />
        {map.features.map(({ zone, path }) => (
          <path key={zone} d={path} className={zone === selected ? "timezone-shape selected" : "timezone-shape"}
            role="button" tabIndex={0} aria-label={zone} aria-pressed={zone === selected}
            onClick={(event) => pick(event, zone)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                const box = event.currentTarget.getBBox();
                onPick(zone, 90 - (box.y + box.height / 2) * 180 / map.height,
                  (box.x + box.width / 2) * 360 / map.width - 180);
              }
            }}>
            <title>{zone}</title>
          </path>
        ))}
      </svg>
      <div className="muted timezone-map-caption">Click a timezone area to select it and save the approximate mission location. Boundaries are approximate; UTC conversion uses IANA rules.</div>
    </div>
  );
}
