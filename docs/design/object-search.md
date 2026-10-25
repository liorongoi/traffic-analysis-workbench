# Object search (still images): rules

Requirement IMG-03: a typed request becomes a structured search spec, shown to the operator before anything runs. No generated code runs. Uncertain attributes stay uncertain.

## Where search runs

- Only in missions created in the **object search** mode (Q09 default: counting, plates and search are separate missions). An object search mission processes its images like a general traffic mission (vehicle detection, night analysis, review), and adds the Search tab.

## Interpreting the request

- `trafficwb.interpret` is a deterministic parser over a fixed English vocabulary (Q11 default: English only). The design package listed it as the baseline next to a local language model. It was chosen for this version because its output is predictable, it can be tested against a written request set, and it needs no model pack or memory budget. A local language model can replace it later behind the same spec and validation.
- Vocabulary:
  - vehicle classes: car, bus, truck, motorcycle, bicycle, any vehicle;
  - subtypes: pickup, van, SUV, taxi, minibus, semi-trailer;
  - colours: red, orange, yellow, green, blue, white, grey, black, brown (silver and gray are read as grey);
  - exclusions: "not", "except", "non-", "no", "without";
  - time windows: "between 17:00 and 05:00", "after 22:00", "before 6", "7am to 9:30am", "22:00-02:00";
  - weekdays and day ranges: "on sunday", "monday to friday";
  - dates: ISO, or day.month.year;
  - image condition: "at night", "in daylight", "at dusk";
  - minimum count: "at least 3";
  - uncertainty: "only certain ones";
  - region names that exist in the mission.
- Words outside the vocabulary are listed as **not understood**. The search cannot run until the operator edits them away or ticks "ignore these words".
- "Weekend", "weekdays" and "working days" are not guessed, because they differ between Israel, Iran, Lebanon and Latvia. The operator names the days.
- The spec is plain JSON. `validate_spec` checks it before it is stored and again before it runs. It refuses unknown keys, wrong types and values outside the vocabulary.
- `search_request` stores:
  - the machine interpretation;
  - the spec the operator confirmed, which may be edited;
  - the differences between the two;
  - the ignored words;
  - who confirmed it, and when.

## Spec

```
targets:   [{category, subtype, colours (any of), exclude_colours}]
exclude:   {categories, subtypes}
regions:   {include: [names], exclude: [names]}
time:      {windows: [{days, start, end}], date_from, date_to, conditions: [day|dusk|night]}
min_count: images need at least this many results
uncertain: list_separately | exclude
not_understood, notes
```

Time rules:
- A window that ends at or before its start runs past midnight.
- `24:00` means midnight, and `00:00-24:00` is the whole day.
- Times are local time in the mission zone.
- They are checked against the capture's whole stated interval, the same rule as for scanning schedules (see `time-and-schedules.md`).

## Matching

The search checks the current objects of each image: detections with operator corrections applied, minus removed ones, plus night candidates an operator confirmed. Each object gets **match**, **possible** with reasons, or no match:

| Fact | Established | Not established |
|---|---|---|
| Class | detector class is one of the target's classes | — |
| Subtype | never: the detector does not identify subtypes | possible: "pickup is not identified by the detector (detected as car)" |
| Excluded subtype | — | possible: "cannot rule out van" |
| Colour observed | match if wanted, no match if not | — |
| Colour uncertain | — | possible if a competing colour is wanted; no match if none is |
| Colour not observable | — | possible, with the reason (infrared, too small, too dark, burned out, mostly lamps) |
| Regions | inside the view's analysis regions, or inside a named region | possible if the view needs review and regions were not applied |

- With "only certain matches", possible results are left out and counted in the summary.
- Unresolved night light candidates are never searched, because type and colour cannot be known from lights. The summary reports how many there were.
- Named regions: the object's bottom-centre point must be inside one of the named regions of its view, whatever that region's kind. Images whose view has none of the named regions are counted, not searched.
- The summary also counts images that were outside the time, in another light condition, not yet processed, or below the minimum count. It counts colour readings by state.

## Colour reader

`trafficwb.attributes` records its version with every reading.

- It reads the central part of the box: roof, bonnet and doors, as seen from a raised camera.
- It uses the road around the vehicle as the reference. The road's tint is removed, and black and white are judged against the road's lightness. This stops a blue morning cast or orange street light being read as paint.
- Lit red and orange lamps and their glow are not counted as paint. White headlights are not masked, because they look like sunlit white paint.
- Colour is **not observable** for:
  - infrared or monochrome images;
  - boxes under 12 px wide or high, or under 300 px²;
  - a dark road around the vehicle;
  - burned-out crops;
  - crops that are mostly lamps.
- In dusk and night colour images, colour is at most **uncertain**.
- **Observed** needs one colour on at least 40% of the pixels, 15 points ahead of the next colour. Black needs 60%, because windows and shadows are dark too.
- Readings are cached per object box and algorithm version.
- Operator colour corrections are append-only, and later searches use them instead of the machine reading. Earlier search results stay as they were run.

## Review and export

- Each hit can be marked correct or not a match. Verdicts are append-only and the latest counts. They are evidence for later evaluation of the colour reader.
- Export writes two tables. One holds the hits. The other holds the request, the exact spec that ran, the operator's edits, the summary and the colour reader version. Files are never overwritten.
