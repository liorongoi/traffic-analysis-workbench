# Plates: rules

Requirements IMG-04 and IMG-05. Plates are read only in **plate recognition** missions (images). There is no owner lookup.

## Stages

1. Vehicle detection runs as in other missions.
2. For each car, bus, truck or motorcycle, the plate finder looks inside the vehicle box (5% margin; small vehicle crops are enlarged to 320 px first). Only a plate whose centre is inside the vehicle box counts, so a neighbour's plate caught by the margin is not taken. The best plate is kept.
3. The plate crop is saved under `evidence/plates/` and read.

Each vehicle gets one of four outcomes: plate found, no plate found, vehicle too small (box under 24 px), plate too small to read (under 12 px high). The last two are counted, never read.

The plate stage runs after night analysis in the same job. If it is cancelled, the plates not yet read are read at the next scan. A new finder or reader version reads every plate again; older reads stay stored with the models that made them.

## What is stored (machine output, never edited)

- plate box and finder score, crop file;
- **raw text** exactly as the reader emitted it (pad characters removed), in Unicode and reading order;
- every slot with its confidence;
- **normalised text**: separators removed, Latin upper case, Persian and Eastern Arabic digits written as ASCII digits, Persian and Arabic letters kept, leading zeros kept;
- display text (the layout's grouping, e.g. 12-345-67), country (the mission's), layout, the reader's own country guess;
- readability and the reasons for it;
- the finder and reader model versions and the reading rules.

## Readability

- A character under 0.60 confidence becomes **"?"**. It is never replaced by the most likely character for the layout.
- **Unreadable**: no confident character, or mean confidence under 0.40, or the plate is too small, or no installed reader covers the country's script.
- **Partial**: some "?", or a length outside the country's range, or a box that is not plate-shaped (width/height under 1.8: a two-line plate, a sign or a route number), or a finder score under 0.50.
- **Full**: everything else. A full read that fits no layout keeps its text and says "fits no known layout (foreign or special plate?)".

## Countries

| Country | Plates carry | Read by the installed reader | Layouts (working hypotheses, to be checked) |
|---|---|---|---|
| Israel | digits | yes | 8 digits as 123-45-678; 7 digits as 12-345-67 |
| Latvia | Latin letters and digits | yes | one or two letters and up to four digits; personalised 2–8 characters |
| Iran | Persian digits and letters | no: found, not read | two digits, letter, three digits, two-digit region |
| Lebanon | Latin letter, digits, Arabic script | no: found, not read | letter and up to six digits |

Q12 default: current standard private plates only. The layouts come from the design package's working understanding and are not verified against official sources or your samples.

## Evaluation only, and enabling a country

- Every country starts **evaluation only**. Its reads are shown and exported with that label.
- **Evaluate** compares the machine reads with plates an operator reviewed in this mission (confirmed, corrected or marked unreadable). It reports exact match and character error rate per layout, and counts machine "full" reads of plates the operator marked unreadable (guesses).
- It passes with at least 100 reviewed plates, at least 90% exact match, a character error rate of at most 5%, and **no guesses**. These defaults are recorded with each evaluation and are yours to change.
- **Enable** is allowed only after a passing evaluation, and only for a country the reader covers. It is recorded with the operator's name; "back to evaluation only" is recorded too.

## Operator review (append-only)

| Action | Meaning |
|---|---|
| Confirm | The machine read is right (only for full reads) |
| Correct | The operator types the plate. It must be fully readable: "?" is refused. Persian and Arabic input is stored as typed and normalised for comparison |
| Unreadable | The operator cannot read it either |
| No plate | Not a plate, or no plate visible |
| Reopen | Back to the machine read |

## Export

One row per vehicle looked at: file, checksum, capture time, vehicle, outcome, plate box, raw text, character confidences, machine text and readability, layout, country, the reader's country guess, reasons, review, final text and readability, number of operator changes, evaluation only (yes/no), and the finder and reader versions. Plate text is always written as text, so Excel keeps leading zeros.

## Not in this version

- Reading Persian and Arabic plates (needs a reader trained on such plates).
- Plates in video.
- Drawing a plate box by hand where the finder found none (the API accepts a box with a correction; the screen does not offer it yet).
- Plate overlays on the investigation screen.
