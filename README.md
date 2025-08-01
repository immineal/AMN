# AMN (ASCII Musical Notation) 2.1

**AMN** is a plain-text format for representing musical scores. It is designed for human readability, version control, and robust, bidirectional conversion with the Standard MIDI File format.

This document describes version 2.1 of the AMN specification, which introduces strict validation rules and clarifies the system's data integrity guarantees to ensure consistency and reliability.

## Core Philosophy

1.  **Human-First Authoring:** The syntax is designed to be written and read easily by musicians.
2.  **Guaranteed Consistency:** An AMN file will produce the exact same performance (MIDI output) regardless of the conversion tool used.
3.  **Data Integrity:** The system prioritizes the accurate, lossless representation of performed musical events.

## Data Integrity and Conversion Guarantees

AMN 2.1 makes the following explicit guarantees:

*   **MIDI -> AMN -> MIDI:** This conversion path is **100% lossless** for all note and timing data. The output MIDI file will be identical to the input MIDI file in terms of note events, timing, tempo, and time signatures.
*   **AMN -> MIDI:** This conversion is **deterministic**. An AMN file will always produce the exact same MIDI performance.
*   **Semantic Asymmetry:** AMN includes "semantic sugar" like ornament syntax (`(tr)`, `(mord)`). This is an authoring-time convenience. When converting from MIDI to AMN, the system represents the literal performance and does not attempt to guess the original semantic intent. Therefore, an ornament written in AMN will be converted to a literal sequence of notes on a round trip (`AMN -> MIDI -> AMN`).

## File Structure

An AMN file (`.amn`) consists of a global **Header** followed by one or more **Track** blocks. Comments start with `#`.

### Header

| Directive | Syntax | Default | Description |
| :--- | :--- | :--- | :--- |
| **`Tempo`** | `Tempo: <bpm>` | 120 | The initial tempo of the piece. |
| **`TimeSig`**| `TimeSig: <n>/<d>` | 4/4 | The initial time signature. |
| **`GraceStyle`**|`GraceStyle: <dur> <vel>`| `s 0.85`| **(New in 2.1)** Defines grace note performance. |

The `GraceStyle` directive uses a duration symbol (`t`, `s`, etc.) and a velocity ratio (0.0-1.0) to ensure consistent playback.

### Tracks and Core Events

The syntax for tracks, notes, chords, rests, and measures remains the same as in version 2.0.

## Advanced Features and Rules (AMN 2.1)

### 1. Polyphony and Voice Leading

Independent melodic lines are represented using `(Voice N)` prefixes.

**Validation Rule (2.1):** For every measure in a multi-voice track, the total duration of events in each voice **must be identical**. A compliant `amn2midi` converter MUST validate this and halt with an error if a desynchronization is found.

**Example of a VALID multi-voice measure:**
```amn
[Track: Piano]
(Voice 1) C5:h G4:h         |
(Voice 2) E4:q F4:q G4:q A4:q |
```

### 2. Complex Rhythms (Tuplets)

Tuplets use the `N:D{...}` syntax (e.g., `3:2{C4:e E4:e G4:e}`). The `T{...}` alias for triplets is retained.

### 3. Dynamic Changes

Inline `(Tempo: ...)` and `(TimeSig: ...)` directives are supported.

### 4. Musical Ornaments

Ornaments provide a shorthand for common performance patterns.

**Conversion Rule (2.1):**
-   **`amn2midi`:** Ornaments are expanded into a literal sequence of MIDI note events. For example, `C4:q(tr)` is converted into a rapid alternation between C4 and D4.
-   **`midi2amn`:** The converter will represent the exact notes from the MIDI file and will **not** attempt to collapse a performance back into ornament syntax.

**Grace Notes:** Grace note performance is now explicitly defined by the `GraceStyle` header directive, removing implementation ambiguity.

-   **Syntax:** `g(Note1 Note2...) MainNote:duration`
-   **Performance:** Each note inside `g(...)` is played with the duration and velocity ratio defined by `GraceStyle` (or its default) immediately before the main note.

## Complete Example (AMN 2.1)

This example demonstrates the well-defined nature of the 2.1 specification.

```amn
# A piece demonstrating AMN 2.1 features.
# The performance of this file is unambiguous.

Tempo: 90
TimeSig: 4/4
GraceStyle: t 0.8 # Grace notes are fast and slightly softer

[Track: Flute]
# A grace note leading into a trilled quarter note.
g(F#5) G5:q(tr) E5:h. |

[Track: Piano]
# This track is valid because both voices sum to a whole note in the first measure.
(Voice 1) C4:h E4:h     |
(Voice 2) R:q G3:q G3:h |
```