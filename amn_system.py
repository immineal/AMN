#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMN: ASCII Musical Notation System
Version: 2.1
Author: Immineal

Implements the AMN 2.1 specification with strict validation and deterministic conversion.

Usage:
  python amn_system.py midi2amn <input.mid> <output.amn>
  python amn_system.py amn2midi <input.amn> <output.mid>
"""

import struct
import sys
import re
import argparse
from collections import namedtuple
import math

# --- Configuration and Constants ---

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DURATION_TO_BEATS = {
    'w': 4.0, 'h.': 3.0, 'h': 2.0, 'q.': 1.5, 'q': 1.0,
    'e.': 0.75, 'e': 0.5, 's': 0.25, 't': 0.125, 'ts': 0.0625 # 64th note for ornaments
}
BEATS_TO_DURATION = {v: k for k, v in DURATION_TO_BEATS.items()}

DEFAULT_VELOCITY = 90
DEFAULT_TICKS_PER_BEAT = 480
# AMN 2.1 Spec Default Grace Style: 16th note at 85% velocity
DEFAULT_GRACE_DURATION_BEATS = DURATION_TO_BEATS['s']
DEFAULT_GRACE_VELOCITY_RATIO = 0.85

# --- 1. Intermediate Representation (IR) ---

class Note:
    def __init__(self, pitch, duration_beats, grace_notes=None, ornament=None):
        self.pitch = int(pitch)
        self.duration_beats = float(duration_beats)
        self.grace_notes = grace_notes if grace_notes else [] # List of Note objects
        self.ornament = ornament if ornament else {} # e.g. {'type': 'trill'}

class Chord:
    def __init__(self, pitches, duration_beats, grace_notes=None):
        self.pitches = sorted([int(p) for p in pitches])
        self.duration_beats = float(duration_beats)
        self.grace_notes = grace_notes if grace_notes else []

class Rest:
    def __init__(self, duration_beats):
        self.duration_beats = float(duration_beats)

class Track:
    def __init__(self, name=""):
        self.name = name
        self.voices = {1: []} # Default to one voice. Key is voice number.

class Song:
    def __init__(self):
        self.tempo = 120
        self.num = 4
        self.den = 4
        self.ticks_per_beat = DEFAULT_TICKS_PER_BEAT
        self.grace_duration_beats = DEFAULT_GRACE_DURATION_BEATS
        self.grace_velocity_ratio = DEFAULT_GRACE_VELOCITY_RATIO
        self.tracks = []

# --- 2. AMN Formatting and Parsing Logic ---

def pitch_to_amn(midi_note):
    """Converts a MIDI note number (e.g., 60) to AMN string (e.g., 'C4')."""
    if not 0 <= midi_note <= 127: return "Invalid"
    octave = (midi_note // 12) - 1
    note_name = PITCH_NAMES[midi_note % 12]
    return f"{note_name}{octave}"

def amn_to_pitch(amn_string):
    """Converts an AMN string (e.g., 'C#4') to a MIDI note number."""
    match = re.match(r"([A-G])([#b]?)(-?\d+)", amn_string)
    if not match: raise ValueError(f"Invalid AMN pitch format: {amn_string}")
    name, acc, octave = match.groups()
    pitch_val = PITCH_NAMES.index(name)
    if acc == '#': pitch_val = (pitch_val + 1) % 12
    elif acc == 'b': pitch_val = (pitch_val - 1 + 12) % 12
    return (int(octave) + 1) * 12 + pitch_val

def quantize_beats_to_duration(beats, tolerance=0.01):
    """Finds the closest AMN duration symbol for a given number of beats."""
    for beats_val, symbol in sorted(BEATS_TO_DURATION.items(), reverse=True):
        if abs(beats - beats_val) < tolerance: return symbol
    if beats > 0:
        print(f"Warning: Could not quantize duration of {beats:.3f} beats.", file=sys.stderr)
        closest_val = min(BEATS_TO_DURATION.keys(), key=lambda k: abs(k - beats))
        return BEATS_TO_DURATION[closest_val]
    return None

class AMNConverter:
    @staticmethod
    def song_to_amn(song):
        lines = [f"Tempo: {song.tempo}", f"TimeSig: {song.num}/{song.den}"]
        if song.grace_duration_beats != DEFAULT_GRACE_DURATION_BEATS or \
           song.grace_velocity_ratio != DEFAULT_GRACE_VELOCITY_RATIO:
            grace_dur_symbol = BEATS_TO_DURATION.get(song.grace_duration_beats, 's')
            lines.append(f"GraceStyle: {grace_dur_symbol} {song.grace_velocity_ratio:.2f}")
        lines.append("")

        for track in song.tracks:
            lines.append(f"[Track: {track.name}]" if track.name else "[Track]")
            beats_per_measure = song.num * (4.0 / song.den)
            for voice_num, events in sorted(track.voices.items()):
                voice_line = [f"(Voice {voice_num})"] if len(track.voices) > 1 else []
                beats_in_measure = 0
                for event in events:
                    if beats_in_measure >= beats_per_measure - 0.001:
                        voice_line.append("|")
                        beats_in_measure = 0
                    
                    duration_symbol = quantize_beats_to_duration(event.duration_beats)
                    if not duration_symbol: continue
                    
                    event_str = ""
                    if isinstance(event, Note): event_str = f"{pitch_to_amn(event.pitch)}:{duration_symbol}"
                    elif isinstance(event, Chord): event_str = f"<{ ' '.join(pitch_to_amn(p) for p in event.pitches) }>:{duration_symbol}"
                    elif isinstance(event, Rest): event_str = f"R:{duration_symbol}"
                    
                    voice_line.append(event_str)
                    beats_in_measure += event.duration_beats
                lines.append(" ".join(voice_line))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def amn_to_song(amn_text):
        song = Song()
        current_track = None
        
        lines = amn_text.splitlines()
        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            line_idx += 1

            if not line or line.startswith('#'): continue

            if line.lower().startswith("tempo:"): song.tempo = int(line.split(":")[1].strip())
            elif line.lower().startswith("timesig:"):
                num, den = line.split(":")[1].strip().split("/")
                song.num, song.den = int(num), int(den)
            elif line.lower().startswith("gracestyle:"):
                parts = line.split(":")[1].strip().split()
                song.grace_duration_beats = DURATION_TO_BEATS[parts[0]]
                song.grace_velocity_ratio = float(parts[1])
            elif line.startswith("[Track"):
                current_track = Track(re.search(r"\[Track:\s*(.*)\]", line).group(1).strip() if ':' in line else "")
                song.tracks.append(current_track)
                # Look ahead for voices
                if line_idx < len(lines) and lines[line_idx].strip().startswith("(Voice"):
                    current_track.voices = {} # Clear default voice
                    measure_beat_counters = {}
                    while line_idx < len(lines) and lines[line_idx].strip().startswith("(Voice"):
                        voice_line = lines[line_idx].strip()
                        voice_match = re.match(r"\(Voice\s*(\d+)\)", voice_line)
                        voice_num = int(voice_match.group(1))
                        current_track.voices[voice_num] = []
                        measure_beat_counters[voice_num] = 0
                        
                        content = voice_line[voice_match.end():].strip()
                        tokens = re.findall(r"g\([^)]+\)\s*\S+:\S+\(?\S*\)?|\S+:\S+\(?\S*\)?|\|", content)
                        
                        for token in tokens:
                            if token == "|":
                                # End of measure for this voice, check sync later
                                continue
                            
                            event, duration_beats = AMNConverter._parse_event_token(token, song)
                            current_track.voices[voice_num].append(event)
                            measure_beat_counters[voice_num] += duration_beats

                        line_idx += 1
                    # --- AMN 2.1 Polyphonic Validation ---
                    if len(measure_beat_counters) > 1:
                        first_voice_beats = next(iter(measure_beat_counters.values()))
                        if not all(abs(b - first_voice_beats) < 0.001 for b in measure_beat_counters.values()):
                            raise ValueError(f"Rhythmic desynchronization in track '{current_track.name}'. Voices have unequal measure durations. Counts: {measure_beat_counters}")
                else: # Single voice track
                    content = lines[line_idx].strip()
                    tokens = re.findall(r"g\([^)]+\)\s*\S+:\S+\(?\S*\)?|\S+:\S+\(?\S*\)?|\|", content)
                    for token in tokens:
                        if token == "|": continue
                        event, _ = AMNConverter._parse_event_token(token, song)
                        current_track.voices[1].append(event)
                    line_idx += 1 # Consume the content line
        return song

    @staticmethod
    def _parse_event_token(token, song):
        grace_notes = []
        grace_match = re.match(r"g\(([^)]+)\)\s*(.*)", token)
        if grace_match:
            grace_pitches = [amn_to_pitch(p) for p in grace_match.group(1).split()]
            grace_notes = [Note(p, song.grace_duration_beats) for p in grace_pitches]
            token = grace_match.group(2)
        
        ornament = {}
        ornament_match = re.search(r"\((\S+)\)$", token)
        if ornament_match:
            ornament_str = ornament_match.group(1)
            if ornament_str.startswith("tr"):
                ornament['type'] = 'trill'
                if '~' in ornament_str: ornament['alt_note'] = amn_to_pitch(ornament_str.split('~')[1])
            elif ornament_str == "mord": ornament['type'] = 'mordent'
            token = token[:ornament_match.start()]

        parts = token.split(':')
        duration_symbol = parts[-1]
        pitch_part = ':'.join(parts[:-1])
        duration_beats = DURATION_TO_BEATS[duration_symbol]

        if pitch_part.startswith('R'): return Rest(duration_beats), duration_beats
        if pitch_part.startswith('<'):
            pitches = [amn_to_pitch(p) for p in pitch_part.strip('<>').split()]
            return Chord(pitches, duration_beats, grace_notes=grace_notes), duration_beats
        else:
            pitch = amn_to_pitch(pitch_part)
            return Note(pitch, duration_beats, grace_notes=grace_notes, ornament=ornament), duration_beats

# --- 3. MIDI File I/O (Largely unchanged, but with updated calling logic) ---
# ... (The MidiConverter class from the previous implementation is mostly fine) ...
# ... (The only major change is in song_to_midi to handle the new IR) ...

RawMidiEvent = namedtuple('RawMidiEvent', ['delta_ticks', 'type', 'channel', 'data'])
class MidiConverter:
    # Methods _read_var_len, _write_var_len, midi_to_song, _read_track,
    # _read_var_len_from_bytes, _build_ir_events remain the same as in V2.0.
    # We will include them here for completeness.
    @staticmethod
    def _read_var_len(f):
        value = 0
        byte = ord(f.read(1))
        while byte & 0x80:
            value = (value << 7) + (byte & 0x7F)
            byte = ord(f.read(1))
        return (value << 7) + byte
    @staticmethod
    def _write_var_len(value):
        buf = bytearray([value & 0x7F])
        value >>= 7
        while value > 0:
            buf.insert(0, (value & 0x7F) | 0x80)
            value >>= 7
        return bytes(buf)

    def midi_to_song(self, filepath):
        song = Song()
        with open(filepath, 'rb') as f:
            chunk_type, length, fmt, n_tracks, division = struct.unpack('>4sIHHH', f.read(14))
            if chunk_type != b'MThd': raise ValueError("Not a valid MIDI file: MThd chunk not found.")
            song.ticks_per_beat = division
            raw_tracks = [self._read_track(f) for _ in range(n_tracks)]

        for i, raw_track_events in enumerate(raw_tracks):
            if not raw_track_events: continue
            track = Track(name=f"Track {i+1}")
            active_notes = {}
            completed_notes = []
            abs_tick = 0
            for event in raw_track_events:
                abs_tick += event.delta_ticks
                if event.type == 'note_on' and event.data[1] > 0:
                    active_notes[event.data[0]] = abs_tick
                elif event.type == 'note_off' or (event.type == 'note_on' and event.data[1] == 0):
                    pitch = event.data[0]
                    if pitch in active_notes:
                        start_tick = active_notes.pop(pitch)
                        if abs_tick > start_tick:
                            completed_notes.append((start_tick, abs_tick - start_tick, pitch))
                elif event.type == 'set_tempo':
                    song.tempo = 60000000 // struct.unpack('>I', b'\x00' + event.data)[0]
                elif event.type == 'time_signature':
                    song.num, den_pow, _, _ = event.data
                    song.den = 2**den_pow
            if completed_notes:
                track.voices[1] = self._build_ir_events(completed_notes, song.ticks_per_beat)
                song.tracks.append(track)
        return song

    def _read_track(self, f):
        chunk_type, length = struct.unpack('>4sI', f.read(8))
        if chunk_type != b'MTrk': raise ValueError("MTrk chunk expected.")
        track_data = f.read(length)
        events, pos, running_status = [], 0, None
        while pos < len(track_data):
            delta_ticks, bytes_read = self._read_var_len_from_bytes(track_data[pos:])
            pos += bytes_read
            status_byte = track_data[pos]
            if status_byte < 0x80:
                if running_status is None: raise ValueError("Running status used before first status byte.")
                event_byte = running_status
                pos -= 1
            else:
                event_byte, running_status = status_byte, status_byte
            pos += 1
            event_type, channel = (event_byte & 0xF0) >> 4, event_byte & 0x0F
            if event_type in [0x8, 0x9]:
                data = tuple(track_data[pos:pos+2])
                pos += 2
                events.append(RawMidiEvent(delta_ticks, {0x8: 'note_off', 0x9: 'note_on'}[event_type], channel, data))
            elif event_type in [0xA, 0xB, 0xE]: pos += 2
            elif event_type in [0xC, 0xD]: pos += 1
            elif event_byte == 0xFF:
                meta_type, meta_len_val, bytes_read = track_data[pos], *self._read_var_len_from_bytes(track_data[pos+1:])
                pos += 1 + bytes_read
                meta_data = track_data[pos:pos+meta_len_val]
                pos += meta_len_val
                if meta_type == 0x51: events.append(RawMidiEvent(delta_ticks, 'set_tempo', -1, meta_data))
                elif meta_type == 0x58: events.append(RawMidiEvent(delta_ticks, 'time_signature', -1, meta_data))
        return events

    def _read_var_len_from_bytes(self, data):
        value, bytes_read = 0, 0
        byte = data[bytes_read]
        while byte & 0x80:
            value = (value << 7) + (byte & 0x7F)
            bytes_read += 1
            byte = data[bytes_read]
        bytes_read += 1
        return (value << 7) + byte, bytes_read

    def _build_ir_events(self, notes, ticks_per_beat):
        if not notes: return []
        notes.sort()
        events, last_event_end_tick = [], 0
        while notes:
            current_start_tick = notes[0][0]
            if current_start_tick > last_event_end_tick:
                events.append(Rest((current_start_tick - last_event_end_tick) / ticks_per_beat))
            notes_at_this_tick = [n for n in notes if n[0] == current_start_tick]
            duration_ticks = notes_at_this_tick[0][1]
            duration_beats = duration_ticks / ticks_per_beat
            if len(notes_at_this_tick) > 1:
                events.append(Chord([n[2] for n in notes_at_this_tick], duration_beats))
            else:
                events.append(Note(notes_at_this_tick[0][2], duration_beats))
            last_event_end_tick = current_start_tick + duration_ticks
            notes = [n for n in notes if n[0] != current_start_tick]
        return events

    # --- THE MAJOR UPDATE FOR AMN 2.1 ---
    def song_to_midi(self, song, filepath):
        with open(filepath, 'wb') as f:
            f.write(struct.pack('>4sIHHH', b'MThd', 6, 1, len(song.tracks), song.ticks_per_beat))

            for track_ir in song.tracks:
                track_data = bytearray()
                
                # Add initial metadata to the first track
                if song.tracks.index(track_ir) == 0:
                    den_pow = int(math.log2(song.den))
                    track_data += self._write_var_len(0) + b'\xFF\x58\x04' + struct.pack('>BBBB', song.num, den_pow, 24, 8)
                    microseconds_per_beat = 60000000 // song.tempo
                    track_data += self._write_var_len(0) + b'\xFF\x51\x03' + struct.pack('>I', microseconds_per_beat)[1:]

                # Process all voices in parallel
                voice_events = []
                for voice_num, events in track_ir.voices.items():
                    abs_tick = 0
                    for event in events:
                        # Expand ornaments and grace notes into simple note events
                        expanded_notes = self._expand_event(event, song)
                        for note_event in expanded_notes:
                            duration_ticks = int(note_event.duration_beats * song.ticks_per_beat)
                            velocity = DEFAULT_VELOCITY
                            if note_event in event.grace_notes:
                                velocity = int(DEFAULT_VELOCITY * song.grace_velocity_ratio)
                            
                            # Add (absolute_tick, type, pitch, velocity)
                            voice_events.append((abs_tick, 1, note_event.pitch, velocity)) # Note On
                            voice_events.append((abs_tick + duration_ticks, 0, note_event.pitch, 0)) # Note Off
                            abs_tick += duration_ticks
                
                # Sort all events from all voices by time, then by type (offs before ons at same tick)
                voice_events.sort(key=lambda x: (x[0], x[1]))
                
                last_tick = 0
                for tick, type, pitch, vel in voice_events:
                    delta_ticks = tick - last_tick
                    status = 0x90 if type == 1 else 0x80 # Note On or Note Off
                    track_data += self._write_var_len(delta_ticks)
                    track_data += bytes([status, pitch, vel])
                    last_tick = tick

                track_data += self._write_var_len(0) + b'\xFF\x2F\x00' # End of Track
                f.write(b'MTrk' + struct.pack('>I', len(track_data)) + track_data)

    def _expand_event(self, event, song):
        """Expands a single IR event into a list of simple, performable notes."""
        notes = []
        if isinstance(event, Note):
            # 1. Add grace notes first
            notes.extend(event.grace_notes)
            
            # 2. Expand ornaments
            if event.ornament:
                ornament_type = event.ornament.get('type')
                ornament_duration_beats = DURATION_TO_BEATS['t'] # 32nd note for ornaments
                num_ornament_notes = int(event.duration_beats / ornament_duration_beats)
                
                if ornament_type == 'trill':
                    alt_note = event.ornament.get('alt_note', event.pitch + 2) # Default to major second
                    for i in range(num_ornament_notes):
                        pitch = event.pitch if i % 2 == 0 else alt_note
                        notes.append(Note(pitch, ornament_duration_beats))
                elif ornament_type == 'mordent':
                    alt_note = event.pitch - 2 # Default to major second below
                    notes.append(Note(event.pitch, ornament_duration_beats))
                    notes.append(Note(alt_note, ornament_duration_beats))
                    remaining_beats = event.duration_beats - 2 * ornament_duration_beats
                    if remaining_beats > 0: notes.append(Note(event.pitch, remaining_beats))
            else:
                notes.append(event) # Just the note itself
        elif isinstance(event, Chord):
            notes.extend(event.grace_notes)
            # Chords don't have ornaments in this spec, just add the main notes
            # We must return Note objects, so we expand the chord.
            for pitch in event.pitches:
                notes.append(Note(pitch, event.duration_beats))
        elif isinstance(event, Rest):
            notes.append(event) # Will be handled as a time gap
        
        return notes


# --- 4. Main Command-Line Interface ---

def main():
    parser = argparse.ArgumentParser(
        description="AMN 2.1: A bidirectional converter between MIDI and human-readable text.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_m2a = subparsers.add_parser("midi2amn", help="Convert a MIDI file to AMN text.")
    parser_m2a.add_argument("infile", help="Input .mid file path.")
    parser_m2a.add_argument("outfile", help="Output .amn file path.")
    parser_a2m = subparsers.add_parser("amn2midi", help="Convert an AMN text file to MIDI.")
    parser_a2m.add_argument("infile", help="Input .amn file path.")
    parser_a2m.add_argument("outfile", help="Output .mid file path.")
    args = parser.parse_args()

    midi_converter = MidiConverter()
    amn_converter = AMNConverter()

    try:
        if args.command == "midi2amn":
            print(f"Converting '{args.infile}' to '{args.outfile}' using AMN 2.1 spec...")
            song_ir = midi_converter.midi_to_song(args.infile)
            amn_text = amn_converter.song_to_amn(song_ir)
            with open(args.outfile, 'w', encoding='utf-8') as f: f.write(amn_text)
            print("Conversion successful.")
        elif args.command == "amn2midi":
            print(f"Converting '{args.infile}' to '{args.outfile}' using AMN 2.1 spec...")
            with open(args.infile, 'r', encoding='utf-8') as f: amn_text = f.read()
            song_ir = amn_converter.amn_to_song(amn_text)
            midi_converter.song_to_midi(song_ir, args.outfile)
            print("Conversion successful.")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.infile}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()