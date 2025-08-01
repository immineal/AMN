#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMN: ASCII Musical Notation System
A robust, bidirectional converter between MIDI and a human-readable text format.

Author: AI Assistant
Version: 1.0

Usage:
  python amn_system.py midi2amn <input.mid> <output.amn>
  python amn_system.py amn2midi <input.amn> <output.mid>
"""

import struct
import sys
import re
import argparse
from collections import namedtuple

# --- Configuration and Constants ---

# MIDI note numbers for one octave. C4 is Middle C (MIDI note 60)
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapping from AMN duration symbols to beat fractions
# (e.g., 'q' is a quarter note, which is 1.0 beats in a 4/4 signature)
DURATION_TO_BEATS = {
    'w': 4.0, 'h.': 3.0, 'h': 2.0, 'q.': 1.5, 'q': 1.0,
    'e.': 0.75, 'e': 0.5, 's': 0.25, 't': 0.125
}
# Reverse mapping for quantization
BEATS_TO_DURATION = {v: k for k, v in DURATION_TO_BEATS.items()}

# Default values for MIDI generation
DEFAULT_VELOCITY = 90
DEFAULT_TICKS_PER_BEAT = 480

# --- 1. Intermediate Representation (IR) ---
# These classes represent the music abstractly.

class Note:
    def __init__(self, pitch, duration_beats):
        self.pitch = int(pitch)  # MIDI note number (0-127)
        self.duration_beats = float(duration_beats)

class Chord:
    def __init__(self, pitches, duration_beats):
        self.pitches = sorted([int(p) for p in pitches]) # List of MIDI numbers
        self.duration_beats = float(duration_beats)

class Rest:
    def __init__(self, duration_beats):
        self.duration_beats = float(duration_beats)

class Track:
    def __init__(self, name=""):
        self.name = name
        self.events = [] # A list of Note, Chord, or Rest objects

class Song:
    def __init__(self):
        self.tempo = 120
        self.num = 4
        self.den = 4
        self.ticks_per_beat = DEFAULT_TICKS_PER_BEAT
        self.tracks = []

# --- 2. AMN Formatting and Parsing Logic ---

def pitch_to_amn(midi_note):
    """Converts a MIDI note number (e.g., 60) to AMN string (e.g., 'C4')."""
    if not 0 <= midi_note <= 127:
        raise ValueError("MIDI note out of range (0-127)")
    octave = (midi_note // 12) - 1
    note_name = PITCH_NAMES[midi_note % 12]
    return f"{note_name}{octave}"

def amn_to_pitch(amn_string):
    """Converts an AMN string (e.g., 'C#4') to a MIDI note number."""
    match = re.match(r"([A-G])([#b]?)(-?\d+)", amn_string)
    if not match:
        raise ValueError(f"Invalid AMN pitch format: {amn_string}")

    name, acc, octave = match.groups()
    pitch_val = PITCH_NAMES.index(name)

    if acc == '#':
        pitch_val = (pitch_val + 1) % 12
    elif acc == 'b': # Handle flats by converting to sharps, e.g. Db -> C#
        name = PITCH_NAMES[(pitch_val - 1) % 12]
        pitch_val = PITCH_NAMES.index(name)

    # Re-get the non-accidental name for index lookup
    if len(name) > 1:
        name = name[0]
    pitch_val = PITCH_NAMES.index(name)
    
    if acc == '#':
        pitch_val += 1
    elif acc == 'b':
        pitch_val -=1

    return (int(octave) + 1) * 12 + pitch_val


def quantize_beats_to_duration(beats, tolerance=0.05):
    """Finds the closest AMN duration symbol for a given number of beats."""
    for beats_val, symbol in sorted(BEATS_TO_DURATION.items(), reverse=True):
        if abs(beats - beats_val) < tolerance:
            return symbol
    # Fallback for complex rhythms not perfectly represented
    if beats > 0:
        print(f"Warning: Could not perfectly quantize duration of {beats:.2f} beats. Approximating.", file=sys.stderr)
        # Find the closest without tolerance
        closest_val = min(BEATS_TO_DURATION.keys(), key=lambda k: abs(k - beats))
        return BEATS_TO_DURATION[closest_val]
    return None # Should not happen for positive beats


class AMNConverter:
    """Handles conversion between the IR and AMN text format."""

    @staticmethod
    def song_to_amn(song):
        """Converts an IR Song object to an AMN formatted string."""
        lines = []
        lines.append(f"Tempo: {song.tempo}")
        lines.append(f"TimeSig: {song.num}/{song.den}")
        lines.append("")

        for track in song.tracks:
            if track.name:
                lines.append(f"[Track: {track.name}]")
            else:
                lines.append("[Track]")

            beats_in_measure = 0
            beats_per_measure = song.num * (4.0 / song.den)
            track_line = []

            for event in track.events:
                if beats_in_measure >= beats_per_measure - 0.001: # Epsilon for float issues
                    track_line.append("|")
                    beats_in_measure = 0

                duration_symbol = quantize_beats_to_duration(event.duration_beats)
                if not duration_symbol: continue

                if isinstance(event, Note):
                    pitch_str = pitch_to_amn(event.pitch)
                    track_line.append(f"{pitch_str}:{duration_symbol}")
                elif isinstance(event, Chord):
                    pitch_strs = " ".join(pitch_to_amn(p) for p in event.pitches)
                    track_line.append(f"<{pitch_strs}>:{duration_symbol}")
                elif isinstance(event, Rest):
                    track_line.append(f"R:{duration_symbol}")

                beats_in_measure += event.duration_beats

            lines.append(" ".join(track_line))
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def amn_to_song(amn_text):
        """Parses an AMN string into an IR Song object."""
        song = Song()
        current_track = None

        # Regex to find all musical events, including chords
        event_pattern = re.compile(
            r"(<[^>]+>:\w+\.?|[\w#b]+\-?\d+:\w+\.?|R:\w+\.?|\|)"
        )

        for line in amn_text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.lower().startswith("tempo:"):
                song.tempo = int(line.split(":")[1].strip())
            elif line.lower().startswith("timesig:"):
                num, den = line.split(":")[1].strip().split("/")
                song.num = int(num)
                song.den = int(den)
            elif line.startswith("[Track"):
                current_track = Track()
                match = re.search(r"\[Track:\s*(.*)\]", line)
                if match and match.group(1):
                    current_track.name = match.group(1).strip()
                song.tracks.append(current_track)
            elif current_track:
                tokens = event_pattern.findall(line)
                for token in tokens:
                    if token == '|': continue

                    parts = token.split(':')
                    duration_symbol = parts[-1]
                    pitch_part = ':'.join(parts[:-1])
                    
                    try:
                        duration_beats = DURATION_TO_BEATS[duration_symbol]
                    except KeyError:
                        raise ValueError(f"Unknown duration symbol: {duration_symbol}")

                    if pitch_part.startswith('R'):
                        current_track.events.append(Rest(duration_beats))
                    elif pitch_part.startswith('<'):
                        pitch_strs = pitch_part.strip('<>').split()
                        pitches = [amn_to_pitch(p) for p in pitch_strs]
                        current_track.events.append(Chord(pitches, duration_beats))
                    else:
                        pitch = amn_to_pitch(pitch_part)
                        current_track.events.append(Note(pitch, duration_beats))
        return song


# --- 3. MIDI File I/O (From Scratch) ---

RawMidiEvent = namedtuple('RawMidiEvent', ['delta_ticks', 'type', 'channel', 'data'])

class MidiConverter:
    """Handles reading and writing of Standard MIDI Files."""

    @staticmethod
    def _read_var_len(f):
        """Reads a MIDI variable-length quantity."""
        value = 0
        byte = ord(f.read(1))
        while byte & 0x80:
            value = (value << 7) + (byte & 0x7F)
            byte = ord(f.read(1))
        return (value << 7) + byte

    @staticmethod
    def _write_var_len(value):
        """Writes a MIDI variable-length quantity."""
        buf = bytearray()
        buf.append(value & 0x7F)
        value >>= 7
        while value > 0:
            buf.insert(0, (value & 0x7F) | 0x80)
            value >>= 7
        return bytes(buf)

    def midi_to_song(self, filepath):
        """Reads a .mid file and converts it into an IR Song object."""
        song = Song()
        with open(filepath, 'rb') as f:
            # Header Chunk
            chunk_type, length, fmt, n_tracks, division = struct.unpack('>4sIHHH', f.read(14))
            if chunk_type != b'MThd':
                raise ValueError("Not a valid MIDI file: MThd chunk not found.")
            song.ticks_per_beat = division

            raw_tracks = []
            for _ in range(n_tracks):
                raw_tracks.append(self._read_track(f))

        # Process raw events into the IR
        for i, raw_track_events in enumerate(raw_tracks):
            if not raw_track_events: continue
            
            track = Track(name=f"Track {i+1}")
            active_notes = {}  # {pitch: start_tick}
            completed_notes = [] # [(start_tick, duration_ticks, pitch)]
            
            abs_tick = 0
            for event in raw_track_events:
                abs_tick += event.delta_ticks
                
                if event.type == 'note_on' and event.data[1] > 0:
                    active_notes[event.data[0]] = abs_tick
                elif event.type == 'note_off' or (event.type == 'note_on' and event.data[1] == 0):
                    pitch = event.data[0]
                    if pitch in active_notes:
                        start_tick = active_notes.pop(pitch)
                        duration_ticks = abs_tick - start_tick
                        if duration_ticks > 0:
                            completed_notes.append((start_tick, duration_ticks, pitch))
                elif event.type == 'set_tempo':
                    # Tempo is in microseconds per quarter note
                    song.tempo = 60000000 // struct.unpack('>I', b'\x00' + event.data)[0]
                elif event.type == 'time_signature':
                    song.num, den_pow, _, _ = event.data
                    song.den = 2**den_pow

            # Quantize and group notes, chords, and rests
            if completed_notes:
                track.events = self._build_ir_events(completed_notes, song.ticks_per_beat)
                song.tracks.append(track)
        
        return song

    def _read_track(self, f):
        """Reads a single MTrk chunk."""
        chunk_type, length = struct.unpack('>4sI', f.read(8))
        if chunk_type != b'MTrk':
            raise ValueError("MTrk chunk expected.")
        
        track_data = f.read(length)
        
        events = []
        pos = 0
        running_status = None
        while pos < len(track_data):
            delta_ticks, bytes_read = self._read_var_len_from_bytes(track_data[pos:])
            pos += bytes_read
            
            status_byte = track_data[pos]
            if status_byte < 0x80: # Running status
                if running_status is None:
                    raise ValueError("Running status used before first status byte.")
                event_byte = running_status
                pos -= 1 # Reread data byte
            else:
                event_byte = status_byte
                running_status = event_byte

            pos += 1
            event_type = (event_byte & 0xF0) >> 4
            channel = event_byte & 0x0F
            
            if event_type in [0x8, 0x9, 0xA, 0xB, 0xE]: # 2 data bytes
                data = tuple(track_data[pos:pos+2])
                pos += 2
                event_name = {0x8: 'note_off', 0x9: 'note_on'}.get(event_type)
                if event_name:
                    events.append(RawMidiEvent(delta_ticks, event_name, channel, data))
            elif event_type in [0xC, 0xD]: # 1 data byte
                pos += 1
            elif event_byte == 0xFF: # Meta Event
                meta_type = track_data[pos]
                pos += 1
                meta_len, bytes_read = self._read_var_len_from_bytes(track_data[pos:])
                pos += bytes_read
                meta_data = track_data[pos:pos+meta_len]
                pos += meta_len

                if meta_type == 0x51: # Set Tempo
                    events.append(RawMidiEvent(delta_ticks, 'set_tempo', -1, meta_data))
                elif meta_type == 0x58: # Time Signature
                    events.append(RawMidiEvent(delta_ticks, 'time_signature', -1, meta_data))
                # We ignore other meta events (like track name, copyright, etc.) per requirements
        
        return events

    def _read_var_len_from_bytes(self, data):
        value = 0
        bytes_read = 0
        byte = data[bytes_read]
        while byte & 0x80:
            value = (value << 7) + (byte & 0x7F)
            bytes_read += 1
            byte = data[bytes_read]
        bytes_read += 1
        return (value << 7) + byte, bytes_read

    def _build_ir_events(self, notes, ticks_per_beat):
        """Converts a list of (start, duration, pitch) into IR events."""
        if not notes:
            return []
        
        # Sort by start time, then pitch
        notes.sort()
        
        events = []
        last_event_end_tick = 0
        
        while notes:
            current_start_tick = notes[0][0]
            
            # Check for a rest
            if current_start_tick > last_event_end_tick:
                rest_duration_ticks = current_start_tick - last_event_end_tick
                rest_duration_beats = rest_duration_ticks / ticks_per_beat
                events.append(Rest(rest_duration_beats))

            # Group notes starting at the same time (chords)
            notes_at_this_tick = [n for n in notes if n[0] == current_start_tick]
            
            # For simplicity, assume all notes in a chord have the same duration
            # This is a reasonable assumption for quantized music.
            chord_duration_ticks = notes_at_this_tick[0][1]
            duration_beats = chord_duration_ticks / ticks_per_beat
            
            if len(notes_at_this_tick) > 1:
                pitches = [n[2] for n in notes_at_this_tick]
                events.append(Chord(pitches, duration_beats))
            else:
                pitch = notes_at_this_tick[0][2]
                events.append(Note(pitch, duration_beats))

            last_event_end_tick = current_start_tick + chord_duration_ticks
            # Remove processed notes
            notes = [n for n in notes if n[0] != current_start_tick]
            
        return events


    def song_to_midi(self, song, filepath):
        """Writes an IR Song object to a .mid file."""
        with open(filepath, 'wb') as f:
            # Header
            f.write(struct.pack('>4sIHHH', b'MThd', 6, 1, len(song.tracks), song.ticks_per_beat))

            for track_ir in song.tracks:
                track_data = bytearray()
                
                # Add initial metadata to the first track
                if song.tracks.index(track_ir) == 0:
                    # Time signature
                    den_pow = {1:0, 2:1, 4:2, 8:3, 16:4}.get(song.den, 2)
                    track_data += self._write_var_len(0) + b'\xFF\x58\x04' + struct.pack('>BBBB', song.num, den_pow, 24, 8)
                    # Tempo
                    microseconds_per_beat = 60000000 // song.tempo
                    track_data += self._write_var_len(0) + b'\xFF\x51\x03' + struct.pack('>I', microseconds_per_beat)[1:]

                last_event_tick = 0
                for event_ir in track_ir.events:
                    duration_ticks = int(event_ir.duration_beats * song.ticks_per_beat)
                    
                    # Rests are just time gaps, so we add their duration to the next event's delta
                    if isinstance(event_ir, Rest):
                        last_event_tick += duration_ticks
                        continue
                    
                    delta_ticks = last_event_tick
                    last_event_tick = 0 # Reset for next event delta
                    
                    # Generate note on/off pairs
                    on_events = []
                    off_events = []
                    
                    pitches = event_ir.pitches if isinstance(event_ir, Chord) else [event_ir.pitch]
                    for pitch in pitches:
                        on_events.append( (0, b'\x90' + bytes([pitch, DEFAULT_VELOCITY])) )
                        off_events.append( (duration_ticks, b'\x80' + bytes([pitch, 0])) )

                    # Sort all events by time, then type (on before off)
                    all_sub_events = sorted(on_events + off_events)

                    current_tick = 0
                    for tick, data in all_sub_events:
                        d_tick = tick - current_tick
                        track_data += self._write_var_len(delta_ticks + d_tick)
                        track_data += data
                        current_tick = tick
                        delta_ticks = 0 # Only first event has the large delta

                # End of Track meta event
                track_data += self._write_var_len(0) + b'\xFF\x2F\x00'
                
                # Write MTrk chunk
                f.write(b'MTrk')
                f.write(struct.pack('>I', len(track_data)))
                f.write(track_data)


# --- 4. Main Command-Line Interface ---

def main():
    parser = argparse.ArgumentParser(
        description="AMN: A bidirectional converter between MIDI and human-readable text.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # midi2amn command
    parser_m2a = subparsers.add_parser("midi2amn", help="Convert a MIDI file to AMN text.")
    parser_m2a.add_argument("infile", help="Input .mid file path.")
    parser_m2a.add_argument("outfile", help="Output .amn file path.")

    # amn2midi command
    parser_a2m = subparsers.add_parser("amn2midi", help="Convert an AMN text file to MIDI.")
    parser_a2m.add_argument("infile", help="Input .amn file path.")
    parser_a2m.add_argument("outfile", help="Output .mid file path.")

    args = parser.parse_args()

    midi_converter = MidiConverter()
    amn_converter = AMNConverter()

    try:
        if args.command == "midi2amn":
            print(f"Converting '{args.infile}' to '{args.outfile}'...")
            song_ir = midi_converter.midi_to_song(args.infile)
            amn_text = amn_converter.song_to_amn(song_ir)
            with open(args.outfile, 'w', encoding='utf-8') as f:
                f.write(amn_text)
            print("Conversion successful.")

        elif args.command == "amn2midi":
            print(f"Converting '{args.infile}' to '{args.outfile}'...")
            with open(args.infile, 'r', encoding='utf-8') as f:
                amn_text = f.read()
            song_ir = amn_converter.amn_to_song(amn_text)
            midi_converter.song_to_midi(song_ir, args.outfile)
            print("Conversion successful.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.infile}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        # For debugging, you might want to print the full traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()