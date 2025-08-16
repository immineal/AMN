import unittest
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from amn_system import (
    Song, Note, Rest, Chord, Track, TempoChange, TimeSignatureChange,
    pitch_to_amn, amn_to_pitch,
    AMNConverter, AMNParsingError
)

class TestPitchConversion(unittest.TestCase):
    def test_pitch_to_amn(self):
        self.assertEqual(pitch_to_amn(60), "C4")
        self.assertEqual(pitch_to_amn(69), "A4")
        self.assertEqual(pitch_to_amn(48), "C3")
        self.assertEqual(pitch_to_amn(73), "C#5")

    def test_amn_to_pitch(self):
        self.assertEqual(amn_to_pitch("C4"), 60)
        self.assertEqual(amn_to_pitch("A4"), 69)
        self.assertEqual(amn_to_pitch("C#5"), 73)
        self.assertEqual(amn_to_pitch("Gb3"), 54)

class TestAmnToSong(unittest.TestCase):
    def setUp(self):
        self.amn_converter = AMNConverter()

    def test_simple_header_and_note(self):
        amn_text = """
Tempo: 100
TimeSig: 3/4

[Track: Melody]
C4:h. |
"""
        song = self.amn_converter.amn_to_song(amn_text)
        self.assertIsInstance(song, Song)
        self.assertEqual(song.tempo, 100)
        self.assertEqual(song.num, 3)
        self.assertEqual(song.den, 4)
        self.assertEqual(len(song.tracks), 1)

        track = song.tracks[0]
        self.assertEqual(track.name, "Melody")
        self.assertEqual(len(track.voices), 1)

        voice = track.voices[1]
        self.assertEqual(len(voice), 1)

        event = voice[0]
        self.assertIsInstance(event, Note)
        self.assertEqual(event.pitch, 60) # C4
        self.assertEqual(event.duration_beats, 3.0) # h.

    def test_triplet_tuplet(self):
        amn_text = """
[Track: Piano]
3:2{C4:e D4:e E4:e} F4:q
"""
        # A 3:2 tuplet of three 8th notes should take up the same time as two 8th notes (1 quarter note).
        # Total duration should be 2.0 beats (1.0 for the tuplet, 1.0 for the F4:q).
        # Each note in the tuplet should have duration 0.5 * (2/3) = 0.333...
        song = self.amn_converter.amn_to_song(amn_text)
        track = song.tracks[0]
        events = track.voices[1]

        self.assertEqual(len(events), 4)

        note1, note2, note3, note4 = events

        self.assertIsInstance(note1, Note)
        self.assertEqual(note1.pitch, 60) # C4
        self.assertAlmostEqual(note1.duration_beats, 0.5 * (2/3))

        self.assertIsInstance(note2, Note)
        self.assertEqual(note2.pitch, 62) # D4
        self.assertAlmostEqual(note2.duration_beats, 0.5 * (2/3))

        self.assertIsInstance(note3, Note)
        self.assertEqual(note3.pitch, 64) # E4
        self.assertAlmostEqual(note3.duration_beats, 0.5 * (2/3))

        self.assertIsInstance(note4, Note)
        self.assertEqual(note4.pitch, 65) # F4
        self.assertEqual(note4.duration_beats, 1.0)

        total_beats = sum(e.duration_beats for e in events)
        self.assertAlmostEqual(total_beats, 2.0)

    def test_inline_tempo_change(self):
        amn_text = """
[Track: Test]
C4:q (Tempo: 150) D4:q
"""
        song = self.amn_converter.amn_to_song(amn_text)
        track = song.tracks[0]
        events = track.voices[1]

        self.assertEqual(len(events), 3)

        self.assertIsInstance(events[0], Note)
        self.assertEqual(events[0].pitch, 60)

        self.assertIsInstance(events[1], TempoChange)
        self.assertEqual(events[1].tempo, 150)
        self.assertEqual(events[1].duration_beats, 0)

        self.assertIsInstance(events[2], Note)
        self.assertEqual(events[2].pitch, 62)

    def test_inline_directive_affects_midi(self):
        # Create a temp directory for MIDI files
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            amn_no_change = """
Tempo: 120
[Track: Test]
C4:q D4:q E4:h
"""
            amn_with_change = """
Tempo: 120
[Track: Test]
C4:q (Tempo: 240) D4:q E4:h
"""

            song_no_change = self.amn_converter.amn_to_song(amn_no_change)
            song_with_change = self.amn_converter.amn_to_song(amn_with_change)

            from amn_system import MidiConverter
            midi_converter = MidiConverter()

            path_no_change = os.path.join(tmpdir, "no_change.mid")
            path_with_change = os.path.join(tmpdir, "with_change.mid")

            midi_converter.song_to_midi(song_no_change, path_no_change)
            midi_converter.song_to_midi(song_with_change, path_with_change)

            with open(path_no_change, "rb") as f:
                midi_bytes_no_change = f.read()

            with open(path_with_change, "rb") as f:
                midi_bytes_with_change = f.read()

            self.assertNotEqual(midi_bytes_no_change, midi_bytes_with_change,
                                "MIDI output should differ when an inline tempo directive is added.")
            # A simple check: the file with the extra meta event should be larger
            self.assertGreater(len(midi_bytes_with_change), len(midi_bytes_no_change))

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.amn_converter = AMNConverter()

    def test_invalid_pitch_error(self):
        amn_text = """
[Track: Bad Note]
C4:q Z9:q
"""
        with self.assertRaises(AMNParsingError) as cm:
            self.amn_converter.amn_to_song(amn_text)

        self.assertEqual(cm.exception.line_number, 3)
        self.assertIn("Invalid AMN pitch format", cm.exception.message)
        self.assertIn("Z9:q", cm.exception.line_content)

if __name__ == '__main__':
    unittest.main()
