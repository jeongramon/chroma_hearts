"""Preserved headers, lost cell prefixes, overflow, and bytes-only controls."""
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from _sqlite_record_probe import local_size, scan_records


def vint(n):
    groups = [n & 127]
    while n >> 7:
        n >>= 7
        groups.insert(0, (n & 127) | 128)
    return bytes(groups)


def fixture(text, prefix=True, broken=False):
    size = 4096
    data = bytearray(size * 4)
    data[:16] = b'SQLite format 3\x00'
    struct.pack_into('>H', data, 16, size)
    struct.pack_into('>II', data, 32, 2, 3)
    struct.pack_into('>I', data, 56, 1)
    struct.pack_into('>IIII', data, size, 0, 2, 3, 4)
    raw = text.encode()
    serials = b'\x01\x2b' + vint(len(raw) * 2 + 13) + b'\x00\x00\x00'
    head = bytes([1 + len(serials)]) + serials
    payload = head + b'\x17chroma:document' + raw
    pos = size * 2 + 200
    local = local_size(len(payload), size, False)
    data[pos:pos + local] = payload[:local]
    if local < len(payload):
        struct.pack_into('>I', data, pos + local, 4)
        data[size * 3 + 4:size * 3 + 4 + len(payload) - local] = payload[local:]
    if prefix:
        pre = vint(len(payload)) + vint(77)
        data[pos - len(pre):pos] = pre
    if broken:
        data[pos] = 0
    return bytes(data), pos


class RecordProbeTests(unittest.TestCase):
    def test_fts_header_survives_overwritten_freeblock_prefix(self):
        text = 'document in a current freeblock ' * 40
        raw = text.encode()
        serials = b'\x00' + vint(len(raw) * 2 + 13)
        payload = bytes([len(serials) + 1]) + serials + raw
        data = bytearray(4096 * 2)
        data[:16] = b'SQLite format 3\x00'
        struct.pack_into('>H', data, 16, 4096)
        struct.pack_into('>I', data, 56, 1)
        data[4096] = 13
        struct.pack_into('>HHH', data, 4097, 200, 0, 200)
        struct.pack_into('>HH', data, 4296, 0, len(payload) + 4)
        data[4300:4300 + len(payload)] = payload
        rows, _ = scan_records(bytes(data))
        match = next(r for r in rows if r['document'] == text)
        self.assertEqual(match['cell_prefixes'], [])
        self.assertEqual(match['header_offset'], 4300)

    def test_record_without_cell_prefix_is_not_called_complete_cell(self):
        text = '남은 원문 "data"\n' * 30
        for prefix in (False, True):
            data, pos = fixture(text, prefix=prefix)
            rows, _ = scan_records(data)
            matches = [r for r in rows if r['document'] == text]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]['header_offset'], pos)
            self.assertEqual(bool(matches[0]['cell_prefixes']), prefix)
            self.assertEqual(matches[0]['internal_id'], 23)

    def test_text_alone_or_destroyed_header_does_not_count(self):
        text = 'complete original text ' * 80
        data, pos = fixture(text, broken=True)
        rows, _ = scan_records(data)
        self.assertFalse(any(r['document'] == text for r in rows))
        data = bytearray(data)
        data[pos - 20:pos + 40] = b'\x00' * 60
        rows, _ = scan_records(bytes(data))
        self.assertFalse(any(r['document'] == text for r in rows))

    def test_overflow_follows_valid_free_page_pointer(self):
        text = 'overflow document ' * 350
        data, pos = fixture(text)
        rows, _ = scan_records(data)
        match = next(r for r in rows if r['document'] == text)
        self.assertEqual(len(match['segments']), 2)
        corrupt = bytearray(data)
        struct.pack_into('>I', corrupt, pos + match['segments'][0]['length'], 3)
        rows, _ = scan_records(bytes(corrupt))
        self.assertFalse(any(r['document'] == text for r in rows))


if __name__ == '__main__':
    unittest.main()
