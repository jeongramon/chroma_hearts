"""Independent SQLite decoding of unchanged experimental record payloads."""
import sqlite3
import struct

def vint(n):
    parts = [n & 127]
    while n >> 7:
        n >>= 7
        parts.insert(0, (n & 127) | 128)
    return bytes(parts)


def native_decode(payload: bytes, columns: int, templates: dict) -> tuple:
    """Place the unchanged payload into a new cell; SQLite interprets its header."""
    if columns not in templates:
        with sqlite3.connect(':memory:') as conn:
            conn.execute('PRAGMA page_size=4096')
            conn.execute('CREATE TABLE probe (' + ','.join(f'v{i}' for i in range(columns)) + ')')
            assert conn.execute("SELECT rootpage FROM sqlite_schema WHERE name='probe'").fetchone()[0] == 2
            templates[columns] = conn.serialize()
    size, total = 4096, len(payload)
    local = total
    if total > size - 35:
        minimum = ((size - 12) * 32 // 255) - 23
        k = minimum + ((total - minimum) % (size - 4))
        local = k if k <= size - 35 else minimum
    rest = payload[local:]
    overflow_count = (len(rest) + size - 5) // (size - 4)
    data = bytearray(templates[columns]) + bytearray(overflow_count * size)
    struct.pack_into('>I', data, 28, 2 + overflow_count)
    cell = vint(total) + b'\x01' + payload[:local]
    if rest:
        cell += struct.pack('>I', 3)
    start = size - len(cell)
    data[size:size * 2] = b'\x00' * size
    struct.pack_into('>BHHHBH', data, size, 13, 0, 1, start, 0, start)
    data[size + start:size * 2] = cell
    for i in range(overflow_count):
        page = i + 3
        base = (page - 1) * size
        struct.pack_into('>I', data, base, page + 1 if i + 1 < overflow_count else 0)
        part = rest[i * (size - 4):(i + 1) * (size - 4)]
        data[base + 4:base + 4 + len(part)] = part
    with sqlite3.connect(':memory:') as conn:
        conn.deserialize(bytes(data))
        result = conn.execute('SELECT * FROM probe').fetchall()
        assert len(result) == 1
        return result[0]
