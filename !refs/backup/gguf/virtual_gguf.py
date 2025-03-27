import os
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VirtualGGUF:
    def __init__(self, virtual_space, model_path):
        self.virtual_space = virtual_space
        self.model_path = model_path
        self.blocks = virtual_space.models.get(model_path, {}).get("metadata", {})
        self.total_size = max(block["offset"] + block["size"] for block in self.blocks.values()) if self.blocks else 0

    def read(self, offset, length):
        data = bytearray()
        remaining = length
        current_offset = offset

        for block_key, block_info in sorted(self.blocks.items(), key=lambda x: x[1]["offset"]):
            block_start = block_info["offset"]
            block_end = block_start + block_info["size"]
            
            if current_offset >= block_end:
                continue
            if current_offset < block_start:
                gap = min(block_start - current_offset, remaining)
                data.extend(b"\0" * gap)
                current_offset += gap
                remaining -= gap
            
            if remaining <= 0:
                break

            block = self.virtual_space.load_block(block_key, self.model_path)
            if block is None:
                data.extend(b"\0" * min(remaining, block_end - block_start))
                current_offset += block_end - block_start
                remaining -= block_end - block_start
                continue

            block_data = block.numpy().tobytes()
            block_offset = current_offset - block_start
            read_size = min(remaining, block_end - current_offset)
            data.extend(block_data[block_offset:block_offset + read_size])
            current_offset += read_size
            remaining -= read_size

            if remaining <= 0:
                break

        logger.debug(f"Read {len(data)} bytes from offset {offset}")
        return bytes(data[:length])

    def size(self):
        return self.total_size

class VirtualGGUFFile:
    def __init__(self, virtual_gguf):
        self.vgguf = virtual_gguf
        self.pos = 0

    def read(self, n=-1):
        if n == -1:
            n = self.vgguf.size() - self.pos
        data = self.vgguf.read(self.pos, n)
        self.pos += len(data)
        return data

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.vgguf.size() + offset

    def tell(self):
        return self.pos

    def size(self):
        return self.vgguf.size()