# parse GGUF metadata: print general.architecture and a few key KVs
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_TYPE_UINT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]

def main(path):
    with open(path, "rb") as f:
        assert f.read(4) == GGUF_MAGIC, "not a GGUF file"
        version = read_u32(f)
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        kv = {}
        for _ in range(n_kv):
            key = read_string(f)
            t = read_u32(f)
            if t == GGUF_TYPE_STRING:
                val = read_string(f)
            elif t == GGUF_TYPE_UINT32:
                val = struct.unpack("<I", f.read(4))[0]
            elif t == GGUF_TYPE_ARRAY:
                at = read_u32(f)
                n = struct.unpack("<Q", f.read(8))[0]
                arr = []
                for _ in range(n):
                    if at == GGUF_TYPE_UINT32:
                        arr.append(struct.unpack("<I", f.read(4))[0])
                    elif at == GGUF_TYPE_STRING:
                        arr.append(read_string(f))
                    else:
                        arr.append("?")
                val = arr
            else:
                val = f"<type {t}>"
            kv[key] = val
        keep = ["general.architecture", "general.name", "llama.block_count",
                "llama.attention.head_count", "llama.embedding_length",
                "qwen3next.block_count", "qwen35.block_count", "qwen35moe.block_count",
                "gemma4.block_count", "minicpm5.block_count"]
        print(f"== {path}")
        for k in keep:
            if k in kv:
                print(f"   {k} = {kv[k]}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
