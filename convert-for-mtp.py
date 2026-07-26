#!/usr/bin/env python3
"""
Transplant extra tensors (e.g. MTP layers) from one GGUF file into another,
producing a mixed-quantization GGUF.

Note: Tested with ik_llama.cpp GGUF Python module.

Usage:
    python convert-for-mtp.py <target.gguf> <source.gguf> <output.gguf>

Arguments:
    target  — base GGUF (tensors + metadata kept as-is)
    source  — GGUF with extra blocks to transplant (e.g. blk.64.* for MTP)
    output  — resulting mixed-quantization GGUF

The script preserves the exact on-disk layout including per-row metadata
for quantization types like IQ4_KS that have row_meta_size > 0. This is
critical for GPU inference to work correctly.

Example:
    # Transplant MTP block from Q8_0 into IQ4_KS base model
    python convert-for-mtp.py "I:/models4/LuffyTheFox/Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact.gguf" Qwen3.6-35B-A3B-MTP-ONLY-Q4_K_M.gguf "./LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact-MTP.gguf"
"""

import hashlib
import sys
import struct
from pathlib import Path

from gguf import GGUFReader, GGUFValueType


def get_field_value(reader: GGUFReader, key: str):
    """Safely get a field value from GGUFReader."""
    field = reader.get_field(key)
    return field.contents() if field else None


def calculate_on_disk_sizes(tensors, file_size):
    """Calculate on-disk size for each tensor (including per-row metadata/padding)."""
    n_tensors = len(tensors)
    sizes = []
    for i in range(n_tensors):
        if i < n_tensors - 1:
            sizes.append(tensors[i + 1].data_offset - tensors[i].data_offset)
        else:
            sizes.append(file_size - tensors[i].data_offset)
    return sizes


def write_kv_value(fout, kv_type, value):
    """Write a KV value to the output file."""
    if kv_type == GGUFValueType.STRING:
        value_bytes = value.encode("utf-8")
        fout.write(struct.pack("<Q", len(value_bytes)))
        fout.write(value_bytes)
    elif kv_type == GGUFValueType.ARRAY:
        # This is handled separately in the main code
        pass
    elif kv_type in (GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.BOOL):
        fout.write(struct.pack("<B", value))
    elif kv_type in (GGUFValueType.UINT16, GGUFValueType.INT16):
        fout.write(struct.pack("<H", value))
    elif kv_type in (GGUFValueType.UINT32, GGUFValueType.INT32):
        fout.write(struct.pack("<I", value))
    elif kv_type == GGUFValueType.FLOAT32:
        fout.write(struct.pack("<f", value))
    elif kv_type in (GGUFValueType.UINT64, GGUFValueType.INT64):
        fout.write(struct.pack("<Q", value))
    elif kv_type == GGUFValueType.FLOAT64:
        fout.write(struct.pack("<d", value))


def write_array_value(fout, sub_type, arr):
    """Write an array KV value to the output file."""
    fout.write(struct.pack("<I", int(sub_type)))
    fout.write(struct.pack("<Q", len(arr)))

    for elem in arr:
        if sub_type == GGUFValueType.STRING:
            elem_bytes = elem.encode("utf-8")
            fout.write(struct.pack("<Q", len(elem_bytes)))
            fout.write(elem_bytes)
        elif sub_type in (GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.BOOL):
            fout.write(struct.pack("<B", elem))
        elif sub_type in (GGUFValueType.UINT16, GGUFValueType.INT16):
            fout.write(struct.pack("<H", elem))
        elif sub_type in (GGUFValueType.UINT32, GGUFValueType.INT32):
            fout.write(struct.pack("<I", elem))
        elif sub_type == GGUFValueType.FLOAT32:
            fout.write(struct.pack("<f", elem))
        elif sub_type in (GGUFValueType.UINT64, GGUFValueType.INT64):
            fout.write(struct.pack("<Q", elem))
        elif sub_type == GGUFValueType.FLOAT64:
            fout.write(struct.pack("<d", elem))


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <target.gguf> <source.gguf> <output.gguf>",
            file=sys.stderr,
        )
        sys.exit(1)

    target_path, source_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # ------------------------------------------------------------------
    # 1. Open both files
    # ------------------------------------------------------------------
    print(f"Reading target: {target_path}")
    target_reader = GGUFReader(target_path)

    print(f"Reading source: {source_path}")
    source_reader = GGUFReader(source_path)

    target_file_size = Path(target_path).stat().st_size
    source_file_size = Path(source_path).stat().st_size

    print(
        f"  Target tensors: {len(target_reader.tensors)}, KVs: {len([k for k in target_reader.fields if not k.startswith('GGUF.')])}"
    )
    print(
        f"  Source tensors: {len(source_reader.tensors)}, KVs: {len([k for k in source_reader.fields if not k.startswith('GGUF.')])}"
    )

    # ------------------------------------------------------------------
    # 2. Read architecture and MTP metadata from source
    # ------------------------------------------------------------------
    arch = get_field_value(target_reader, "general.architecture")
    if arch is None:
        print("ERROR: Target GGUF has no general.architecture key")
        sys.exit(1)

    source_block_count = get_field_value(source_reader, f"{arch}.block_count")
    source_nextn = get_field_value(source_reader, f"{arch}.nextn_predict_layers")

    if source_nextn is None:
        print("ERROR: Source GGUF has no nextn_predict_layers key")
        sys.exit(1)

    target_block_count = get_field_value(target_reader, f"{arch}.block_count")

    print(f"\n  Arch: {arch}")
    print(f"  Target block_count: {target_block_count}")
    print(
        f"  Source block_count: {source_block_count}, nextn_predict_layers: {source_nextn}"
    )

    # Identify extra tensors in the source (blocks beyond target's count)
    source_extra = [
        t
        for t in source_reader.tensors
        if t.name.startswith(f"blk.{target_block_count}.")
    ]
    print(f"\n  Extra tensors to transplant: {len(source_extra)}")

    if not source_extra:
        print(
            f"ERROR: No tensors found with prefix 'blk.{target_block_count}.' in source"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Prepare tensor lists and calculate sizes
    # ------------------------------------------------------------------
    # Combine tensors: all from target + extra from source
    all_tensors = list(target_reader.tensors) + source_extra

    # Calculate on-disk sizes for source tensors (including per-row metadata)
    target_on_disk_sizes = calculate_on_disk_sizes(
        target_reader.tensors, target_file_size
    )
    source_on_disk_sizes = calculate_on_disk_sizes(
        source_reader.tensors, source_file_size
    )

    # Create mapping for source tensors
    source_tensor_map = {
        t.name: (t, size)
        for t, size in zip(source_reader.tensors, source_on_disk_sizes)
    }

    # ------------------------------------------------------------------
    # 4. Write output file
    # ------------------------------------------------------------------
    print(f"\nWriting output: {output_path}")

    with (
        open(target_path, "rb") as target_fin,
        open(source_path, "rb") as source_fin,
        open(output_path, "wb") as fout,
    ):
        # 4.1 Write header
        # Magic (4 bytes)
        fout.write(b"GGUF")
        # Version (4 bytes)
        fout.write(struct.pack("<I", 3))
        # Tensor count (8 bytes)
        fout.write(struct.pack("<Q", len(all_tensors)))

        # Calculate KV count
        kv_count = len(
            [k for k in target_reader.fields.keys() if not k.startswith("GGUF.")]
        )
        kv_count += 1  # block_count override
        # Add source-only KVs (excluding block_count and nextn_predict_layers)
        for key in source_reader.fields:
            if (
                not key.startswith("GGUF.")
                and key not in target_reader.fields
                and key != f"{arch}.block_count"
                and key != f"{arch}.nextn_predict_layers"
            ):
                kv_count += 1
        # KV count (8 bytes)
        fout.write(struct.pack("<Q", kv_count))

        # 4.2 Write KV data from target (with block_count override)
        written_keys = set()

        for key, field in target_reader.fields.items():
            if key.startswith("GGUF."):
                continue

            # Skip block_count (we'll override it)
            if key == f"{arch}.block_count":
                continue

            # Write key
            key_bytes = key.encode("utf-8")
            fout.write(struct.pack("<Q", len(key_bytes)))
            fout.write(key_bytes)

            # Write type
            kv_type = field.types[0]
            fout.write(struct.pack("<I", int(kv_type)))

            # Write value
            if kv_type == GGUFValueType.STRING:
                write_kv_value(fout, kv_type, field.contents())
            elif kv_type == GGUFValueType.ARRAY:
                sub_type = (
                    field.types[1] if len(field.types) > 1 else GGUFValueType.FLOAT32
                )
                write_array_value(fout, sub_type, field.contents())
            else:
                write_kv_value(fout, kv_type, field.contents())

            written_keys.add(key)

        # Add block_count from source
        key = f"{arch}.block_count"
        key_bytes = key.encode("utf-8")
        fout.write(struct.pack("<Q", len(key_bytes)))
        fout.write(key_bytes)
        fout.write(struct.pack("<I", int(GGUFValueType.UINT32)))
        fout.write(struct.pack("<I", source_block_count))
        written_keys.add(key)

        # Add nextn_predict_layers from source
        key = f"{arch}.nextn_predict_layers"
        key_bytes = key.encode("utf-8")
        fout.write(struct.pack("<Q", len(key_bytes)))
        fout.write(key_bytes)
        fout.write(struct.pack("<I", int(GGUFValueType.UINT32)))
        fout.write(struct.pack("<I", source_nextn))
        written_keys.add(key)

        # Copy source-only KVs
        for key, field in source_reader.fields.items():
            if (
                key.startswith("GGUF.")
                or key in written_keys
                or key == f"{arch}.nextn_predict_layers"
            ):
                continue

            # Write key
            key_bytes = key.encode("utf-8")
            fout.write(struct.pack("<Q", len(key_bytes)))
            fout.write(key_bytes)

            # Write type
            kv_type = field.types[0]
            fout.write(struct.pack("<I", int(kv_type)))

            # Write value
            if kv_type == GGUFValueType.STRING:
                write_kv_value(fout, kv_type, field.contents())
            elif kv_type == GGUFValueType.ARRAY:
                sub_type = (
                    field.types[1] if len(field.types) > 1 else GGUFValueType.FLOAT32
                )
                write_array_value(fout, sub_type, field.contents())
            else:
                write_kv_value(fout, kv_type, field.contents())

        # 4.3 Write tensor info
        # Calculate offsets for all tensors
        current_offset = 0
        tensor_offsets = []

        for i, tensor in enumerate(all_tensors):
            if i < len(target_reader.tensors):
                size = target_on_disk_sizes[i]
            else:
                _, size = source_tensor_map[tensor.name]

            tensor_offsets.append(current_offset)
            current_offset += size

        # Write tensor info for each tensor
        for i, tensor in enumerate(all_tensors):
            # Tensor name
            name_bytes = tensor.name.encode("utf-8")
            fout.write(struct.pack("<Q", len(name_bytes)))
            fout.write(name_bytes)

            # Dimensions (in GGUF file order: fastest-varying first)
            shape = tensor.shape.tolist()
            fout.write(struct.pack("<I", len(shape)))
            for dim in shape:
                fout.write(struct.pack("<Q", dim))

            # Quantization type
            fout.write(struct.pack("<I", int(tensor.tensor_type)))

            # Offset
            fout.write(struct.pack("<Q", tensor_offsets[i]))

        # 4.4 Pad to alignment if needed
        current_pos = fout.tell()
        alignment = get_field_value(target_reader, "general.alignment") or 32
        padding_needed = (alignment - (current_pos % alignment)) % alignment
        if padding_needed:
            fout.write(b"\x00" * padding_needed)

        # 4.5 Copy tensor data
        print(f"Copying {len(all_tensors)} tensors...")
        for i, tensor in enumerate(all_tensors):
            if i < len(target_reader.tensors):
                # Target tensor
                offset = target_reader.tensors[i].data_offset
                size = target_on_disk_sizes[i]
                fin = target_fin
            else:
                # Source extra tensor
                src_tensor, size = source_tensor_map[tensor.name]
                offset = src_tensor.data_offset
                fin = source_fin

            fin.seek(offset)
            raw_data = fin.read(size)
            fout.write(raw_data)

            if (i + 1) % 50 == 0 or i == len(all_tensors) - 1:
                print(f"  Copied {i + 1}/{len(all_tensors)} tensors")

    # ------------------------------------------------------------------
    # 5. Verify output
    # ------------------------------------------------------------------
    output_size = Path(output_path).stat().st_size
    print(f"\nOutput: {output_path}")
    print(f"  Size: {output_size / 1_000_000_000:.2f} GB")
    print(f"  Tensors: {len(all_tensors)}")

    # Validate
    print("\nValidating output...")
    errors = []

    try:
        out_reader = GGUFReader(output_path)

        # Check block_count
        out_block_count = get_field_value(out_reader, f"{arch}.block_count")
        if out_block_count != source_block_count:
            errors.append(
                f"block_count: expected {source_block_count}, got {out_block_count}"
            )

        # Check nextn_predict_layers
        out_nextn = get_field_value(out_reader, f"{arch}.nextn_predict_layers")
        if out_nextn != source_nextn:
            errors.append(
                f"nextn_predict_layers: expected {source_nextn}, got {out_nextn}"
            )

        # Check extra tensors exist
        out_tensor_names = {t.name for t in out_reader.tensors}
        for tensor in source_extra:
            if tensor.name not in out_tensor_names:
                errors.append(f"Missing tensor: {tensor.name}")

        # Spot-check tensor data integrity
        print("  Spot-checking tensor data integrity...")
        out_tensors = {t.name: t for t in out_reader.tensors}

        # Check a target tensor
        for name in ["token_embd.weight"]:
            if name in out_tensors and name in {t.name for t in target_reader.tensors}:
                target_t = next(
                    (t for t in target_reader.tensors if t.name == name), None
                )
                out_t = out_tensors.get(name)
                if target_t and out_t:
                    target_hash = hashlib.sha256(target_t.data.tobytes()).hexdigest()[
                        :16
                    ]
                    out_hash = hashlib.sha256(out_t.data.tobytes()).hexdigest()[:16]
                    if target_hash == out_hash:
                        print(f"    {name}: OK ({out_hash})")
                    else:
                        errors.append(f"Data mismatch: {name}")

        # Check an extra tensor
        if source_extra:
            extra_name = source_extra[0].name
            source_t = source_tensor_map[extra_name][0]
            out_t = out_tensors.get(extra_name)
            if out_t:
                source_hash = hashlib.sha256(source_t.data.tobytes()).hexdigest()[:16]
                out_hash = hashlib.sha256(out_t.data.tobytes()).hexdigest()[:16]
                if source_hash == out_hash:
                    print(f"    {extra_name}: OK ({out_hash})")
                else:
                    errors.append(f"Data mismatch: {extra_name}")

    except Exception as e:
        errors.append(f"Failed to read output: {e}")

    if errors:
        print("\nVALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("  OK — all checks passed")
        print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    main()
