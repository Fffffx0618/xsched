#!/usr/bin/env python3
"""Check the SM86 guardian blob against inject.cu using CUDA 11.5 ptxas.

The four BPT markers become debugger-parameter LDCs; RET and padding are
removed so execution falls through into the copied original kernel.
"""
import re
import struct
import subprocess
import tempfile
from pathlib import Path


def words(text):
    return [int(value, 16) for value in re.findall(r"/\* 0x([0-9a-f]{16}) \*/", text)]


def main():
    here = Path(__file__).resolve().parent
    version = subprocess.check_output(["nvcc", "--version"], text=True)
    if "release 11.5," not in version:
        raise SystemExit("This blob is generated with CUDA 11.5; use that toolkit to verify it.")
    with tempfile.TemporaryDirectory(prefix="xsched-sm86-") as directory:
        binary = Path(directory) / "inject.cubin"
        subprocess.run(["nvcc", "-cubin", str(here / "inject.cu"), "-o", str(binary),
                        "--keep-device-functions", "-Xptxas", "-astoolspatch", "-arch=sm_86"], check=True)
        assembly = subprocess.check_output(["cuobjdump", "-sass", str(binary)], text=True)
    function = assembly.split("Function : check_preempt\n", 1)[1].split("Function :", 1)[0]
    ret = re.search(r"/\*([0-9a-f]+)\*/\s+RET\.ABS", function)
    if not ret or int(ret.group(1), 16) != 0x2D0:
        raise SystemExit("Unexpected guardian layout; inspect ptxas output before changing the blob.")
    data = words(function[:ret.start()])
    for offset, instruction in [(0x10, 0x00062000FF047B82), (0x20, 0x00062100FF057B82),
                                (0xC0, 0x00062400FF067B82), (0xD0, 0x00062500FF077B82)]:
        index = offset // 8
        if data[index] != 0x000000040000795C:
            raise SystemExit(f"Missing BPT marker at {offset:#x}")
        data[index:index + 2] = [instruction, 0x000FC00000000800]
    source = (here.parent / "src/arch/sm86.cpp").read_text()
    array = source.split("guardian_instructions[]", 1)[1].split("};", 1)[0]
    actual = [int(value, 16) for value in re.findall(r"0x([0-9a-f]{16})", array)]
    recorded = (here / "inject_sm86.asm").read_text().split("Function : check_preempt\n", 1)[1]
    recorded = recorded.split("// restore_exec", 1)[0]
    if data != actual or data != words(recorded):
        raise SystemExit("SM86 guardian source, disassembly and embedded blob differ")
    print(f"SM86 guardian: source / assembly / blob match ({len(struct.pack('<' + 'Q' * len(data), *data))} bytes)")


if __name__ == "__main__":
    main()
