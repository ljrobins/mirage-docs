"""
Counting Source Code Lines
==========================
This script counts the number of lines in the Mirage source code.
"""

import os

import mirage as mr


def count_lines_in_dir(rel_dir):
    lines = 0
    for fname in os.listdir(rel_dir):
        if ".py" in fname:
            with open(os.path.join(rel_dir, fname), "r") as f:
                lines += len(f.readlines())
    return lines


rel_dirs = [
    ".",
    "sim",
    "synth",
    "vis",
]

lines = 0
for rel_dir in rel_dirs:
    dir_ = os.path.join(os.environ['SRCDIR'], rel_dir)
    lines += count_lines_in_dir(dir_)

test_lines = count_lines_in_dir( os.path.join(os.environ['SRCDIR'], 'tests'))

print(f"Mirage Python source code has {lines} lines")
print(f"Mirage Python test code has {test_lines} lines")