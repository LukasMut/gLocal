#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys

os.environ['PYTHONIOENCODING'] = 'UTF-8'

def parse_cuda_version(nvidia_smi_file : str) -> str:
    cuda_regex = r'\bCUDA\b(?=\b\sVersion\b)'
    version_regex = r'\d{2}(?=\.\d)'
    patch_regex = r'(?<=\.)\d+'
    file = open(nvidia_smi_file, 'r').read()
    m = re.compile(cuda_regex).search(file)
    start, _ = m.span()
    substring = file[start:]
    m = re.compile(version_regex).search(substring)
    start, _ = m.span()
    m = re.compile(patch_regex).search(substring)
    _, end = m.span()
    version = substring[start:end]
    return version

if __name__ == '__main__':
    nvidia_smi_file = sys.argv[1]
    cuda_version = parse_cuda_version(nvidia_smi_file)
    print(cuda_version)