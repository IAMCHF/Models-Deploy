#!/bin/bash
# Check requirements for all 8 failed models
for model in aratako-miocodec-25hz-44-1khz-v2 datadog-toto-2-0-22m google-videoprism-lvt-base-f16r288 ibm-granite-granite-timeseries-patchtst-fm-r1 ibm-research-ttm-r3 neoquasar-kronos-base openmoss-team-moss-voicegenerator yuchenshen-fomo-0d; do
  echo "========================================"
  echo "=== $model ==="
  echo "========================================"
  echo "--- requirements.txt ---"
  cat /workspace/models/$model/requirements.txt 2>/dev/null
  echo ""
  echo "--- app.py imports (non-standard) ---"
  grep -E "^(from|import) " /workspace/models/$model/app.py 2>/dev/null | grep -vE "from (fastapi|pydantic|uvicorn|os|base64|json|logging|pathlib|typing|io|tempfile|time|numpy|torch|transformers|PIL|requests|collections|dataclasses|enum|re|sys|math|random|subprocess|shutil|glob|warnings|functools|contextlib|abc|threading|multiprocessing|concurrent|asyncio|datetime|hashlib|struct|wave|audioop|array|statistics|heapq|bisect|itertools|operator|fractions|decimal|string|unicodedata|codecs|copy|pickle|shelve|marshal|dbm|sqlite3|zlib|gzip|bz2|lzma|zipfile|tarfile|csv|json|xml|html|urllib|http|ftplib|socket|ssl|email|smtplib|imaplib|telnetlib|poplib|nntplib|webbrowser|cgi|cgitb|wsgiref|argparse|optparse|getopt|fileinput|linecache|tokenize|token|keyword|symbol|ast|parser|dis|inspect|pdb|profile|pstats|timeit|trace|turtle|tkinter|tix|scrolledtext|ttk|idlelib|test|venv|ensurepip|distutils|setuptools|pkg_resources|pip|wheel|site|sysconfig|platform|os|posix|pwd|grp|spwd|tty|pty|fcntl|resource|select|selectors|signal|mmap|msvcrt|winreg|winsound|imp|importlib|zipimport|runpy|modulefinder|pkgutil|code|codeop|difflib|fnmatch|glob|pathlib|shutil|tempfile|atexit|gc|inspect|linecache|tokenize|token|keyword|symbol|ast|parser|dis|inspect|pdb|profile|pstats|timeit|trace|turtle|tkinter|tix|scrolledtext|ttk|idlelib|test|venv|ensurepip|distutils|setuptools|pkg_resources|pip|wheel|site|sysconfig|platform)" 2>/dev/null
  echo ""
done
