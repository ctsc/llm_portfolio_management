"""portstudy: quant baseline plus LLM views in a weekly Black-Litterman pipeline.

Every linear-algebra problem in the study is small (about 100 stocks), and multithreaded
OpenBLAS makes those solves hundreds of times slower on the desktop than a single thread. The
BLAS thread pool is therefore limited to one thread for the whole process.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
except ImportError:  # threadpoolctl ships with scikit-learn; the env vars still apply
    pass
