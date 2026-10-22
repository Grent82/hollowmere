import os

# Memory vectors are a few thousand rows; one BLAS thread is plenty, and each extra
# OpenBLAS thread reserves its own buffer, which fails on a machine already low on
# memory with the model loaded. Must be set before numpy is first imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
