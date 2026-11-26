"""Test database is isolated from users' actual data."""
import os
import tempfile
os.environ['DEC_EVAL_DATA_DIR'] = tempfile.mkdtemp(prefix='deceval_tests_')

# Keep background NVML sampling out of unit tests.
os.environ.setdefault('DEC_EVAL_GPU_MONITOR', '0')
