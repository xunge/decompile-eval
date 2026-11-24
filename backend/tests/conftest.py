"""Test database is isolated from users' actual data."""
import os
import tempfile
os.environ['DEC_EVAL_DATA_DIR'] = tempfile.mkdtemp(prefix='deceval_tests_')
