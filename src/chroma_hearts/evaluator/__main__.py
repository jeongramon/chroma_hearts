"""Allow ``python -m chroma_hearts.evaluator [-i RUN_DIR] [-o REPORT_BASE]``."""

from chroma_hearts.config import parse_evaluator_args

from .main import main

main(parse_evaluator_args())
