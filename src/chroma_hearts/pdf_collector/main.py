"""
Collect sample PDFs from arXiv for Chroma Hearts.
"""

from argparse import Namespace
from pathlib import Path
import random
import time

import requests

from chroma_hearts.config import get_pdf_collector_defaults as get_default_args
from chroma_hearts.config import parse_pdf_collector_args as parse_args


def main(args: Namespace | None = None) -> None:
    """Download random arXiv PDFs into the output directory."""
    if args is None:
        args = parse_args()

    count = max(1, args.count)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Target count: {count}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "random-arxiv-pdf-collector/0.1",
    })

    saved = 0
    tries = 0

    while saved < count:
        tries += 1

        year = random.randint(10, 26)
        month = random.randint(1, 12)
        # Keep the numeric range modest to reduce miss rate.
        number = random.randint(1, 30000)

        arxiv_id = f"{year:02d}{month:02d}.{number:05d}"
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        try:
            response = session.get(url, timeout=20)

            if response.status_code == 200 and response.content[:4] == b"%PDF":
                path = output_dir / f"{arxiv_id}.pdf"
                path.write_bytes(response.content)
                saved += 1
                print(f"[{saved}/{count}] saved {arxiv_id}")
            else:
                print(f"miss {arxiv_id}: {response.status_code}")

        except requests.RequestException as exc:
            print(f"error {arxiv_id}: {exc}")

        # Avoid sending excessive requests to arXiv.
        time.sleep(3)

    print(f"[INFO] Done. saved={saved}, tries={tries}")


if __name__ == "__main__":
    main()
