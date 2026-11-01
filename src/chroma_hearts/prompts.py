"""Interactive CLI prompt helpers."""

from pathlib import Path


def prompt_path(label: str, default: Path) -> Path:
    """Prompt for a path; empty input keeps the default."""
    value = input(f"{label} [{default}]: ").strip()
    return Path(value).expanduser().resolve() if value else default


def prompt_int(label: str, default: int) -> int:
    """Prompt for an integer; empty input keeps the default."""
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("[ERROR] Please enter an integer.")


def prompt_menu() -> str:
    """Show the main menu and return the selected option."""
    print("\nChroma Hearts")
    print("0. Collect Sample PDFs")
    print("1. Sample Chroma DB Generate")
    print("2. Recover Chroma DB")
    print("3. Evaluate Recovery Rate")
    print("4. Exit")

    while True:
        choice = input("\nSelect an option: ").strip()
        if choice in {"0", "1", "2", "3", "4"}:
            return choice
        print("[ERROR] Please enter 0, 1, 2, 3, or 4.")
