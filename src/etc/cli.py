"""Command-line interface for ETC."""

import argparse


def main() -> None:
    """Run the ETC command-line interface."""
    parser = argparse.ArgumentParser(
        prog="etc",
        description=(
            "Evaluate topological coverage in metabolic networks."
        ),
    )

    parser.parse_args()


if __name__ == "__main__":
    main()