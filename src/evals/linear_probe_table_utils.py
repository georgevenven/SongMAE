import json
from pathlib import Path

from src.evals.syllable_metrics import macro_fer_breakdown


SPECIES = [("canary", "Canary"), ("zf", "Zebra"), ("bf", "Bengalese")]


def load_capped_runs(root):
    runs = {}
    for path in sorted(Path(root).glob("*/*/*/cap_*/metrics.json")):
        species, _, model, cap = path.parts[-5:-1]
        data = json.loads(path.read_text())
        rates = macro_fer_breakdown(data["class_labels"], data["confusion_matrix"])
        assert abs(rates["macro_fer"] - data["macro_fer"]) < 1e-12, path
        runs.setdefault((species, model, int(cap.removeprefix("cap_"))), []).append(rates)
    return runs


def average(values):
    if not values:
        return None
    return {key: sum(value[key] for value in values) / len(values) for key in values[0]}


def value(runs, species, model, cap):
    if species:
        return average(runs.get((species, model, cap), []))
    values = [average(runs.get((key, model, cap), [])) for key, _ in SPECIES]
    return average([row for row in values if row is not None])


def format_cell(row):
    if row is None:
        return "-"
    return (
        f'{100 * row["macro_fer"]:.2f} '
        f'({100 * row["macro_parsing_error"]:.2f}/{100 * row["macro_identity_error"]:.2f})'
    )


def print_tables(title, columns, rows, cell, markdown):
    separator = " | " if markdown else "\t"
    for species, label in SPECIES + [(None, "Mean across species")]:
        print(f"{title} - {label}")
        headers = ["Model"] + [column_label for column_label, _ in columns]
        if markdown:
            print("| " + separator.join(headers) + " |")
            print("| " + separator.join(["---"] * len(headers)) + " |")
        else:
            print(separator.join(headers))
        for row_label, row_key in rows:
            cells = [row_label] + [
                format_cell(cell(species, row_key, column)) for _, column in columns
            ]
            print(("| " + separator.join(cells) + " |") if markdown else separator.join(cells))
        print()
