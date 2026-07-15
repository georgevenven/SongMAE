def macro_fer_breakdown(labels, confusion):
    """Split macro FER into silence-boundary and syllable-identity errors."""
    assert len(labels) == len(confusion)
    rates = []
    for true_label, row in zip(labels, confusion):
        assert len(row) == len(labels)
        frames = int(sum(row))
        if not frames:
            continue
        errors = [
            (pred_label, int(count))
            for pred_label, count in zip(labels, row)
            if pred_label != true_label
        ]
        parsing = sum(
            count for pred_label, count in errors if (true_label == 0) != (pred_label == 0)
        )
        identity = sum(count for pred_label, count in errors if true_label != 0 and pred_label != 0)
        assert sum(count for _, count in errors) == parsing + identity
        rates.append((parsing / frames, identity / frames))

    assert rates
    parsing = sum(rate[0] for rate in rates) / len(rates)
    identity = sum(rate[1] for rate in rates) / len(rates)
    return {
        "macro_fer": parsing + identity,
        "macro_parsing_error": parsing,
        "macro_identity_error": identity,
    }
