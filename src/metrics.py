def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)


def union_ranges(ranges):
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = [sorted_ranges[0]]

    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        if current_start <= last_end:
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            merged_ranges.append((current_start, current_end))

    return merged_ranges


def intersect_two_ranges(range1, range2):
    start1, end1 = range1
    start2, end2 = range2

    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)

    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None


def difference(ranges, target):
    result = []
    target_start, target_end = target

    for start, end in ranges:
        if end < target_start or start > target_end:
            result.append((start, end))
        elif start < target_start and end > target_end:
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            result.append((start, target_start))
        elif end > target_end:
            result.append((target_end, end))

    return result


def find_target_in_document(document, target):
    start_index = document.find(target)
    if start_index == -1:
        return None
    end_index = start_index + len(target)
    return start_index, end_index


def calculate_metrics(cls, retrieved_chunks, golden_references, corpus):
    reference_ranges = [
        (int(ref["start_index"]), int(ref["end_index"])) for ref in golden_references
    ]

    unused_highlights = reference_ranges.copy()

    numerator_sets = []
    chunk_ranges = []

    for chunk in retrieved_chunks:
        chunk_text = chunk["text"]

        chunk_range = cls.find_target_in_document(corpus, chunk_text)
        if chunk_range is None:
            continue

        chunk_start, chunk_end = chunk_range
        chunk_ranges = cls.union_ranges([(chunk_start, chunk_end)] + chunk_ranges)

        for ref_start, ref_end in reference_ranges:
            intersection = cls.intersect_two_ranges(
                (chunk_start, chunk_end), (ref_start, ref_end)
            )
            if intersection is not None:
                unused_highlights = cls.difference(unused_highlights, intersection)
                numerator_sets = cls.union_ranges([intersection] + numerator_sets)

    numerator_val = cls.sum_of_ranges(numerator_sets)
    recall_denominator = cls.sum_of_ranges(reference_ranges)
    precision_denominator = cls.sum_of_ranges(chunk_ranges)

    recall = numerator_val / recall_denominator if recall_denominator > 0 else 0.0
    precision = (
        numerator_val / precision_denominator if precision_denominator > 0 else 0.0
    )
    return {"precision": precision, "recall": recall}
