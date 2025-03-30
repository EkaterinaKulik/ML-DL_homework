from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    numerator = sum(len(set(pred) & (set(ref.possible) | set(ref.sure))) 
                    for pred, ref in zip(predicted, reference))
    denominator = sum(len(pred) for pred in predicted)
    return numerator, denominator



def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    numerator = sum(len(set(pred) & set(ref.sure)) for pred, ref in zip(predicted, reference))
    denominator = sum(len(ref.sure) for ref in reference)
    return numerator, denominator

def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    total_predicted = sum(len(pred) for pred in predicted)
    total_sure = sum(len(ref.sure) for ref in reference)
    intersection_union = sum(len(set(pred) & (set(ref.possible) | set(ref.sure))) 
                             for pred, ref in zip(predicted, reference))
    intersection_sure = sum(len(set(pred) & set(ref.sure))
                            for pred, ref in zip(predicted, reference))
    
    aer = 1 - (intersection_union + intersection_sure) / (total_predicted + total_sure)
    return max(0, min(aer, 1))