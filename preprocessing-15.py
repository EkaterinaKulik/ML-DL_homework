from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import os
from collections import Counter
import re

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]

def safe_parse_xml(filename: str) -> ET.ElementTree:
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    content = content.replace('\ufeff', '')
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', content)
    return ET.ElementTree(ET.fromstring(content))

def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    try:
        tree = safe_parse_xml(filename)
    except ET.ParseError:
        return [], []
    
    root = tree.getroot()
    sentence_pairs = []
    alignments = []
    
    for s in root.findall('s'):
        source_text = (s.findtext('english') or "").strip()
        target_text = (s.findtext('czech') or "").strip()
        sure_text = (s.findtext('sure') or "").strip()
        possible_text = (s.findtext('possible') or "").strip()
        
        sentence_pairs.append(SentencePair(source=source_text.split(), target=target_text.split()))
        
        sure_pairs = [tuple(map(int, pair.split('-'))) for pair in sure_text.split()] if sure_text else []
        possible_pairs = [tuple(map(int, pair.split('-'))) for pair in possible_text.split()] if possible_text else []
        
        alignments.append(LabeledAlignment(sure=sure_pairs, possible=possible_pairs))
    
    return sentence_pairs, alignments

def collect_all_sentences_and_alignments(data_dir: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
  
    all_sentences = []
    all_targets = []

    for root_dir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wa"):  
                file_path = os.path.join(root_dir, file)

                sentence_pairs, alignments = extract_sentences(file_path)

                all_sentences.extend(sentence_pairs)
                all_targets.extend(alignments)

    return all_sentences, all_targets


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counter = Counter()
    target_counter = Counter()
    source_first_occurrence = {}
    target_first_occurrence = {}
    
    for pair in sentence_pairs:
        for token in pair.source:
            source_counter[token] += 1
            if token not in source_first_occurrence:
                source_first_occurrence[token] = len(source_first_occurrence)
        for token in pair.target:
            target_counter[token] += 1
            if token not in target_first_occurrence:
                target_first_occurrence[token] = len(target_first_occurrence)
    
    sorted_source = sorted(source_counter.items(), key=lambda x: (-x[1], source_first_occurrence[x[0]]))
    sorted_target = sorted(target_counter.items(), key=lambda x: (-x[1], target_first_occurrence[x[0]]))
    
    if freq_cutoff is not None:
        sorted_source = sorted_source[:freq_cutoff]
        sorted_target = sorted_target[:freq_cutoff]
    
    source_dict = {token: idx + 1 for idx, (token, _) in enumerate(sorted_source)}
    target_dict = {token: idx + 1 for idx, (token, _) in enumerate(sorted_target)}
    
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    
    for pair in sentence_pairs:
        source_tokens = [source_dict[word] for word in pair.source if word in source_dict]
        target_tokens = [target_dict[word] for word in pair.target if word in target_dict]
        
        if source_tokens and target_tokens:
            tokenized_sentence_pairs.append(
                TokenizedSentencePair(
                    source_tokens=np.array(source_tokens, dtype=np.int32),
                    target_tokens=np.array(target_tokens, dtype=np.int32)
                )
            )
    return tokenized_sentence_pairs
