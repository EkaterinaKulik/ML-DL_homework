from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple
from collections import Counter

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words+1, num_target_words+1), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        eps = 1e-12
        posteriors = []
        for pair in parallel_corpus:
            src = pair.source_tokens - 1  
            tgt = pair.target_tokens - 1   
            T = self.translation_probs[src][:, tgt]  
            col_sum = T.sum(axis=0, keepdims=True) + eps
            gamma = T / col_sum
            posteriors.append(gamma)
        return posteriors


    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        eps = 1e-12
        total_ll = 0.0
        for pair, _ in zip(parallel_corpus, posteriors):
            src = pair.source_tokens - 1
            tgt = pair.target_tokens - 1
            n = len(src)
            T = self.translation_probs[src][:, tgt].astype(np.float32)
            col_sum = T.sum(axis=0) + eps
            ll = np.sum(np.log(col_sum)) - len(tgt) * np.log(n)
            total_ll += ll
        return total_ll

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        eps = 1e-12
        new_counts = np.zeros_like(self.translation_probs, dtype=np.float32)
        for pair, gamma in zip(parallel_corpus, posteriors):
            src = pair.source_tokens - 1
            tgt = pair.target_tokens - 1
            np.add.at(new_counts, (src[:, None], tgt[None, :]), gamma)
        row_sums = new_counts.sum(axis=1, keepdims=True) + eps
        new_translation_probs = new_counts / row_sums
        self.translation_probs = new_translation_probs.astype(np.float32)
        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):

        alignments = []

        for pair in sentences:
            src = pair.source_tokens - 1
            tgt = pair.target_tokens - 1
            T = self.translation_probs[src][:, tgt]  
            best_src_idx = np.argmax(T, axis=0)
            alignment = [(int(best_src_idx[i]) + 1, i + 1) for i in range(len(tgt))]
            alignments.append(alignment)

        return alignments

    def _compute_word_frequencies(self, parallel_corpus: List[TokenizedSentencePair]):
    
        source_counts = Counter()
        target_counts = Counter()
    
        for sentence_pair in parallel_corpus: 
          source_counts.update(sentence_pair.source_tokens)
          target_counts.update(sentence_pair.target_tokens)


        total_source = sum(source_counts.values()) + 1e-9  
        total_target = sum(target_counts.values()) + 1e-9

        source_probs = {word: count / total_source for word, count in source_counts.items()}
        target_probs = {word: count / total_target for word, count in target_counts.items()}
    
        return source_probs, target_probs

    def _invert_translation_probs(self, parallel_corpus: List[TokenizedSentencePair]):
      
      source_probs, target_probs = self._compute_word_frequencies(parallel_corpus)
    
      inverted_probs = np.zeros((self.num_target_words, self.num_source_words), dtype=np.float32) 

      for s in range(self.num_source_words):

        for t in range(self.num_target_words):
          
          if target_probs.get(t, 0) > 0:
                inverted_probs[t, s] = (self.translation_probs[s, t] * source_probs.get(s, 0)) / (target_probs[t] + 1e-9) 


      row_sums = inverted_probs.sum(axis=1, keepdims=True) + 1e-9  
      inverted_probs /= row_sums
    
      return inverted_probs



class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        if (src_length, tgt_length) not in self.alignment_probs:
          self.alignment_probs[(src_length, tgt_length)] = np.full(
            (src_length, tgt_length), 1 / src_length, dtype=np.float32)
            
        return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
      posteriors = []
      eps = 1e-12

      for sentence_pair in parallel_corpus:
        src_sentence = sentence_pair.source_tokens - 1 
        tgt_sentence = sentence_pair.target_tokens - 1  
        src_len, tgt_len = len(src_sentence), len(tgt_sentence)

        phi = self._get_probs_for_lengths(src_len, tgt_len)
        trans_probs = self.translation_probs[src_sentence, :][:, tgt_sentence] 

        denom = np.sum(phi * trans_probs, axis=0, keepdims=True) + eps
        gamma = (phi * trans_probs) / denom
        posteriors.append(gamma)

      return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
      eps = 1e-15
      elbo = 0.0

      for sentence_pair, gamma in zip(parallel_corpus, posteriors):
        src_sentence = sentence_pair.source_tokens - 1
        tgt_sentence = sentence_pair.target_tokens - 1
        src_len, tgt_len = len(src_sentence), len(tgt_sentence)

        phi = self._get_probs_for_lengths(src_len, tgt_len)  
        trans_probs = self.translation_probs[src_sentence, :][:, tgt_sentence]  

        prob_sum = np.sum(phi * trans_probs, axis=0)  
        prob_sum = np.clip(prob_sum, eps, None)
        elbo += np.sum(np.log(prob_sum) * np.sum(gamma, axis=0)) 
      return elbo

    

    def _m_step(self, parallel_corpus, posteriors):
      eps = 1e-12
      new_translation_probs = np.zeros_like(self.translation_probs)
      alignment_counts = {}

      for sentence_pair, gamma in zip(parallel_corpus, posteriors):
        src_sentence = sentence_pair.source_tokens - 1
        tgt_sentence = sentence_pair.target_tokens - 1
        src_len, tgt_len = len(src_sentence), len(tgt_sentence)

        if (src_len, tgt_len) not in alignment_counts:
            alignment_counts[(src_len, tgt_len)] = np.zeros((src_len, tgt_len), dtype=np.float32)

        for i, tgt_word in enumerate(tgt_sentence):
            for j, src_word in enumerate(src_sentence):
                new_translation_probs[src_word, tgt_word] += gamma[j, i]

        alignment_counts[(src_len, tgt_len)] += gamma

      row_sums = new_translation_probs.sum(axis=1, keepdims=True) + eps
      mask = (row_sums > eps).flatten()
      new_translation_probs[mask, :] /= row_sums[mask, :]  
      self.translation_probs = new_translation_probs

      for (src_len, tgt_len), counts in alignment_counts.items():
        row_sums = counts.sum(axis=0, keepdims=True) + eps
        mask = (row_sums > eps).flatten()  
        counts[:, mask] /= row_sums[:, mask]  
        self.alignment_probs[(src_len, tgt_len)] = counts

      return self._compute_elbo(parallel_corpus, posteriors)
     