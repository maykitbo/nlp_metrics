from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import BERTScorer



def BLEU(reference, candidate):
    reference_tokens = [reference.split()]  # BLEU expects a list of reference lists
    candidate_tokens = candidate.split()  # Candidate is a list of tokens
    # Use smoothing function for cases of perfect matches and zero matches
    smoothing_function = SmoothingFunction().method2
    # Calculate BLEU score
    score = corpus_bleu([reference_tokens], [candidate_tokens], smoothing_function=smoothing_function)
    return score


def ROUGE(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]  # get_scores returns a list of scores per sentence. We only have one sentence.


bert_scorer = BERTScorer(model_type='bert-base-uncased')

def BERT_SCORE(reference, candidate):
    P, R, F1 = bert_scorer.score([candidate], [reference])
    return {'P': P.mean().item(), 'R': R.mean().item(), 'F1': F1.mean().item()}













