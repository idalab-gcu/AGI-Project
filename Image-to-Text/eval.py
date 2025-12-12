import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

def compute_metrics(ref_text, hyp_text):
    # Tokenize
    try:
        ref_tok = nltk.word_tokenize(ref_text.lower())
        hyp_tok = nltk.word_tokenize(hyp_text.lower())
    except:
        ref_tok = ref_text.split()
        hyp_tok = hyp_text.split()

    smoothie = SmoothingFunction().method1
    
    scores = {
        "BLEU-1": sentence_bleu([ref_tok], hyp_tok, weights=(1,0,0,0), smoothing_function=smoothie),
        "BLEU-4": sentence_bleu([ref_tok], hyp_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie),
        "METEOR": meteor_score([ref_tok], hyp_tok)
    }
    
    if rouge_scorer:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores["ROUGE-L"] = scorer.score(ref_text, hyp_text)['rougeL'].fmeasure
    else:
        scores["ROUGE-L"] = 0.0
        
    return scores
