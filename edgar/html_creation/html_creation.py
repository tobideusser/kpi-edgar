import os
import pickle
from typing import List, Dict, Union, Callable, Tuple

import jinja2
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs
from tqdm import tqdm

from edgar import package_path
from edgar.data_classes import Corpus, Sentence, Relation, Entity


def _store_examples(examples: List[Dict], file_path: str, template: str):
    template_path = os.path.join(package_path, 'kpi_relation_extraction', 'html_creation',
                                 'html_result_templates', template)

    # read template
    with open(template_path) as f:
        template = jinja2.Template(f.read())

    # write to disc
    template.stream(examples=examples).dump(file_path)


def _compute_metrics(gt_all: List[str], pred_all: List[str]):

    labels = list(set(gt_all) | set(pred_all))

    micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
    micro = [m * 100 for m in micro]
    macro = [m * 100 for m in macro]

    clf_report: Dict = classification_report(gt_all,
                                             pred_all,
                                             output_dict=True,
                                             zero_division=0)

    return micro, macro, clf_report


def _score(
        anno: List[Tuple],
        pred: List[Tuple]
):

    gt_flat = []
    pred_flat = []
    union = set(anno) | set(pred)

    for elem in union:
        if elem in anno:
            type_ = elem[-1]
            gt_flat.append(type_)
        else:
            gt_flat.append("0")

        if elem in pred:
            type_ = elem[-1]
            pred_flat.append(type_)
        else:
            pred_flat.append("0")

    return _compute_metrics(gt_flat, pred_flat)


def _relation_to_html(relation: Tuple, sentence: List[str]) -> str:
    head_start, head_end = relation[0]
    head_type = relation[1]
    tail_start, tail_end = relation[2]
    tail_type = relation[3]

    head_tag = f' <span class="head"><span class="type">{head_type}</span>'
    tail_tag = f' <span class="tail"><span class="type">{tail_type}</span>'

    if head_start < tail_start:
        e1_start, e2_start = head_start, tail_start
        e1_end, e2_end = head_end, tail_end
        e1_tag, e2_tag = head_tag, tail_tag
    else:
        e1_start, e2_start = tail_start, head_start,
        e1_end, e2_end = tail_end, head_end,
        e1_tag, e2_tag = tail_tag, head_tag,

    ctx_before = " ".join(sentence[:e1_start])
    ent1 = " ".join(sentence[e1_start:e1_end])
    ctx_between = " ".join(sentence[e1_end:e2_start])
    ent2 = " ".join(sentence[e2_start:e2_end])
    ctx_after = " ".join(sentence[e2_end:])

    html = f'{ctx_before}{e1_tag}{ent1}</span> {ctx_between}{e2_tag}{ent2}</span> {ctx_after}'
    return html


def _entity_to_html(entity: Tuple, sentence: List[str]) -> str:
    span, type_ = entity
    start, end = span

    # todo: use tokens and word2token_alignment_mask
    context_before = ' '.join(sentence[:start])
    ent = ' '.join(sentence[start:end])
    context_after = ' '.join(sentence[end:])

    html = f'{context_before} <span class="entity"><span class="type">{type_}</span>{ent}</span> {context_after}'
    return html


def _convert_example(
    sentence: Sentence,
    anno: List[Union[Entity, Relation, None]],
    pred: List[Union[Entity, Relation, None]],
    to_html: Callable,
    type_: str
):
    sentence = [word.value for word in sentence]
    anno = anno if anno is not None else []
    pred = pred if pred is not None else []

    if type_ == 'entity':
        anno_t = [(elem_anno.span, elem_anno.type_) for elem_anno in anno]
        pred_t = [(elem_pred.span, elem_pred.type_) for elem_pred in pred]
    elif type_ == 'relation':
        anno_t = [(elem_anno.head_entity.span, elem_anno.head_entity.type_,
                   elem_anno.tail_entity.span, elem_anno.tail_entity.type_,
                   elem_anno.type_) for elem_anno in anno]
        pred_t = [(elem_pred.head_entity.span, elem_pred.head_entity.type_,
                   elem_pred.tail_entity.span, elem_pred.tail_entity.type_,
                   elem_pred.type_) for elem_pred in pred]
    else:
        raise NotImplementedError

    # get micro precision/recall/f1 scores
    if anno or pred:
        precision, recall, f1 = _score(anno_t, pred_t)[0]
    else:
        # corner case: no ground truth and no predictions
        precision, recall, f1 = [100] * 3

    union = set(anno_t) | set(pred_t)
    scores = [elem_pred.score for elem_pred in pred]

    # true positives
    tp = []
    # false negatives
    fn = []
    # false positives
    fp = []

    for elem in union:
        type_verbose = elem[-1]

        if elem in anno_t:
            if elem in pred_t:
                score = scores[pred_t.index(elem)]
                tp.append((to_html(elem, sentence), type_verbose, score))
            else:
                fn.append((to_html(elem, sentence), type_verbose, -1))
        else:
            score = scores[pred_t.index(elem)]
            fp.append((to_html(elem, sentence), type_verbose, score))

    # sort true and false positives by score (probability)
    tp = sorted(tp, key=lambda p: p[-1], reverse=True)
    fp = sorted(fp, key=lambda p: p[-1], reverse=True)

    return {'text': " ".join(sentence),
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'length': len(sentence)}


def create_html(sentences: List[Sentence], out_dir: str):
    # convert entities to html
    entity_examples = [_convert_example(sentence=sentence, anno=sentence.entities_anno, pred=sentence.entities_pred,
                                        to_html=_entity_to_html, type_='entity')
                       for sentence in tqdm(sentences, desc='Converting entities')]

    # convert relations with entity types to html
    rel_examples = [_convert_example(sentence=sentence, anno=sentence.relations_anno, pred=sentence.relations_pred,
                                     to_html=_relation_to_html, type_='relation')
                    for sentence in tqdm(sentences, desc='Converting relations')]

    # store entities
    _store_examples(entity_examples,
                    file_path=os.path.join(out_dir, f'entities.html'),
                    template='entity_examples.html')

    _store_examples(sorted(entity_examples, key=lambda k: k['length']),
                    file_path=os.path.join(out_dir, f'entities_sorted.html'),
                    template='entity_examples.html')

    # with relations
    _store_examples(rel_examples,
                    file_path=os.path.join(out_dir, f'relations.html'),
                    template='relation_examples.html')

    _store_examples(sorted(rel_examples, key=lambda k: k['length']),
                    file_path=os.path.join(out_dir, f'relations_sorted.html'),
                    template='relation_examples.html')


def main():
    predicted_corpus_dir = "/scratch/data/ali/kpi_relation_extractor/banz/experiments/" \
                           "consistency_check/ModelTraining/004"
    predicted_corpus_path = os.path.join(predicted_corpus_dir, 'corpus_predicted_with_scores_new.p')
    corpus = pickle.load(open(predicted_corpus_path, 'rb'))

    corpus = Corpus.from_dict(corpus)

    sentences = [sentence for sentence in corpus.sentences if sentence.split_type == 'valid']

    create_html(sentences, out_dir=predicted_corpus_dir)


if __name__ == '__main__':
    main()
