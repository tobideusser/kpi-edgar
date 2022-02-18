import logging
import os
from typing import Dict, Optional, Union, List, Tuple, Any
import numbers

import torch
from fluidml.common import Task
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from edgar.data_classes import Corpus, Labels, Entity, Relation
from edgar.trainer import Evaluator
from edgar.trainer.utils import set_device, get_device, set_seed_number, set_seeds, detach_batch
from edgar.data_loading import KPIRelationDataset, BatchCollator
from edgar.html_creation import create_html
from edgar.models import JointNERAndREModel
from edgar.trainer.train_logger import TrainLogger
logger = logging.getLogger(__name__)


class ModelPredicting(Task):
    publishes = ['corpus_predicted', 'metrics']

    def __init__(
            self,
            split_types: Optional[Union[str, List]] = None,
            model_params: Optional[Dict] = None,
            evaluator_params: Optional[Dict] = None,
            batch_size: Optional[int] = None,
            seed: Optional[int] = None,
            evaluation: bool = True,
            html_creation: bool = True,
            train_mode: bool = True,
            save_hdf5: bool = False
    ):
        super().__init__()

        self.seed: int = seed
        self.model_params: Dict = model_params
        self.evaluator_params: Dict = evaluator_params
        self.batch_size = batch_size if batch_size is not None else 1
        self.split_types: Optional[List] = [split_types] if isinstance(split_types, str) else split_types

        self.evaluation = evaluation
        self.html_creation = html_creation
        self.train_mode = train_mode
        self.save_hdf5 = save_hdf5

    def _predict_corpus(
            self,
            model: JointNERAndREModel,
            dataloader: DataLoader,
            evaluator: Optional[Evaluator] = None
    ) -> Tuple[Dict, Dict]:

        logger.debug('Predict corpus...')
        corpus_predictions = {}

        # if self.save_hdf5:
        #     current_run_dir = self.get_store_context()
        #     h5_file_path = os.path.join(current_run_dir, 'h5/text_entity_vectors.h5')
        #     f_text_entities = h5py.File(h5_file_path, 'w')
        f_text_entities = self.open(name='text_kpi_embeddings',
                                    task_name=self.name,
                                    task_unique_config=self.unique_config,
                                    mode='w',
                                    sub_dir='h5',
                                    type_='h5',
                                    libver='latest') if self.save_hdf5 else None

        for j, batch in enumerate(tqdm(dataloader, desc="Predict batch")):
            batch_size = batch['token_ids'].shape[0]

            batch_output = model.predict(batch)

            # Detach entity vectors and write to h5 if configured
            for i in range(batch_size):
                # delete relation embeddings to save space of corpus object (not used atm)
                for key in ['relations_pred', 'relations_anno', 'entities_anno']:
                    for ent in batch_output[key][i]:
                        # todo: debug -> probably from different version of the other project!
                        try:
                            ent['embedding'] = ent['embedding'].detach().cpu().numpy()
                            del ent['embedding']
                        except:
                            pass

                for key in ['entities_pred']:
                    for ent in batch_output[key][i]:
                        try:
                            ent['embedding'] = ent['embedding'].detach().cpu().numpy()

                            if f_text_entities:
                                unique_id = f"{batch_output['unique_id'][i]}@{key}@{(ent['start'], ent['end'])}"
                                ent['unique_id'] = unique_id
                                f_text_entities.f.create_dataset(name=unique_id, data=ent['embedding'])
                                del ent['embedding']
                        except:
                            pass

            # Detach tensors and move to cpu
            batch_output = detach_batch(batch_output)
            torch.cuda.empty_cache()

            if self.evaluation:
                evaluator.increment_metrics(batch_output=batch_output)

            for i in range(batch_size):
                corpus_predictions[batch['unique_id'][i]] = {
                    'entities_pred': batch_output['entities_pred'][i],
                    'relations_pred': batch_output['relations_pred'][i],
                    'entities_score': batch_output['ner_score'][i],
                    'relations_score': batch_output['re_scores'][i],
                }

        # f_text_entities.close()

        metrics = None
        if self.evaluation:
            metrics = evaluator.get_metrics(reset=True)

        return corpus_predictions, metrics

    @staticmethod
    def _write_predictions_to_corpus(corpus: Corpus, corpus_predictions: Dict) -> Corpus:
        logger.debug('Write predictions in corpus...')
        for sentence in tqdm(corpus.sentences, desc='Write predictions'):
            sentence_preds = corpus_predictions.get(sentence.unique_id)
            if sentence_preds:
                # convert predicted entities in correct format
                # sentence.entities_pred = [
                #     Entity.from_dict({'start': start,
                #                       'end': end,
                #                       'type_': type_,
                #                       'score': score})
                #     for score, ((start, end), type_) in zip(sentence_preds['entities_score'],
                #                                             sentence_preds['entities_pred'].items())
                # ]
                sentence.entities_pred = [
                    Entity.from_dict(sentence_pred) for sentence_pred in sentence_preds['entities_pred']
                ]

                # convert predicted relations in correct format
                # entity_spans = [ent.span for ent in sentence.entities_pred]
                # for score, rel_pred in zip(sentence_preds['relations_score'], sentence_preds['relations_pred']):
                #     head_span = rel_pred.pop('head')
                #     tail_span = rel_pred.pop('tail')
                #     rel_pred['head_idx'] = entity_spans.index(head_span)
                #     rel_pred['tail_idx'] = entity_spans.index(tail_span)
                #     del rel_pred['head_type']
                #     del rel_pred['tail_type']
                #     rel_pred['score'] = score
                sentence.relations_pred = [Relation.from_dict(rel) for rel in sentence_preds['relations_pred']]

        return corpus

    @staticmethod
    def _register_entity_values_and_relation_entities(corpus_predicted: Corpus) -> Corpus:
        for sent in corpus_predicted.sentences:
            if sent.entities_anno:
                for ent in sent.entities_anno:
                    ent.get_value(sent.words)
            if sent.relations_anno:
                for rel in sent.relations_anno:
                    rel.get_entities(sent.entities_anno)
            if sent.entities_pred:
                for ent in sent.entities_pred:
                    ent.get_value(sent.words)
            if sent.relations_pred:
                for rel in sent.relations_pred:
                    rel.get_entities(sent.entities_pred)
        return corpus_predicted

    @staticmethod
    def _clf_report_dict_to_str(clf_report: Dict) -> str:
        '''Got function from train_logger class'''
        headers = ('Type', 'Precision', 'Recall', 'F1', 'Support')
        digits = 2
        max_name_width = max(len(name) for name in clf_report.keys())
        head_fmt = '    {:>{max_name_width}s} ' + ' {:>9}' * (len(headers) - 1)
        report = '    \n'
        report += head_fmt.format(*headers, max_name_width=max_name_width)
        report += '    \n\n'
        row_fmt = '    {:>{max_name_width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
        for name, metrics in clf_report.items():
            if name == 'micro avg':
                report += '    \n'
            row = (name,) + tuple(metrics.values())
            report += row_fmt.format(*row, max_name_width=max_name_width, digits=digits)
        return report

    def _metrics_to_str(self, metrics: Dict[str, Any]) -> str:
        log_message = '\n'
        max_name_length = max(len(name) for name in metrics.keys())

        for name, value in metrics.items():
            if 'clf_report' in name:
                value = self._clf_report_dict_to_str(clf_report=value)
            if isinstance(value, numbers.Number):
                value = round(value, 4)
            log_message += f'{name:{max_name_length}} {value}\n'
        return log_message

    def run(self,
            best_model: Dict,
            corpus_tokenized: Union[Dict, Corpus],
            labels: Dict,
            sub_word_tokenizer: PreTrainedTokenizerFast):

        if "ModelTraining" in self.unique_config:
            key = "ModelTraining"
        elif "ModelSelfTraining" in self.unique_config:
            key = "ModelSelfTraining"
        else:
            key = None

        if key:
            self.seed = self.unique_config[key].get("seed", 42) if self.seed is None else self.seed
            self.model_params = self.unique_config[key].get(
                "model_params") if self.model_params is None else self.model_params
            self.evaluator_params = self.unique_config[key].get(
                "evaluator_params") if self.evaluator_params is None else self.evaluator_params

        # try:
        #     self.seed = self.seed if self.seed is not None else self.unique_config["ModelTraining"]["seed"]
        #     self.model_params = self.model_params if self.model_params is not None \
        #         else self.unique_config["ModelTraining"]["model_params"]
        #     self.evaluator_params = self.evaluator_params if self.evaluator_params is not None \
        #         else self.unique_config["ModelTraining"]["evaluator_params"]
        #
        # except KeyError:
        #     self.seed = self.seed if self.seed is not None else self.unique_config["ModelSelfTraining"]["seed"]
        #     self.model_params = self.model_params if self.model_params is not None \
        #         else self.unique_config["ModelSelfTraining"]["model_params"]
        #     self.evaluator_params = self.evaluator_params if self.evaluator_params is not None \
        #         else self.unique_config["ModelSelfTraining"]["evaluator_params"]

        set_device(self.resource.device)
        set_seed_number(seed=self.seed)
        set_seeds()

        if isinstance(corpus_tokenized, Dict):
            logger.info('Converting corpus_tokenized dict to Corpus object.')
            corpus = Corpus.from_dict(corpus_tokenized)
        else:
            corpus = corpus_tokenized

        labels = Labels.from_dict(labels)

        # load model architecture / untrained model
        logger.debug('Instantiate torch model and load state dict.')
        sub_word_tokenizer.type_ = self.unique_config['SubWordTokenization']['tokenizer_name']
        model = JointNERAndREModel.from_config(tokenizer=sub_word_tokenizer,
                                               labels=labels,
                                               **self.model_params).to(get_device())
        # load model state dict into model and set model in validation mode
        model.load_state_dict(best_model)
        model.eval()

        logger.debug('Instantiate torch dataset.')
        if self.split_types:
            sentences = [sentence for sentence in corpus.sentences
                         if sentence.split_type in self.split_types
                         and len(sentence.token_ids) <= model.encoder.encoder.config.max_position_embeddings]
        else:
            sentences = [sentence for sentence in corpus.sentences
                         if len(sentence.token_ids) <= model.encoder.encoder.config.max_position_embeddings]
        # TODO change back
        dataset = KPIRelationDataset(sentences=sentences)
        batch_collator = BatchCollator(pad_token_id=sub_word_tokenizer.pad_token_id)

        logger.debug('Instantiate torch dataloader.')
        dataloader = DataLoader(dataset=dataset,
                                collate_fn=batch_collator,
                                shuffle=False,
                                batch_size=self.batch_size,
                                drop_last=False)

        evaluator = None
        if self.evaluation:
            logger.debug('Instantiate evaluator.')
            evaluator = Evaluator.from_config(**self.evaluator_params)

        # predict corpus
        corpus_predictions, metrics = self._predict_corpus(model, dataloader, evaluator)
        result_print = self._metrics_to_str(metrics)
        print(result_print)
        # writing result to file
        with open("/scratch/data/edgar/above200B/" + sub_word_tokenizer.type_, 'w') as out:
            for line in result_print.split("\n"):
                out.write(line + "\n")
        # # Write prediction result to corpus
        # corpus = self._write_predictions_to_corpus(corpus, corpus_predictions)
        #
        # # Register entity values and relation entity objects in entities and relations (annotated and predicted)
        # corpus = self._register_entity_values_and_relation_entities(corpus)
        #
        # if self.html_creation:
        #     logger.info('Create html evaluation.')
        #     current_run_dir = self.get_store_context()
        #     html_report_dir = os.path.join(current_run_dir, 'html')
        #     os.makedirs(html_report_dir, exist_ok=True)
        #
        #     if self.split_types:
        #         sentences = [sentence for sentence in corpus.sentences if sentence.split_type in self.split_types]
        #     else:
        #         sentences = [sentence for sentence in corpus.sentences]
        #     create_html(sentences, out_dir=os.path.join(current_run_dir, 'html'))
        #
        # if self.train_mode:
        #     self.save(corpus.to_dict(), 'corpus_predicted', type_='pickle')
        #     if metrics:
        #         self.save(metrics, 'metrics', type_='json')
        # else:
        #     return corpus
