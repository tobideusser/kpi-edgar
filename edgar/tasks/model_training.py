import json
import logging
import os
from typing import Dict, Union, Optional

from fluidml.common import Task
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
import torch
from edgar.data_classes import Corpus, Labels
from edgar.trainer import Trainer
from edgar.trainer.checkpointer import Checkpointer
from edgar.trainer.evaluator import Evaluator
from edgar.trainer.optimization import Optimizer, LearningRateScheduler
from edgar.trainer.train_logger import TrainLogger
from edgar.trainer.utils import set_device, set_seed_number, get_device, set_seeds
from edgar.annotation_merger import AnnotationMerger
from edgar.data_loading import KPIRelationDataset, BatchCollator
from edgar.models import JointNERAndREModel

logger = logging.getLogger(__name__)


class ModelTraining(Task):

    publishes = ['best_model']

    def __init__(
            self,
            model_params: Dict,
            dataloader_params: Dict,
            optimizer_params: Dict,
            lr_scheduler_params: Dict,
            evaluator_params: Dict,
            checkpointer_params: Dict,
            train_logger_params: Dict,
            trainer_params: Dict,
            seed: int = 42,
            combine_train_valid: bool = False,
            warm_start: bool = False
    ):
        super().__init__()

        # config
        self.model_params = model_params
        self.combine_train_valid = combine_train_valid
        self.dataloader_params = dataloader_params
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.evaluator_params = evaluator_params
        self.checkpointer_params = checkpointer_params
        self.train_logger_params = train_logger_params
        self.trainer_params = trainer_params
        self.warm_start = warm_start
        self.seed = seed
        self.type_ = self.model_params.pop("type_")
        self.path_pretrained_model = self.model_params.pop("path_pretrained_model", None)
        if self.type_ in ["pretrained_decoder", "pretrained_full"] and not self.path_pretrained_model:
            raise AttributeError("If 'type_' is of a pretrained variant, a path to the pretrained model hast to be "
                                 "specified in 'path_pretrained_model'!")

    def _create_torch_datasets(self, corpus: Corpus) -> Dict[str, KPIRelationDataset]:
        if self.combine_train_valid:
            sentences = [sentence for sentence in corpus.sentences if sentence.split_type in ["train", "valid"]]
            logger.debug(f'train valid combined len: {len(sentences)}')
            datasets = {'train': KPIRelationDataset(sentences=sentences)}
        else:
            datasets = {'train': None, 'valid': None}
            for split_type in datasets:
                sentences = [sentence for sentence in corpus.sentences if sentence.split_type == split_type]
                logger.debug(f'{split_type} len: {len(sentences)}')
                datasets[split_type] = KPIRelationDataset(sentences=sentences)
        return datasets

    def _create_torch_dataloaders(self,
                                  datasets: Dict[str, KPIRelationDataset],
                                  batch_collator: BatchCollator) -> Dict[str, DataLoader]:
        dataloaders = {}
        set_seeds()
        for split_type, split_dataset in datasets.items():
            shuffle = True if split_type == 'train' else False
            dataloaders[split_type] = DataLoader(dataset=split_dataset,
                                                 collate_fn=batch_collator,
                                                 shuffle=shuffle,
                                                 **self.dataloader_params)
        return dataloaders

    @staticmethod
    def _convert_token_level_annotations(corpus_tokenized: Corpus, labels: Labels) -> Corpus:
        for sentence in corpus_tokenized.sentences:
            # convert span annotations to from word to token level
            for entity in sentence.entities_anno:
                entity.start = sentence.word2token_start_ids[entity.start] - 1
                entity.end = sentence.word2token_end_ids[entity.end - 1]

            # convert iobes annotations to from word to token level
            entities_anno_iobes = []
            for i, tag in enumerate(sentence.entities_anno_iobes):
                start = sentence.word2token_start_ids[i]
                end = sentence.word2token_end_ids[i] + 1
                num_tokens = end - start

                if tag == 'O' or tag.startswith('I-'):
                    for j in range(num_tokens):
                        entities_anno_iobes.append(tag)

                elif tag.startswith('B-'):
                    for j in range(num_tokens):
                        if j == 0:
                            entities_anno_iobes.append(tag)
                        else:
                            entities_anno_iobes.append(tag.replace('B-', 'I-'))

                elif tag.startswith('E-'):
                    for j in range(num_tokens):
                        if num_tokens > 1 and j != num_tokens - 1:
                            entities_anno_iobes.append(tag.replace('E-', 'I-'))
                        else:
                            entities_anno_iobes.append(tag)

                elif tag.startswith('S-'):
                    for j in range(num_tokens):
                        if num_tokens == 1:
                            entities_anno_iobes.append(tag)
                        elif j == 0:
                            entities_anno_iobes.append(tag.replace('S-', 'B-'))
                        elif 0 < j < num_tokens - 1:
                            entities_anno_iobes.append(tag.replace('S-', 'I-'))
                        else:
                            entities_anno_iobes.append(tag.replace('S-', 'E-'))

            entities_anno_iobes_ids = [labels.iobes.val2idx[tag] for tag in entities_anno_iobes]
            sentence.entities_anno_iobes = entities_anno_iobes
            sentence.entities_anno_iobes_ids = entities_anno_iobes_ids
            sentence.n_words = len(entities_anno_iobes)

        return corpus_tokenized

    def run(self,
            corpus_tokenized: Union[Dict, Corpus],
            labels: Dict,
            sub_word_tokenizer: PreTrainedTokenizerFast):

        set_device(self.resource.device)
        set_seed_number(seed=self.seed)
        set_seeds()

        if isinstance(corpus_tokenized, Dict):
            logger.info('Converting corpus_tokenized dict to Corpus object.')
            corpus = Corpus.from_dict(corpus_tokenized)
        else:
            corpus = corpus_tokenized

        labels = Labels.from_dict(labels)

        corpus = AnnotationMerger.filter_annotated_samples(corpus)

        if not self.model_params['encoder_params']['word_pooling']:
            corpus = self._convert_token_level_annotations(corpus, labels)

        logger.info('Instantiate train and valid torch datasets.')
        datasets = self._create_torch_datasets(corpus)

        batch_collator = BatchCollator(pad_token_id=sub_word_tokenizer.pad_token_id)

        logger.info('Instantiate train and valid torch dataloaders.')
        set_seeds()
        dataloaders = self._create_torch_dataloaders(datasets, batch_collator)

        logger.info('Instantiate encoder+decoder model.')
        set_seeds()
        sub_word_tokenizer.type_ = self.unique_config['SubWordTokenization']['tokenizer_name']
        model = JointNERAndREModel.from_config(tokenizer=sub_word_tokenizer,
                                               labels=labels,
                                               **self.model_params).to(get_device())
        if self.type_ == "pretrained_full":
            path_to_state_dict = os.path.join(self.path_pretrained_model, "models", "best_model.pt")
            model.load_state_dict(torch.load(path_to_state_dict))
        elif self.type_ == "pretrained_decoder":
            # load config
            path_to_pretrained_cfg = os.path.join(self.path_pretrained_model, "config.json")
            with open(path_to_pretrained_cfg) as json_cfg:
                pretrained_cfg = json.load(json_cfg)

            # extract the model params and tokeniser
            pretrained_model_params = pretrained_cfg["ModelTraining"]["model_params"]

            # load tokeniser
            path_to_pretrained_tokenizer = os.path.join(self.path_pretrained_model, "sub_word_tokenizer")
            pretrained_sub_word_tokenizer = PreTrainedTokenizerFast.from_pretrained(path_to_pretrained_tokenizer)
            pretrained_sub_word_tokenizer.type_ = pretrained_cfg["SubWordTokenization"]["tokenizer_name"]

            # init a seperate model
            pretrained_model = JointNERAndREModel.from_config(
                tokenizer=pretrained_sub_word_tokenizer,
                labels=labels,
                **pretrained_model_params
            ).to(get_device())

            model.reinitialise_encoder(tokenizer=sub_word_tokenizer, encoder_params=self.model_params["encoder_params"])

        logger.info('Instantiate optimizer.')
        optimizer = Optimizer.from_config(params=model.parameters(),
                                          **self.optimizer_params)

        logger.info('Instantiate learning rate scheduler.')
        batch_size = self.dataloader_params['batch_size']
        num_epochs = self.trainer_params['num_epochs']
        num_grad_accumulation_steps = self.trainer_params.get('num_grad_accumulation_steps', 1)
        # also for combined train+valid training make sure scheduler is aligned with old training setup
        train_sample_count = sum(1 for sentence in corpus.sentences if sentence.split_type == 'train')
        # train_sample_count = len(datasets['train'])
        updates_epoch = train_sample_count // (batch_size * num_grad_accumulation_steps)
        updates_total = updates_epoch * num_epochs
        lr_warmup = self.lr_scheduler_params.pop('lr_warmup', 0.)
        lr_scheduler = LearningRateScheduler.from_config(optimizer=optimizer,
                                                         num_warmup_steps=lr_warmup * updates_total,
                                                         num_training_steps=updates_total,
                                                         **self.lr_scheduler_params)

        logger.info('Instantiate training logger.')
        run_dir = self.get_store_context()
        log_dir = self.train_logger_params.pop('log_dir', 'logs')
        log_dir = os.path.join(run_dir, log_dir)
        train_logger = TrainLogger.from_config(log_dir=log_dir, **self.train_logger_params)

        logger.info('Instantiate training evaluator.')
        evaluator = Evaluator.from_config(**self.evaluator_params)

        logger.info('Instantiate checkpointer.')
        checkpointer = Checkpointer(task=self, **self.checkpointer_params)

        logger.info('Instantiate Trainer.')
        num_epochs_retrain = self.trainer_params.pop('num_epochs_retrain', None)
        if num_epochs_retrain is not None:
            self.trainer_params['num_epochs'] = num_epochs_retrain
        trainer = Trainer(model=model, optimizer=optimizer, train_dataloader=dataloaders['train'],
                          evaluator=evaluator, valid_dataloader=dataloaders.get('valid'), lr_scheduler=lr_scheduler,
                          train_logger=train_logger, checkpointer=checkpointer, unique_config=self.unique_config,
                          **self.trainer_params)

        logger.info('Start training.')
        trainer.train(warm_start=self.warm_start)
