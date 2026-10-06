import functools
import itertools
import json
import math
import os
import re
import shutil
import tempfile
import typing
import urllib
import zipfile
from typing import Optional

import datasets
from filelock import FileLock
import fsspec
from huggingface_hub.utils import HfHubHTTPError
import numpy as np
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string

def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class SyntheticTokenizer(
  transformers.PreTrainedTokenizer):
  
  def __init__(
    self,
    vocab_size,
    bos_token="[BOS]",
    eos_token="[EOS]",
    sep_token=None,
    cls_token=None,
    pad_token=None,
    mask_token=None,
    unk_token=None,
    **kwargs):
    
    self.tokens = []
    
    for i in range (vocab_size - 2):
      # appending space for readability
      self.tokens.append(str(i) + " ")
    
    self._vocab_str_to_int = {
      '[BOS]': vocab_size - 2,
      '[EOS]': vocab_size - 1,
      ** {ch: i for i, ch in enumerate(self.tokens)}}
    
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def _generate_synthetic_data(dataset_size, 
                             seq_len, vocab_size):
  dataset = np.zeros((dataset_size, seq_len), dtype=int)
  # tokens representing sequence boundary
  dataset[:, 0] = vocab_size - 2  # bos
  dataset[:, -1] = vocab_size - 1  # eos

  for i in range(dataset_size):
    # sample from 0, 1, ..., vocab_size - 3
    temp = np.random.randint(vocab_size - 2)
    for j in reversed(range(1, seq_len - 1)):
      dataset[i, j] = temp
      if temp != 0:
        temp = temp // 4
      else:
        temp = np.random.randint(vocab_size - 2)

  return dataset


def generate_synthetic_dataset(train_dataset_size, 
                               validation_dataset_size, 
                               seq_len, vocab_size):
  np.random.seed(42)
  train_data = torch.from_numpy(
    _generate_synthetic_data(train_dataset_size, 
                             seq_len, vocab_size))
  train_dataset = datasets.Dataset.from_dict({
    'input_ids': train_data, 
    'attention_mask': torch.ones_like(train_data),
  })
  train_dataset.set_format(type='torch')

  np.random.seed(41)
  validation_data = torch.from_numpy(
    _generate_synthetic_data(validation_dataset_size, 
                             seq_len, vocab_size))
  validation_dataset = datasets.Dataset.from_dict({
    'input_ids': validation_data, 
    'attention_mask': torch.ones_like(validation_data),
  })
  validation_dataset.set_format(type='torch')

  return {
    'train': train_dataset,
    'validation': validation_dataset,
  }


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def _download_preprocessed(repo, cache_dir, **kwargs):
  try:
    return datasets.load_dataset(repo, cache_dir=cache_dir, **kwargs)
  except (ConnectionError, FileNotFoundError, requests.exceptions.RequestException,
          HfHubHTTPError) as error:
    # Datasets 2.15 reports missing Hub repositories as plain FileNotFoundError;
    # filesystem errors instead carry errno or a filename and must propagate.
    if isinstance(error, FileNotFoundError) and (
        error.errno is not None or error.filename is not None):
      raise
    LOGGER.warning('Prepared dataset %s is unavailable (%s); '
                   'falling back to raw-data preprocessing.', repo, error)
    return None


def _load_preprocessed_tiny_gsm(config, save_dir):
  """Cache the published TinyGSM splits when their preprocessing matches."""
  repo = getattr(config.data, 'preprocessed_repo', None)
  if not repo or getattr(config.data, 'data_path', None):
    return None
  expected = {
    'train': 'tiny_gsm',
    'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M',
    'separator': r'\n',
    'wrap': False,
    'train_on_prompt': False,
    'train_on_pad': True,
    'filter_too_long': True,
    'val_ratio': 0.01,
    'val_seed': 42,
  }
  if (config.model.length != 512
      or any(getattr(config.data, key, None) != value
             for key, value in expected.items())):
    LOGGER.info('TinyGSM settings differ from the prepared dataset; '
                'using raw-data preprocessing.')
    return None
  # Atomic directory replacement below requires a local filesystem.
  if '://' in save_dir:
    return None

  split_names = ('train', 'validation')
  parent = os.path.dirname(save_dir)
  os.makedirs(parent, exist_ok=True)
  with FileLock(save_dir + '.lock'):
    # Another process may have populated the cache while we waited.
    if all(utils.fsspec_exists(os.path.join(save_dir, split))
           for split in split_names):
      return datasets.load_from_disk(save_dir)
    if os.path.exists(save_dir):
      raise FileExistsError(
        f'Incomplete TinyGSM cache at {save_dir}. Move it aside or remove '
        'it before retrying the download.')

    LOGGER.info(f'Downloading prepared TinyGSM from {repo} to {save_dir}.')
    dataset = _download_preprocessed(repo, config.data.cache_dir)
    if dataset is None:
      return None
    required_columns = {'input_ids', 'prompt_len', 'attention_mask'}
    for split in split_names:
      if split not in dataset or not len(dataset[split]):
        raise ValueError(f'{repo} must contain nonempty {split} data.')
      if not required_columns.issubset(dataset[split].column_names):
        raise ValueError(
          f'{repo}/{split} must contain {sorted(required_columns)}.')
      # Check the format without scanning or transforming the full dataset.
      for index in {0, len(dataset[split]) - 1}:
        row = dataset[split][index]
        prompt_len = row['prompt_len']
        if (len(row['input_ids']) != 512
            or not 0 < prompt_len < 512
            or row['attention_mask'] != [0] * prompt_len + [1] * (512 - prompt_len)):
          raise ValueError(
            f'{repo}/{split} row {index} does not match the prepared '
            '512-token answer-only, train-on-padding format.')

    # Publish only a complete save_to_disk cache, preserving tokens, masks,
    # split membership, and row order from the uploaded dataset.
    with tempfile.TemporaryDirectory(prefix='.tinygsm-', dir=parent) as temp_dir:
      staging = os.path.join(temp_dir, 'dataset')
      dataset.save_to_disk(staging)
      os.replace(staging, save_dir)
    # Workers must reference the final Arrow paths, not a temporary directory.
    return datasets.load_from_disk(save_dir)


def get_tiny_gsm_dataset(config, tokenizer):
  """Reuse cached/preprocessed TinyGSM, or tokenize and split raw examples.

  Examples are encoded as:
    [BOS] question separator answer [EOS]

  The attention mask controls which tokens contribute to the loss. For IDLM
  distillation on TinyGSM we train on answer tokens and padding, but not on the
  prompt tokens.
  """
  mask_tag = 'full' if config.data.train_on_prompt else 'answer_only'
  filter_tag = '_filtered' if config.data.filter_too_long else ''
  wrap_tag = '_wrapped' if config.data.wrap else ''
  pad_tag = '_train_on_pad' if config.data.train_on_pad else ''
  tokenizer_tag = config.data.tokenizer_name_or_path.replace('/', '__')
  dataset_tag = config.data.train
  save_dir = (f'{config.data.cache_dir}/{dataset_tag}_bs{config.model.length}'
              f'_{mask_tag}{filter_tag}{wrap_tag}{pad_tag}_{tokenizer_tag}')
  split_names = ['train', 'validation']

  if all(utils.fsspec_exists(os.path.join(save_dir, split))
         for split in split_names):
    return datasets.load_from_disk(save_dir)

  # Reuse prepared caches from the older TinyGSM code.
  separator_tag = config.data.separator.encode('utf-8').hex()
  tagged_save_dir = f'{save_dir}_sep{separator_tag}'
  if all(utils.fsspec_exists(os.path.join(tagged_save_dir, split))
         for split in split_names):
    return datasets.load_from_disk(tagged_save_dir)

  prepared = _load_preprocessed_tiny_gsm(config, save_dir)
  if prepared is not None:
    return prepared

  data_path = getattr(config.data, 'data_path', None)
  if data_path:
    LOGGER.info(f'Preparing TinyGSM-style dataset from {data_path}.')
    with open(data_path) as f:
      records = json.load(f)
    normalized_records = []
    for record in records:
      question = record.get('question') or record.get('prompt')
      answer = (record.get('code')
                or record.get('response_ground_truth')
                or record.get('answer'))
      if question is None or answer is None:
        raise ValueError(
          'TinyGSM-style local records require question/code, '
          'prompt/response_ground_truth, or question/answer fields.')
      normalized_records.append({'question': question, 'code': answer})
    ds = datasets.Dataset.from_list(normalized_records)
  else:
    LOGGER.info('Preparing TinyGSM dataset.')
    ds = datasets.load_dataset(
      'TinyGSM/TinyGSM', split='train',
      cache_dir=config.data.cache_dir)

  eos = tokenizer.eos_token_id
  bos = tokenizer.bos_token_id
  pad = tokenizer.pad_token_id
  sep_ids = tokenizer(
    config.data.separator, add_special_tokens=False).input_ids
  block_size = config.model.length
  train_on_prompt = config.data.train_on_prompt

  def tokenize_qa(example):
    q_ids = tokenizer(
      example['question'].strip(),
      add_special_tokens=False).input_ids
    a_ids = tokenizer(
      example['code'].strip(),
      add_special_tokens=False).input_ids
    ids = [bos] + q_ids + sep_ids + a_ids + [eos]
    prompt_len = 1 + len(q_ids) + len(sep_ids)
    return {'input_ids': ids, 'prompt_len': prompt_len}

  tokenized = ds.map(
    tokenize_qa,
    num_proc=config.loader.num_workers or None,
    remove_columns=ds.column_names,
    desc='Tokenizing TinyGSM')

  if config.data.filter_too_long:
    if config.data.wrap:
      raise ValueError('TinyGSM filter_too_long requires wrap=False.')
    before = len(tokenized)
    tokenized = tokenized.filter(
      lambda x: len(x['input_ids']) <= block_size,
      num_proc=config.loader.num_workers or None,
      desc='Filtering too-long TinyGSM examples')
    LOGGER.info(
      f'Filtered TinyGSM: {before} -> {len(tokenized)} '
      f'({before - len(tokenized)} removed)')

  if config.data.wrap:
    tokenized = tokenized.remove_columns('prompt_len')

    def wrap_batch(examples):
      all_ids = list(itertools.chain.from_iterable(examples['input_ids']))
      total = (len(all_ids) // block_size) * block_size
      chunks = [all_ids[i:i + block_size]
                for i in range(0, total, block_size)]
      masks = [[1] * block_size] * len(chunks)
      return {'input_ids': chunks, 'attention_mask': masks}

    tokenized = tokenized.map(
      wrap_batch,
      batched=True,
      batch_size=1000,
      num_proc=config.loader.num_workers or None,
      remove_columns=tokenized.column_names,
      desc='Wrapping TinyGSM')
  else:
    def pad_and_mask(example):
      ids = example['input_ids']
      n = len(ids)
      prompt_len = example['prompt_len']
      if n >= block_size:
        ids = ids[:block_size - 1] + [eos]
      else:
        ids = ids + [pad] * (block_size - n)
      mask_start = 0 if train_on_prompt else min(prompt_len, block_size)
      mask_end = block_size if config.data.train_on_pad else min(n, block_size)
      mask = ([0] * mask_start
              + [1] * (mask_end - mask_start)
              + [0] * (block_size - mask_end))
      return {'input_ids': ids, 'attention_mask': mask}

    tokenized = tokenized.map(
      pad_and_mask,
      num_proc=config.loader.num_workers or None,
      desc='Padding TinyGSM')

  tmp = tokenized.train_test_split(
    test_size=config.data.val_ratio,
    seed=config.data.val_seed)
  dataset = datasets.DatasetDict({
    'train': tmp['train'],
    'validation': tmp['test']})
  dataset.save_to_disk(save_dir)
  return dataset


def get_gsm8k_test_dataset(config, tokenizer):
  """Load and tokenize the local GSM8K/TinyGSM test set for conditional eval."""
  tokenizer_tag = config.data.tokenizer_name_or_path.replace('/', '__')
  save_dir = f'{config.data.cache_dir}/gsm8k_test_{tokenizer_tag}_with_text'

  if utils.fsspec_exists(save_dir):
    LOGGER.info(f'Loading GSM8K test from cache: {save_dir}')
    return datasets.load_from_disk(save_dir)

  LOGGER.info(f'Preparing GSM8K test dataset from {config.data.data_path}')
  with open(config.data.data_path) as f:
    records = json.load(f)

  bos = tokenizer.bos_token_id
  sep_ids = tokenizer(
    config.data.separator, add_special_tokens=False).input_ids

  def tokenize_example(example):
    q_ids = tokenizer(
      example['prompt'].strip(), add_special_tokens=False).input_ids
    a_ids = tokenizer(
      example['response_ground_truth'].strip(),
      add_special_tokens=False).input_ids
    prompt = [bos] + q_ids + sep_ids
    return {'input_ids': prompt, 'answer': a_ids}

  dataset = datasets.Dataset.from_list(records).map(
    tokenize_example,
    desc='Tokenizing GSM8K test')
  dataset.save_to_disk(save_dir)
  return dataset


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def _load_preprocessed_owt(config, dataset_name, mode, save_dir, cache_dir,
                         wrap, insert_eos, block_size, streaming, revision):
  """Reuse the published GPT-2 blocks without changing their split or order."""
  if config is None:
    return None
  repo = getattr(config.data, 'preprocessed_repo', None)
  splits = {'openwebtext-train': 'train', 'openwebtext-valid': 'validation'}
  if not repo or dataset_name not in splits:
    return None
  if (mode != splits[dataset_name] or block_size != 1024 or not wrap
      or not insert_eos or streaming or revision is not None
      or config.data.tokenizer_name_or_path != 'gpt2'):
    LOGGER.info('OWT settings differ from the prepared dataset; '
                'using raw-data preprocessing.')
    return None
  if '://' in save_dir:
    return None

  parent = os.path.dirname(save_dir)
  os.makedirs(parent, exist_ok=True)
  with FileLock(save_dir + '.lock'):
    if os.path.exists(save_dir):
      return datasets.load_from_disk(save_dir)
    LOGGER.info(f'Downloading prepared OWT {mode} from {repo} to {save_dir}.')
    dataset = _download_preprocessed(repo, cache_dir, split=mode)
    if dataset is None:
      return None
    required_columns = {'input_ids', 'attention_mask'}
    if not len(dataset) or not required_columns.issubset(dataset.column_names):
      raise ValueError(f'{repo}/{mode} must contain input_ids and attention_mask.')
    # Check boundary rows without scanning or transforming the full dataset.
    for index in {0, len(dataset) - 1}:
      row = dataset[index]
      ids = row['input_ids']
      if (len(ids) != 1024 or ids[0] != 50256 or ids[-1] != 50256
          or row['attention_mask'] != [1] * 1024):
        raise ValueError(f'{repo}/{mode} row {index} does not match the '
                         '1024-token wrapped GPT-2 format.')
    with tempfile.TemporaryDirectory(prefix='.owt-', dir=parent) as temp_dir:
      staging = os.path.join(temp_dir, 'dataset')
      dataset.save_to_disk(staging)
      os.replace(staging, save_dir)
    return datasets.load_from_disk(save_dir)


def get_dataset(dataset_name,
                tokenizer,
                wrap,
                mode,
                cache_dir,
                insert_eos=True,
                block_size=1024,
                num_proc=len(os.sched_getaffinity(0)),
                streaming=False,
                revision : Optional[str]=None,
                config=None):
  eos_tag = ''
  if not insert_eos:
    eos_tag = '_eosFalse'
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat'
  _path = os.path.join(cache_dir, filename)
  
  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  prepared = _load_preprocessed_owt(
    config, dataset_name, mode, _path, cache_dir, wrap, insert_eos,
    block_size, streaming, revision)
  if prepared is not None:
    return prepared.with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')
  LOGGER.info(f'{streaming=}')  

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only',
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    assert revision is None
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    assert revision is None
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'tiny_gsm':
    if config is None:
      raise ValueError('TinyGSM dataset requires the full config.')
    dataset = get_tiny_gsm_dataset(config, tokenizer)
  elif dataset_name == 'gsm8k_test':
    if config is None:
      raise ValueError('GSM8K test dataset requires the full config.')
    return get_gsm8k_test_dataset(config, tokenizer)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == 'synthetic':
    assert streaming
    assert wrap  # i.e., no pad tokens
    dataset = generate_synthetic_dataset(
      train_dataset_size=100000,
      validation_dataset_size=1024,
      seq_len=32,
      vocab_size=256,
    )
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]
    if dataset_name in ('synthetic', 'tiny_gsm'):
      # already tokenized, no further actions required
      return data.with_format('torch')

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      if insert_eos:
        tokens = {'input_ids':
                  [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True)
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    if not streaming:
      tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True)
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


class VocabSizeTokenizerWrapper:
  """Expose len(tokenizer) as vocab_size for tokenizers with added tokens."""

  def __init__(self, tokenizer):
    object.__setattr__(self, '_tokenizer', tokenizer)

  def _wrapped(self):
    return object.__getattribute__(self, '_tokenizer')

  @property
  def vocab_size(self):
    return len(self._wrapped())

  def __len__(self):
    return len(self._wrapped())

  def __call__(self, *args, **kwargs):
    return self._wrapped()(*args, **kwargs)

  def __getattr__(self, name):
    if name == '_tokenizer':
      raise AttributeError(name)
    return getattr(self._wrapped(), name)

  def __setattr__(self, name, value):
    if name == '_tokenizer':
      object.__setattr__(self, name, value)
    else:
      setattr(self._wrapped(), name, value)

  def __repr__(self):
    return f'Wrapped<{self._wrapped()}>'


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  elif config.data.tokenizer_name_or_path == 'synthetic':
    tokenizer = SyntheticTokenizer(vocab_size=256)
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is not None:
      tokenizer.bos_token = tokenizer.cls_token
    elif tokenizer.eos_token is not None:
      tokenizer.bos_token = tokenizer.eos_token
    else:
      raise AttributeError(
        'Tokenizer must have a bos_token, cls_token, '
        f'or eos_token: {tokenizer}')
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  if getattr(tokenizer, 'mask_token_id', None) in {
      tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}:
    tokenizer.mask_token = None

  wrap_tokenizer = config.data.tokenizer_name_or_path not in (
    'gpt2', 'bert-base-uncased', 'synthetic', 'text8')
  if wrap_tokenizer:
    tokenizer = VocabSizeTokenizerWrapper(tokenizer)

  return tokenizer
    

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      insert_eos=config.data.insert_train_eos,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers or None,
      revision=config.data.get("train_revision", None),
      config=config)
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      insert_eos=config.data.insert_valid_eos,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers or None,
      revision=config.data.get("valid_revision", None),
      config=config)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=config.loader.num_workers > 0)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
