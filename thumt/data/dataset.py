# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tensorflow as tf
import operator


def sort_input_files(filenames, reverse=True):
    inputs = []
    input_lens = []
    files = [open(name, "rb") for name in filenames]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, [list(x) for x in zip(*sorted_inputs)]


def sort_input_file(filename, reverse=True):
    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]
    
    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]
    
    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []
    
    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i
    
    return sorted_keys, sorted_inputs


def build_input_fn(filenames, mode, params):
    def train_input_fn():
        src_dataset = tf.data.TextLineDataset(filenames[0])
        hyp_dataset = tf.data.TextLineDataset(filenames[1])
        tgt_dataset = tf.data.TextLineDataset(filenames[2])

        dataset = tf.data.Dataset.zip((src_dataset, hyp_dataset, tgt_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())
        dataset = dataset.prefetch(params.buffer_size)
        dataset = dataset.shuffle(params.buffer_size)

        # Split string
        dataset = dataset.map(
            lambda x, h, y: (tf.strings.split([x]).values,
                            tf.strings.split([h]).values,
                            tf.strings.split([y]).values,
                            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def _func(x, h, y):
            def extended_x(): return tf.concat([x, [tf.constant(params.eos), tf.constant(params.src_lang_tok)]], axis=0)
            def extended_h(): return tf.concat([h, [tf.constant(params.eos), tf.constant(params.hyp_lang_tok)]], axis=0)

            if params.input_type == "":
                return ((tf.concat([x, [tf.constant(params.eos), tf.constant(params.src_lang_tok)]], axis=0),
                         tf.concat([h, [tf.constant(params.eos), tf.constant(params.hyp_lang_tok)]], axis=0),
                         tf.concat([[tf.constant(params.tgt_lang_tok)], y, [tf.constant(params.eos)]], axis=0)),
                         tf.concat([y, [tf.constant(params.eos), tf.constant(params.tgt_lang_tok)]], axis=0))
            elif params.input_type == "single_random":
                flag = tf.random.shuffle([True, False])[0]
                mixed_source = tf.cond(flag, extended_x, extended_h)
                fake_source = tf.constant([], dtype=tf.string)

                return ((mixed_source,
                         fake_source,
                         tf.concat([[tf.constant(params.tgt_lang_tok)], y, [tf.constant(params.eos)]], axis=0)),
                         tf.concat([y, [tf.constant(params.eos), tf.constant(params.tgt_lang_tok)]], axis=0))
            else:
                raise ValueError("Unknown input_type mode %s" % params.input_type)

        # Append BOS and EOS
        dataset = dataset.map(
            _func,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            lambda x, y: ({
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "hypo": x[1],
                "hypo_length": tf.shape(x[1])[0],
                "target": x[2],
                "target_length": tf.shape(x[2])[0]
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        def element_length_func(x, y):
            return tf.maximum(tf.maximum(x["source_length"], x["target_length"]), x["hypo_length"])

        def valid_size(x, y):
            size = element_length_func(x, y)
            return tf.logical_and(size >= min_length, size <= max_length)

        transformation_fn = tf.data.experimental.bucket_by_sequence_length(
            element_length_func,
            boundaries,
            batch_sizes,
            padded_shapes=({
                "source": tf.TensorShape([None]),
                "source_length": tf.TensorShape([]),
                "hypo": tf.TensorShape([None]),
                "hypo_length": tf.TensorShape([]),
                "target": tf.TensorShape([None]),
                "target_length": tf.TensorShape([])
                }, tf.TensorShape([None])),
            padding_values=({
                "source": params.pad,
                "source_length": 0,
                "hypo": params.pad,
                "hypo_length": 0,
                "target": params.pad,
                "target_length": 0
                }, params.pad),
            pad_to_bucket_boundary=False)

        dataset = dataset.filter(valid_size)
        dataset = dataset.apply(transformation_fn)

        dataset = dataset.map(
            lambda x, y: ({
                "source": x["source"],
                "source_mask": tf.sequence_mask(x["source_length"],
                                                tf.shape(x["source"])[1],
                                                tf.float32),
                "hypo": x["hypo"],
                "hypo_mask": tf.sequence_mask(x["hypo_length"],
                                                tf.shape(x["hypo"])[1],
                                                tf.float32),
                "target": x["target"],
                "target_mask": tf.sequence_mask(x["target_length"],
                                                tf.shape(x["target"])[1],
                                                tf.float32)
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def eval_input_fn():
        src_dataset = tf.data.TextLineDataset(filenames[0])
        hyp_dataset = tf.data.TextLineDataset(filenames[1])
        tgt_dataset = tf.data.TextLineDataset(filenames[2])
        dataset = tf.data.Dataset.zip((src_dataset, hyp_dataset, tgt_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())
        dataset = dataset.prefetch(params.buffer_size)

        # Split string
        dataset = dataset.map(
            lambda x, h, y: (tf.strings.split([x]).values,
                            tf.strings.split([h]).values,
                            tf.strings.split([y]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Append BOS and EOS
        dataset = dataset.map(
            lambda x, h, y: (
                (tf.concat([x, [tf.constant(params.eos), tf.constant(params.src_lang_tok)]], axis=0),
                 tf.concat([h, [tf.constant(params.eos), tf.constant(params.hyp_lang_tok)]], axis=0),
                 tf.concat([[tf.constant(params.tgt_lang_tok)], y, [tf.constant(params.eos)]], axis=0)),
                tf.concat([y, [tf.constant(params.eos), tf.constant(params.tgt_lang_tok)]], axis=0)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            lambda x, y: ({
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "hypo": x[1],
                "hypo_length": tf.shape(x[1])[0],
                "target": x[2],
                "target_length": tf.shape(x[2])[0]
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Batching
        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes=({
                "source": tf.TensorShape([None]),
                "source_length": tf.TensorShape([]),
                "hypo": tf.TensorShape([None]),
                "hypo_length": tf.TensorShape([]),
                "target": tf.TensorShape([None]),
                "target_length": tf.TensorShape([])
                }, tf.TensorShape([None])),
            padding_values=({
                "source": params.pad,
                "source_length": 0,
                "hypo": params.pad,
                "hypo_length": 0,
                "target": params.pad,
                "target_length": 0
                }, params.pad))

        dataset = dataset.map(
            lambda x, y: ({
                "source": x["source"],
                "source_mask": tf.sequence_mask(x["source_length"],
                                                tf.shape(x["source"])[1],
                                                tf.float32),
                "hypo": x["hypo"],
                "hypo_mask": tf.sequence_mask(x["hypo_length"],
                                                tf.shape(x["hypo"])[1],
                                                tf.float32),
                "target": x["target"],
                "target_mask": tf.sequence_mask(x["target_length"],
                                                tf.shape(x["target"])[1],
                                                tf.float32)
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def infer_input_fn():
        sorted_key, sorted_data = sort_input_files(filenames)

        datasets = [tf.data.Dataset.from_tensor_slices(data) for data in sorted_data]

        dataset = tf.data.Dataset.zip(tuple(datasets))

        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.map(
            lambda x, h: (tf.strings.split([x]).values,
                          tf.strings.split([h]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            lambda x, h: (tf.concat([x, [tf.constant(params.eos), tf.constant(params.src_lang_tok)]], axis=0),
                          tf.concat([h, [tf.constant(params.eos), tf.constant(params.hyp_lang_tok)]], axis=0)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            lambda x, h: {
                "source": x,
                "source_length": tf.shape(x)[0],
                "hypo": h,
                "hypo_length": tf.shape(h)[0]
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes={
                "source": tf.TensorShape([None]),
                "source_length": tf.TensorShape([]),
                "hypo": tf.TensorShape([None]),
                "hypo_length": tf.TensorShape([])
            },
            padding_values={
                "source": tf.constant(params.pad),
                "source_length": 0,
                "hypo": tf.constant(params.pad),
                "hypo_length": 0
            })

        dataset = dataset.map(
            lambda x: {
                "source": x["source"],
                "source_mask": tf.sequence_mask(x["source_length"],
                                                tf.shape(x["source"])[1],
                                                tf.float32),
                "hypo": x["hypo"],
                "hypo_mask": tf.sequence_mask(x["hypo_length"],
                                                tf.shape(x["hypo"])[1],
                                                tf.float32),
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return sorted_key, dataset

    if mode == "train":
        return train_input_fn
    if mode == "eval":
        return eval_input_fn
    elif mode == "infer":
        return infer_input_fn
    else:
        raise ValueError("Unknown mode %s" % mode)


def get_dataset(filenames, mode, params):
    input_fn = build_input_fn(filenames, mode, params)

    with tf.device("/cpu:0"):
        dataset = input_fn()

    return dataset
