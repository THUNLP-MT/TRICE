# TRICE: a task-agnostic transferring framework for multi-source sequence generation

This is the source code of our work <a href="https://arxiv.org/abs/2105.14809">Transfer Learning for Sequence Generation: from Single-source to Multi-source (ACL 2021)</a>. 

We propose TRICE, a task-agnostic Transferring fRamework for multI-sourCe sEquence generation, 
for transferring pretrained models to multi-source sequence generation tasks 
(e.g., automatic post-editing, multi-source translation, and multi-document summarization).
TRICE achieves new state-of-the-art results on the WMT17 APE task and the multi-source translation task using the WMT14 test set.
Welcome to take a quick glance at our [blog](https://thumtblog.github.io/2021/05/27/TRICE/). 

The implementation is on top of the open-source NMT toolkit [THUMT](https://github.com/thumt/THUMT). 

```bibtex
@misc{huang2021transfer,
      title={Transfer Learning for Sequence Generation: from Single-source to Multi-source}, 
      author={Xuancheng Huang and Jingfang Xu and Maosong Sun and Yang Liu},
      year={2021},
      eprint={2105.14809},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contents
* [Prerequisites](#Prerequisites)
* [Pretrained&#32;model](#Pretrained&#32;model)
* [Finetuning](#Finetuning)
* [Inference](#Inference)
* [Contact](#Contact)

## Prerequisites
* Python >= 3.6
* tensorflow-cpu >= 2.0
* torch >= 1.7
* transformers >= 3.4
* sentencepiece >= 0.1

## Pretrained&#32;model
We adopt [mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) in our experiments. 
Other sequence-to-sequence pretrained models can also be used with only a few modifications.

If your GPUs do not have enough memories, you can prune the original large vocabulary (25k) to a small vocabulary (e.g., 3k) with little performance loss.

## Finetuning

### Single-source finetuning
```powershell
PYTHONPATH=${path_to_TRICE} \
python ${path_to_TRICE}/thumt/bin/trainer.py \
    --input ${train_src1} ${train_src2} ${train_trg} \
    --vocabulary ${vocab_joint} ${vocab_joint} \
    --validation ${dev_src1} ${dev_src2} \
    --references ${dev_ref} \
    --model transformer --half --hparam_set big \
    --output single_finetuned \
    --parameters \
fixed_batch_size=false,batch_size=820,train_steps=120000,update_cycle=5,device_list=[$devs],\
keep_checkpoint_max=2,save_checkpoint_steps=2000,\
eval_steps=2001,decode_alpha=1.0,decode_batch_size=16,keep_top_checkpoint_max=1,\
attention_dropout=0.1,relu_dropout=0.1,residual_dropout=0.1,learning_rate=5e-05,warmup_steps=4000,initial_learning_rate=5e-8,\
separate_encode=false,separate_cross_att=false,segment_embedding=false,\
input_type="single_random",adapter_type="None",num_fine_encoder_layers=0,normalization="after",\
src_lang_tok="en_XX",hyp_lang_tok="de_DE",tgt_lang_tok="de_DE",mbart_model_code="facebook/mbart-large-cc25",\
spm_path="sentence.bpe.model",pad="<pad>",bos="<s>",eos="</s>",unk="<unk>"
```

### Multi-source finetuning
```powershell
PYTHONPATH=${path_to_TRICE} \
python ${path_to_TRICE}/thumt/bin/trainer.py \
    --input ${train_src1} ${train_src2} ${train_tgt} \
    --vocabulary ${vocab_joint} ${vocab_joint} \
    --validation ${dev_src1} ${dev_src2} \
    --references ${dev_ref} \
    --model transformer --half --hparam_set big \
    --checkpoint single_finetuned/eval/model-best.pt \
    --output multi_finetuned \
    --parameters \
fixed_batch_size=false,batch_size=820,train_steps=120000,update_cycle=5,device_list=[0,1,2,3],\
keep_checkpoint_max=2,save_checkpoint_steps=2000,\
eval_steps=2001,decode_alpha=1.0,decode_batch_size=16,keep_top_checkpoint_max=1,\
attention_dropout=0.1,relu_dropout=0.1,residual_dropout=0.1,learning_rate=5e-05,warmup_steps=4000,initial_learning_rate=5e-8,special_learning_rate=5e-04,special_var_name="adapter",\
separate_encode=false,separate_cross_att=true,segment_embedding=true,\
input_type="",adapter_type="Cross-attn",num_fine_encoder_layers=1,normalization="after",\
src_lang_tok="en_XX",hyp_lang_tok="de_DE",tgt_lang_tok="de_DE",mbart_model_code="facebook/mbart-large-cc25",\
spm_path="sentence.bpe.model",pad="<pad>",bos="<s>",eos="</s>",unk="<unk>"
```

### Arguments to be explained

** `special_learning_rate`: if a variable's name contains `special_var_name`, the learning rate of it will be `special_learning_rate`. We give the fine encoder a larger learning rate.  
** `separate_encode`: whether to encode multiple sources separately before the fine encoder.  
** `separate_cross_att`: whether to use separated cross-attention described in our paper.  
** `segment_embedding`: whether to use sinusoidal segment embedding described in our paper.  
** `input_type`: `"single_random"` for single-source finetuning , `""` for multi-source finetuning.  
** `adapter_type`: `"None"` for no fine encoder,  `"Cross-attn"` for fine encoder with cross-attention.  
** ``num_fine_encoder_layers``: number of fine encoder layers.  
** ``src_lang_tok``: language token for the first source sentence. Please refer to [here](https://huggingface.co/facebook/mbart-large-cc25) for language tokens for all 25 languages.  
** ``hyp_lang_tok``: language token for the second source sentence.  
** ``tgt_lang_tok``: language token for the target sentence.  
** ``mbart_model_code``: model code for [transformers](https://huggingface.co/models).  
** ``spm_path``: sentence piece model (can download from [here](https://huggingface.co/facebook/mbart-large-cc25/tree/main)).

Explanations for other arguments could be found in the user manual of [THUMT](https://github.com/thumt/THUMT).

## Inference
```
PYTHONPATH=${path_to_TRICE} \
python ${path_to_TRICE}/thumt/bin/translator.py \
  --input ${test_src1} ${test_src2} --output ${test_tgt} \
  --vocabulary ${vocab_joint} ${vocab_joint} \
  --checkpoints multi_finetuned/eval/model-best.pt \
  --model transformer --half \
  --parameters device_list=[0,1,2,3],decode_alpha=1.0,decode_batch_size=32
# recover sentence piece tokenization
...
# calculate BLEU
...
```

## Contact

If you have questions, suggestions and bug reports, please email [xchuang17@163.com](xchuang17@163.com).