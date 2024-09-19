import argparse
import ast
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig)

from utils import (DEFAULT_HF_MODEL_DIRS, add_common_args, load_tokenizer,
                   read_model_name, supports_inflight_batching)

from vllm import LLM, SamplingParams

os.environ['HTTP_PROXY'] = 'http://12.2.20.28:8388'
os.environ['HTTPS_PROXY'] = 'http://12.2.20.28:8388'



def main(args):
    test_vllm = args.test_vllm
    model_name = 'LlamaForCausalLM'
    
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=None,
        tokenizer_type=args.tokenizer_type,
    )
    
    dataset_name = '/weight/openai_humaneval/'
    dataset_input_key = "prompt"
    dataset_output_key = "canonical_solution"
    dataset_revision = None
    dataset_split = 'test'
    
    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_cache_dir,
                           split=dataset_split)

    max_batch_size = args.batch_size

    # runtime parameters
    top_k = args.top_k
    top_p = args.top_p
    output_len = args.output_len
    test_token_num = args.max_input_length
    max_attention_window_size = args.max_attention_window_size
    sink_token_length = args.sink_token_length

    if args.end_id:
        end_id = args.end_id

    stop_words_list = None
    bad_words_list = None

    temperature = args.temperature
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    early_stopping = args.early_stopping
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if test_vllm:
            with (output_dir / 'vllm.out').open('w') as f:
                f.write(f'Engine path: {args.engine_dir}\n')
                f.write(f'Tokenizer path: {args.tokenizer_dir}\n')

    # TODO: Add random_seed flag in gptj
    rouge_dir = args.rouge_dir if args.rouge_dir and os.path.exists(
        args.rouge_dir) else "rouge"
    metric_vllm = [evaluate.load(rouge_dir) for _ in range(num_beams)]
    for i in range(num_beams):
        metric_vllm[i].seed = 0
    ppls_vllm = [[] for _ in range(num_beams)]
    
    # ----------------------------------------------------------------------------------------------------------------
    def _prepare_inputs(batch_input_texts,
                        eval_task='summarize',
                        add_special_tokens=True,
                        min_input_length=0):
        batch_size = len(batch_input_texts)
        append_str = ' TL;DR: ' if eval_task == 'summarize' else ''
        batch_input_ids = []
        for i in range(batch_size):
            curr_text = batch_input_texts[i] + append_str
            curr_text = curr_text.strip().replace(" n't", "n't")
            input_ids = tokenizer.encode(
                curr_text,
                return_tensors='pt',
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=test_token_num).squeeze(0)

            if input_ids.numel() > min_input_length:
                batch_input_ids.append(input_ids)
        return batch_input_ids
    
    def eval_vllm(datapoint,
                  eval_task='summarize',
                  eval_ppl=False,
                  add_special_tokens=True,
                  min_input_length=0):
        batch_size = len(datapoint['prompt'])
        batch_input_ids = _prepare_inputs(datapoint['prompt'],
                                          eval_task=eval_task,
                                          add_special_tokens=add_special_tokens,
                                          min_input_length=min_input_length)
        batch_input_ids = [x.tolist() for x in batch_input_ids]
        
        batch_size = len(batch_input_ids)
        if batch_size == 0:
            return [], [], [], {}
        input_lengths = [len(x) for x in batch_input_ids]
        
        # import pdb;pdb.set_trace()
        outputs = llm.generate(prompt_token_ids=batch_input_ids, sampling_params=sampling_params)

        output_ids = torch.tensor(batch_input_ids[0] + list(outputs[0].outputs[0].token_ids), dtype=torch.int32)
        output_ids = output_ids.view(1, 1, *output_ids.shape)
        output_beams_list = [
            tokenizer.batch_decode(output_ids[batch_idx, :,
                                                input_lengths[batch_idx]:],
                                    skip_special_tokens=True)
            for batch_idx in range(batch_size)
        ]
        output_ids_list = [
            output_ids[batch_idx, :, input_lengths[batch_idx]:]
            for batch_idx in range(batch_size)
        ]

        ppls = [[] for _ in range(batch_size)]
        # seq_lengths_array = outputs["sequence_lengths"].cpu().tolist()
        seq_lengths_array = [[len(batch_input_ids[0] + list(outputs[0].outputs[0].token_ids))]]
        lengths_info = {
            'input_lengths': input_lengths,
            'seq_lengths': seq_lengths_array
        }
        return output_beams_list, output_ids_list, ppls, lengths_info
    # ----------------------------------------------------------------------------------------------------------------
    
    if test_vllm:
        sampling_params = SamplingParams(temperature=args.temperature, 
                                            top_k=1)

        # Create an LLM.
        # llm = LLM(model="/weight/ckpt_8b/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/",
        #           tensor_parallel_size=1)
        # llm = LLM(model="/weight/models--neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w8a16/",
        #           tensor_parallel_size=4)
        llm = LLM(model="/weight/ckpt_76b/snapshots/846357c7ee5e3f50575fd4294edb3d898c8ea100/",
                  tensor_parallel_size=8)

        datapoint = dataset[0:1]
        output, *_ = eval_vllm(datapoint,
                               eval_task=args.eval_task,
                               eval_ppl=args.eval_ppl,
                               add_special_tokens=args.add_special_tokens,
                               min_input_length=args.min_input_length)
            
        print("---------------------------------------------------------")
        print("vLLM Generated : ")
        print(f" Input : {datapoint[dataset_input_key]}")
        print(f"\n Reference : {datapoint[dataset_output_key]}")
        print(f"\n Output : {output}")
        print("---------------------------------------------------------")
        
        ite_count = 0
        data_point_idx = 0
        total_output_token_count_vllm = 0  # only valid for runtime_rank == 0
        while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
            print(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
            datapoint = dataset[data_point_idx:(data_point_idx + max_batch_size)]
            output_vllm, output_ids_vllm, curr_ppls_vllm, lengths_info = eval_vllm(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens,
                min_input_length=args.min_input_length)
            if output_vllm == []:
                data_point_idx += max_batch_size
                ite_count += 1
                continue

            input_lengths = lengths_info['input_lengths']
            seq_lengths = lengths_info['seq_lengths']
            output_token_count_vllm = sum(
                seq_lengths[bs][bm] - input_lengths[bs]
                for bm in range(len(output_vllm[0]))
                for bs in range(len(output_vllm)))
            total_output_token_count_vllm += output_token_count_vllm
            
            for batch_idx in range(len(output_vllm)):
                for beam_idx in range(num_beams):
                    metric_vllm[beam_idx].add_batch(
                        predictions=[
                            output_vllm[batch_idx][beam_idx]
                        ],
                        references=[
                            datapoint[dataset_output_key][batch_idx]
                        ])
                    if args.eval_ppl:
                        ppls_vllm[beam_idx].append(
                            curr_ppls_vllm[batch_idx][beam_idx])
            if output_dir is not None:
                for i in range(len(output_vllm[0])):
                    for beam_idx in range(num_beams):
                        with (output_dir / 'vllm.out').open('a') as f:
                            f.write(
                                f'[{data_point_idx + i}] [Beam {beam_idx}] {output_vllm[beam_idx][i]}\n'
                            )
            print("---------------------------------------------------------")
            print(f"Input : {datapoint[dataset_input_key]}")
            print(f'vLLM Output: {output_vllm}')
            print(f"Reference : {datapoint[dataset_output_key]}")
            print("---------------------------------------------------------")
            
            data_point_idx += max_batch_size
            ite_count += 1
        del llm
        
    if test_vllm:
        np.random.seed(0)  # rouge score use sampling to compute the score
        for beam_idx in range(num_beams):
            print(f"vLLM beam {beam_idx} result")
            computed_metrics_vllm = metric_vllm[
                beam_idx].compute()
            if args.eval_task != "eval_context_ppl":
                for key in computed_metrics_vllm.keys():
                    print(
                        f'  {key} : {computed_metrics_vllm[key]*100}'
                    )

            if args.check_accuracy and beam_idx == 0 and args.eval_task != "eval_context_ppl":
                assert computed_metrics_vllm[
                    'rouge1'] * 100 > args.vllm_rouge1_threshold
            if args.eval_ppl:
                print(
                    f"  Per-token perplexity: {np.mean(ppls_vllm[beam_idx])}"
                )
                if args.check_accuracy and beam_idx == 0:
                    avg_ppl = np.mean(ppls_vllm[beam_idx])
                    assert avg_ppl < args.vllm_ppl_threshold, f"[FAILED] average PPL ({avg_ppl}) is larger than threshold ({args.vllm_ppl_threshold})"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_vllm', action='store_true')
    parser.add_argument('--eval_task',
                        type=str,
                        default='summarize',
                        choices=[
                            'summarize', 'summarize_long', 'code_completion',
                            'eval_context_ppl'
                        ])
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--vllm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--vllm_ppl_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None,
        help="The local directory of the dataset for evaluation; "
        "will download the dataset from huggingface hub if not specified.")
    parser.add_argument(
        '--dataset_cache_dir',
        type=str,
        default=None,
        help="The local cache directory for dataset; "
        "will use `~/.cache/huggingface/datasets` if not specified.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=100)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument(
        '--min_input_length',
        type=int,
        default=0,
        help='skip the sentences which are shorter than min_input_length.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory where to save output sentences. 'trtllm.out' for "
        "TensorRT-LLM outputs, and 'hf.out' for HF outputs.  If None, do not "
        "save outputs.")
    parser.add_argument(
        '--rouge_dir',
        default=None,
        type=str,
        help=
        "evaluate.load('rouge') will attempt to pull rouge package from HF. Use cached rouge can avoid network outage of host or HF."
    )
    parser = add_common_args(parser)
    args = parser.parse_args()

    main(args)