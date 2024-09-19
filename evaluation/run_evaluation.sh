python ./evaluation_vllm.py --test_vllm \
                            --tokenizer_dir /weight/ckpt_8b/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/ \
                            --data_type bf16 \
                            --engine_dir /aix_model_server/s-model/aix4_instruct_v3_engine/ \
                            --eval_task code_completion \
                            --max_input_len 16384 \
                            --use_py_session \
                            --dataset_dir /weight/openai_humaneval/ \
                            # --output_dir /aix_model_server/evaluation/result