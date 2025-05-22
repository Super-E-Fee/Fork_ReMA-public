export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

MODEL_PATH={YOUR_MODEL_PATH}
OUTPUT_PATH={YOUR_OUTPUT_PATH}

python src/verl/verl/rema_trainer/main_generation.py \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.batch_size=128 \
    data.output_path=$OUTPUT_PATH \
    multi_agent.parameter_sharing=True \
    multi_agent.mta_model=$MODEL_PATH \
    rollout.max_num_turns=10 \
    rollout.gpu_memory_utilization=0.7 \
    rollout.tensor_model_parallel_size=1
