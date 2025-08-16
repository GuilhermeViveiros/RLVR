import "sc
global {
    ducttape_experimental_imports=true
    ducttape_experimental_submitters=true
    ducttape_experimental_multiproc=true
}

func TrainModel
    < initial_model_dir
    > trained_model_dir
    :: repo
    :: model_name
    :: dataset_name
    :: training_mode
    :: wandb_project
    :: max_prompt_length
    :: reward_funcs
    :: num_epochs
    :: num_generations
    :: max_completion_length
    :: save_steps
    

    # For debugging purposes
    export NCCL_DEBUG=WARN

    export WANDB_PROJECT=$wandb_project
    export WANDB_MODE=offline

    # check if the initial model dir exists
    if not exists $initial_model_dir:
        error "Initial model directory does not exist"
    
     # Create base run name
    RUN_NAME="${training_mode}::${model_name}::${dataset_name}"
    echo "RUN_NAME: ${RUN_NAME}"

    # calcualte the gradient accumulation steps according to the batch size and the number of GPUs

    # Run the training command
    deepspeed vision_r1/visionr1_qwen.py \
    --deepspeed ${repo}/Vision-R1/configs/zero3.json \
    --output_dir ${trained_model_dir} \
    --model_name_or_path ${model_name}  \
    --dataset_name ${dataset_name} \
    --max_prompt_length ${max_prompt_length} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --reward_funcs ${reward_funcs} \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs ${num_epochs} \
    --run_name ${RUN_NAME} \
    --save_steps ${save_steps} \
    --save_only_model true \
    --num_generations ${num_generations} \
    --max_completion_length ${max_completion_length} \
    --torch_dtype bfloat16 
}