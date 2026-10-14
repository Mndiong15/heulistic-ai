export interface SchemaField {
  key: string
  label: string
  description: string
  type: 'string' | 'number' | 'boolean' | 'enum'
  options?: string[]
  default?: string | number | boolean
  category: 'Training' | 'Checkpointing' | 'Hardware' | 'Model' | 'Dataset' | 'Advanced'
}

// Fields already hardcoded in the config builder — excluded from the dropdown
export const BUILDER_FIELDS = new Set([
  'base_model', 'tokenizer_type', 'adapter', 'load_in_4bit',
  'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_linear',
  'val_set_size', 'num_epochs', 'micro_batch_size', 'gradient_accumulation_steps',
  'learning_rate', 'sequence_len', 'bf16', 'gradient_checkpointing',
  'sample_packing', 'lr_scheduler',
])

export const AXOLOTL_SCHEMA: SchemaField[] = [
  // ── Training ────────────────────────────────────────────────────────────────
  {
    key: 'seed',
    label: 'Seed',
    description: 'Random seed for reproducibility.',
    type: 'number',
    default: 42,
    category: 'Training',
  },
  {
    key: 'max_steps',
    label: 'Max steps',
    description: 'Maximum number of training steps. Takes precedence over num_epochs.',
    type: 'number',
    category: 'Training',
  },
  {
    key: 'warmup_steps',
    label: 'Warmup steps',
    description: 'Number of linear warmup steps at the start of training. Cannot be used with warmup_ratio.',
    type: 'number',
    category: 'Training',
  },
  {
    key: 'warmup_ratio',
    label: 'Warmup ratio',
    description: 'Fraction of total steps used for linear warmup. Cannot be used with warmup_steps.',
    type: 'number',
    default: 0.03,
    category: 'Training',
  },
  {
    key: 'weight_decay',
    label: 'Weight decay',
    description: 'L2 regularization coefficient applied to model weights.',
    type: 'number',
    default: 0.0,
    category: 'Training',
  },
  {
    key: 'max_grad_norm',
    label: 'Max grad norm',
    description: 'Gradient clipping threshold — gradients larger than this are scaled down.',
    type: 'number',
    default: 1.0,
    category: 'Training',
  },
  {
    key: 'adam_beta1',
    label: 'Adam β₁',
    description: 'AdamW first moment decay rate.',
    type: 'number',
    default: 0.9,
    category: 'Training',
  },
  {
    key: 'adam_beta2',
    label: 'Adam β₂',
    description: 'AdamW second moment decay rate.',
    type: 'number',
    default: 0.999,
    category: 'Training',
  },
  {
    key: 'adam_epsilon',
    label: 'Adam ε',
    description: 'AdamW numerical stability term.',
    type: 'number',
    default: 1e-8,
    category: 'Training',
  },
  {
    key: 'neftune_noise_alpha',
    label: 'NEFTune noise alpha',
    description: 'Adds noise to embeddings during training to improve generalisation (paper default: 5). Only supported on Llama and Mistral.',
    type: 'number',
    category: 'Training',
  },
  {
    key: 'train_on_inputs',
    label: 'Train on inputs',
    description: 'When false (default), loss is computed only on the assistant response tokens, not the user prompt.',
    type: 'boolean',
    default: false,
    category: 'Training',
  },
  {
    key: 'group_by_length',
    label: 'Group by length',
    description: 'Group similar-length examples to reduce padding waste. May slow down the start of training.',
    type: 'boolean',
    default: false,
    category: 'Training',
  },

  // ── Checkpointing ───────────────────────────────────────────────────────────
  {
    key: 'eval_steps',
    label: 'Eval steps',
    description: 'Run evaluation every N steps. Leave unset to evaluate at each epoch.',
    type: 'number',
    category: 'Checkpointing',
  },
  {
    key: 'save_steps',
    label: 'Save steps',
    description: 'Save a checkpoint every N steps. Leave unset to save at each epoch.',
    type: 'number',
    category: 'Checkpointing',
  },
  {
    key: 'save_total_limit',
    label: 'Save total limit',
    description: 'Maximum number of checkpoints to keep on disk. Older ones are deleted.',
    type: 'number',
    default: 3,
    category: 'Checkpointing',
  },
  {
    key: 'logging_steps',
    label: 'Logging steps',
    description: 'Log training metrics every N steps.',
    type: 'number',
    default: 10,
    category: 'Checkpointing',
  },
  {
    key: 'early_stopping_patience',
    label: 'Early stopping patience',
    description: 'Stop training after this many evaluations show increasing loss.',
    type: 'number',
    category: 'Checkpointing',
  },
  {
    key: 'save_strategy',
    label: 'Save strategy',
    description: 'When to save checkpoints.',
    type: 'enum',
    options: ['epoch', 'steps', 'best', 'no'],
    default: 'epoch',
    category: 'Checkpointing',
  },
  {
    key: 'eval_strategy',
    label: 'Eval strategy',
    description: 'When to run evaluation.',
    type: 'enum',
    options: ['epoch', 'steps', 'no'],
    default: 'epoch',
    category: 'Checkpointing',
  },

  // ── Hardware ────────────────────────────────────────────────────────────────
  {
    key: 'attn_implementation',
    label: 'Attention implementation',
    description: 'Which attention kernel to use. flash_attention_2 is fastest on A10G/A100 and requires bf16.',
    type: 'enum',
    options: ['flash_attention_2', 'flash_attention_3', 'sdpa', 'flex_attention', 'eager', 'xformers', 'sage', 'fp8'],
    category: 'Hardware',
  },
  {
    key: 'torch_compile',
    label: 'torch.compile',
    description: 'Compile the model graph with torch.compile for faster training (may increase startup time).',
    type: 'boolean',
    default: false,
    category: 'Hardware',
  },
  {
    key: 'fp8',
    label: 'FP8 training',
    description: 'Enable FP8 mixed precision using TorchAO. Best combined with torch_compile. Requires Hopper+ GPU.',
    type: 'boolean',
    default: false,
    category: 'Hardware',
  },
  {
    key: 'load_in_8bit',
    label: 'Load in 8-bit',
    description: 'Quantize the base model to 8-bit using bitsandbytes to reduce VRAM usage.',
    type: 'boolean',
    default: false,
    category: 'Hardware',
  },
  {
    key: 'layer_offloading',
    label: 'Layer offloading',
    description: 'Offload model layer parameters to CPU during forward pass and prefetch during backward. Reduces GPU VRAM at the cost of speed.',
    type: 'boolean',
    default: false,
    category: 'Hardware',
  },

  // ── Model ───────────────────────────────────────────────────────────────────
  {
    key: 'trust_remote_code',
    label: 'Trust remote code',
    description: 'Allow executing custom model code from HuggingFace Hub. Required for some models (e.g. Falcon, Phi).',
    type: 'boolean',
    default: false,
    category: 'Model',
  },
  {
    key: 'resize_token_embeddings_to_32x',
    label: 'Resize embeddings to 32×',
    description: 'Pad token embedding table to the nearest multiple of 32 after adding new tokens. Can improve training speed.',
    type: 'boolean',
    category: 'Model',
  },
  {
    key: 'hub_model_id',
    label: 'Hub model ID',
    description: 'Push the trained model to this HuggingFace Hub repo (e.g. your-org/my-model).',
    type: 'string',
    category: 'Model',
  },
  {
    key: 'revision_of_model',
    label: 'Model revision',
    description: 'Specific commit hash, tag, or branch of the base model to load from HuggingFace Hub.',
    type: 'string',
    category: 'Model',
  },

  // ── Dataset ─────────────────────────────────────────────────────────────────
  {
    key: 'dataset_prepared_path',
    label: 'Dataset prepared path',
    description: 'Cache tokenised dataset to this path so subsequent runs load faster.',
    type: 'string',
    default: 'last_run_prepared',
    category: 'Dataset',
  },
  {
    key: 'dataset_processes',
    label: 'Dataset processes',
    description: 'Number of CPU processes to use when preprocessing the dataset. Defaults to os.cpu_count().',
    type: 'number',
    category: 'Dataset',
  },
  {
    key: 'shuffle_merged_datasets',
    label: 'Shuffle merged datasets',
    description: 'Shuffle the combined dataset before training. Disable for curriculum learning.',
    type: 'boolean',
    default: true,
    category: 'Dataset',
  },
  {
    key: 'pad_to_sequence_len',
    label: 'Pad to sequence length',
    description: 'Pad all inputs to sequence_len. Reduces memory fragmentation. Enabled automatically with sample_packing.',
    type: 'boolean',
    category: 'Dataset',
  },
  {
    key: 'eval_sequence_len',
    label: 'Eval sequence length',
    description: 'Maximum token length for evaluation inputs. Defaults to sequence_len if not set.',
    type: 'number',
    category: 'Dataset',
  },
  {
    key: 'eval_batch_size',
    label: 'Eval batch size',
    description: 'Per-GPU batch size for evaluation. Defaults to micro_batch_size.',
    type: 'number',
    category: 'Dataset',
  },

  // ── Advanced ────────────────────────────────────────────────────────────────
  {
    key: 'rl',
    label: 'RL training type',
    description: 'Enable reinforcement-learning-based fine-tuning using one of the supported algorithms.',
    type: 'enum',
    options: ['dpo', 'ipo', 'kto', 'simpo', 'orpo', 'grpo', 'ebft'],
    category: 'Advanced',
  },
  {
    key: 'chat_template',
    label: 'Chat template',
    description: 'Chat template to apply to the dataset. tokenizer_default uses whatever is in the model\'s tokenizer_config.json.',
    type: 'enum',
    options: ['tokenizer_default', 'alpaca', 'inst', 'chatml', 'gemma', 'cohere', 'llama3', 'phi_3', 'deepseek_v2', 'jamba', 'jinja'],
    default: 'tokenizer_default',
    category: 'Advanced',
  },
  {
    key: 'default_system_message',
    label: 'Default system message',
    description: 'Override the default system prompt. Currently only supported with the chatml template.',
    type: 'string',
    category: 'Advanced',
  },
  {
    key: 'loss_watchdog_threshold',
    label: 'Loss watchdog threshold',
    description: 'Abort training if loss exceeds this value for several consecutive steps (good estimate: 2× starting loss).',
    type: 'number',
    category: 'Advanced',
  },
  {
    key: 'loss_watchdog_patience',
    label: 'Loss watchdog patience',
    description: 'Number of consecutive high-loss steps before the watchdog aborts training.',
    type: 'number',
    default: 3,
    category: 'Advanced',
  },
  {
    key: 'include_tokens_per_second',
    label: 'Report tokens per second',
    description: 'Log tokens/sec throughput at the end of training.',
    type: 'boolean',
    category: 'Advanced',
  },
]
