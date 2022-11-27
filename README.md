# d2v-audio
data2vec for general audio inputs

work in progress (it works but it doesn't work)

#### Example configuration
```
train_data_dirs:
  - audio/dataset_1/train
  - audio/dataset_2/train

val_data_dirs:
  - audio/dataset_1/val
  - audio/dataset_2/val

sample_rate: 16000
n_fft: 1024
hop_length: 512

min_duration: 9.0
max_duration: 9.0

epochs: 100
batch_size: 256
num_dataloader_workers: 8

d_model: 512
d_ff: 2048
n_layers: 12
n_heads: 8
max_sequence_length: 512

p_masking: 0.15
masking_length: 6

learning_rate_factor: 0.1
warmup_steps: 6000
ema_decay: 0.9999
lambda_var: 0

model_path: data/model.pt

log_interval: 100
```

