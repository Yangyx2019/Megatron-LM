set -e

cd ~/Megatron-LM

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

TP=2
PP=1

torchrun --standalone --nproc_per_node=2 pretrain_gpt.py \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --transformer-impl local \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  \
  --micro-batch-size 2 \
  --global-batch-size 4 \
  --train-iters 200 \
  \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --lr-warmup-iters 2 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  \
  --bf16 \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --split 949,50,1 \
  \
  --log-interval 1 \
  --eval-interval 10 \
  --eval-iters 2 \
  2>&1 | tee pretrain_gpt_tp2_mock.log
