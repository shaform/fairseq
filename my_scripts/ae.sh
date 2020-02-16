CUDA_VISIBLE_DEVICES=7 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_ae_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion triple_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 200 \
    --left-pad-source False \
    --task translation_autoencode \
    --encoder-vq-n-token 10000 \
    --encoder-vq-decay 0.99 \
    --encoder-vq-beta 0.25 \
    --encoder-vq-rho-start 0.1 \
    --encoder-vq-rho-warmup-updates 139800 \
    --best-checkpoint-metric loss_tgt \
    --save-dir models/vq-ae
