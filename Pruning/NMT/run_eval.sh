python -m nmt.nmt \
    --src=en --tgt=vi \
    --ckpt=${1} \
    --hparams_path=nmt/standard_hparams/iwslt15.json \
    --out_dir=./tmp/envi \
    --vocab_prefix=./data_set/vocab  \
    --inference_input_file=./data_set/tst2013.en \
    --inference_output_file=./tmp/envi/output_infer \
    --inference_ref_file=./data_set/tst2013.vi