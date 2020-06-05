#!/usr/bin/env bash
BASE_MODEL="uncased_L-12_H-768_A-12"
LARGE_MODEL="uncased_L-24_H-1024_A-16"
if [[ ! -d "Model" ]]; then
    mkdir Model
fi
# download base model
if [[ ! -d "Model/$BASE_MODEL" ]]; then
    if [[ ! -f "${BASE_MODEL}.zip" ]]; then
        echo "[INFO] download base model"
        wget "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
        echo "[INFO] Download complete."
    fi
    unzip "${BASE_MODEL}.zip" 
    mv "${BASE_MODEL}" "Model"
    rm "${BASE_MODEL}.zip" 
fi
# download large model
if [[ ! -d "Model/$LARGE_MODEL" ]]; then
    if [[ ! -f "${LARGE_MODEL}.zip" ]]; then
        echo "[INFO] Download large model."
        wget "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
        echo "[INFO] Download complete."
    fi
    unzip "${LARGE_MODEL}.zip" 
    mv "${LARGE_MODEL}" "Model"
    rm "${LARGE_MODEL}.zip" 
fi
# download base pretrained MNLI model
des_pass=123456
if [[ ! -d "Model/MNLI_pretrained_model" ]]; then
    if [[ ! -f "MNLI_pretrained_model.zip" ]]; then
        expect -c "
        set timeout -1
        spawn rsync -P --rsh=ssh byshiue@10.19.206.43:/home/byshiue/bert_experiment_data/pretrained_model/MNLI_pretrained_model.zip ./

        expect \"password:\"
        send \"${des_pass}\r\"
        expect eof
        "
    fi
    unzip "MNLI_pretrained_model.zip"
    mv "MNLI_pretrained_model" "Model"
    rm "MNLI_pretrained_model.zip"
fi
# download base pretrained MNLI model
if [[ ! -d "Model/Large_MNLI_pretrained_model" ]]; then
    if [[ ! -f "Large_MNLI_pretrained_model.zip" ]]; then
        expect -c "
        set timeout -1
        spawn rsync -P --rsh=ssh byshiue@10.19.206.43:/home/byshiue/bert_experiment_data/pretrained_model/Large_MNLI_pretrained_model.zip ./

        expect \"password:\"
        send \"${des_pass}\r\"
        expect eof
        "
    fi
    unzip "Large_MNLI_pretrained_model.zip"
    mv "Large_MNLI_pretrained_model" "Model"
    rm "Large_MNLI_pretrained_model.zip"
fi