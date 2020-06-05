data_dir="squad_data"
mkdir ${data_dir}
wget -P ${data_dir} https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -P ${data_dir} https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json