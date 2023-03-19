mkdir eval-summarization

# eval-summarization
wget https://www.dropbox.com/s/yle5awbc31l6pzg/all_results.json?dl=1 -O ./eval-summarization/all_results.json
wget https://www.dropbox.com/s/uhr503pxejw0peb/config.json?dl=1 -O ./eval-summarization/config.json
wget https://www.dropbox.com/s/o5j9thk1w6l9qje/pytorch_model.bin?dl=1 -O ./eval-summarization/pytorch_model.bin
wget https://www.dropbox.com/s/7oocq6thp2k1edn/special_tokens_map.json?dl=1 -O ./eval-summarization/special_tokens_map.json
wget https://www.dropbox.com/s/f93vsot2wxj3ws9/spiece.model?dl=1 -O ./eval-summarization/spiece.model
wget https://www.dropbox.com/s/5mzb0dayipvv7u9/tokenizer_config.json?dl=1 -O ./eval-summarization/tokenizer_config.json
wget https://www.dropbox.com/s/i3ojo5arp2l2qvf/tokenizer.json?dl=1 -O ./eval-summarization/tokenizer.json
wget https://www.dropbox.com/s/tpul1gkxohfbmp7/train_results.json?dl=1 -O ./eval-summarization/train_results.json
wget https://www.dropbox.com/s/o7fpdng1uhqew2m/trainer_state.json?dl=1 -O ./eval-summarization/trainer_state.json
wget https://www.dropbox.com/s/o9m9a7kbm8z649s/training_args.bin?dl=1 -O ./eval-summarization/training_args.bin