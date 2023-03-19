# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# eg. python test_intent.py --test_file data/intent/test.json --ckpt_path /ckpt/intent/my_rnn_attention.pt
python test_intent.py --test_file "${1}" --ckpt_path "${2}"