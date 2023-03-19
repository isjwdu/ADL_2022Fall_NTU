# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# eg. python test_slot.py --test_file data/slot/test.json --ckpt_path /ckpt/slot/bilstm.pt
python test_slot.py --test_file "${1}" --ckpt_path "${2}"