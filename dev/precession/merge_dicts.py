"""
To merge several output dictionaries into a single one:

python merge_dicts.py --out-file analytical_angles_mismatch_cut_before_merger.json mismatch_results/analytical_angles_mismatch_cut_before_merger_*json

python merge_dicts.py --out-file analytical_angles_mismatch_cut_t_zero.json mismatch_results/analytical_angles_mismatch_cut_t_zero_*json
"""

import json

import argparse

parser = argparse.ArgumentParser(description='Computes the match between the approximated version and the true version of the angles')
parser.add_argument('--out-file', type=str, required = False, default = None,
	help='JSON file to write the output')

args, filenames = parser.parse_known_args()

res_dict = None
for f in filenames:
	with open(f, 'r') as f:
		tmp_dict = json.load(f)

	if not res_dict:
		res_dict = {k: [] for k in tmp_dict.keys()}
	
	if res_dict:
		for k,v in tmp_dict.items():
			res_dict[k].extend(v)

if args.out_file:
	with open(args.out_file, 'w') as f:
		json.dump(res_dict, f)
