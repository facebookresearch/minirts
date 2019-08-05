# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import json


def get_all_files(root, file_extension):
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(file_extension):
                files.append(os.path.join(folder, f))
    return files


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='public/data')
args = parser.parse_args()

# folder = os.path.abspath(args.root)
files = get_all_files(args.root, '.json')

files = [f.replace('public', '.') for f in files]

with open('./public/files.json', 'w') as outfile:
    json.dump(files, outfile, indent=4)

print( f'Found {len(files)} files. Saved in "./public/files.json"' )
