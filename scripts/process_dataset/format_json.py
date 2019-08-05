# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import common_utils.global_consts as gc


def print_as_table(n_row, n_col, lines):
    table = []
    total_num = 0
    for r in range(n_row):
        row = []
        for c in range(n_col):
            idx = r * n_col + c
            # print(idx)
            # if idx >= len(lines):
            #     assert n_row == 1
            #     break
            # val = row.append(lines[idx])
            val = lines[idx]
            # print(val)
            if c != 0:
                val = val.strip()
            else:
                val = val.rstrip()
            total_num += 1
            row.append(val)
        # print(row)
        table.append(' '.join(row))
        # print(table)
    assert total_num == len(lines)
    return '\n'.join(table)


def format_json(json_file):
    output = ''
    lines = json_file.readlines()
    i = 0
    row = -1
    col = -1
    while i < len(lines):
        if row != -1 and col != -1:
            num = row * col
            table = print_as_table(row, col, lines[i : i+num])
            output += (table + '\n')
            i += num
            # print(lines[i])
            assert lines[i].strip() == '],' or lines[i].strip() == ']'
            row = -1
            col = -1

        line = lines[i]
        if '"cons_count": [' in line:
            row = 1
            col = len(gc.UnitTypes)
        elif '"terrain": [' in line:
            row = 32
            col = 32
        elif '"visibility": [' in line:
            row = 32
            col = 32
        # else:
        output += line# + '\n')
        i += 1
        # continue

    json_file.seek(0)
    original_json = json.load(json_file) # open(filename, 'r'))
    formatted_json = json.loads(output)
    if original_json != formatted_json:
        print('Error: mismatch between original and formatted:', filename)
        assert False

    return output
