# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import pprint

folder = os.path.abspath(os.path.dirname(__file__))
params = [f for f in os.listdir(folder) if f.endswith('pt.params')]

inst_dict_path = os.path.join(os.path.dirname(folder), 'data', 'dataset', 'dict.pt')

for p in params:
    param_dict = pickle.load(open(p, 'rb'))
    param_dict['args'].inst_dict_path = inst_dict_path
    # pprint.pprint(param_dict)
    pickle.dump(param_dict, open(p, 'wb'))
