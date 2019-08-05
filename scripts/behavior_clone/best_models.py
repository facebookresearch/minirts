# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_root = os.path.join(root, 'pretrained_models')

# executor_root = '/private/home/hengyuan/minirts/scripts/behavior_clone/new_sweep_executor'
best_executors = {
    'rnn': 'executor_rnn.pt',
    'rnn_nohist': 'executor_rnn_nohist.pt',
    'bow': 'executor_bow.pt',
    'onehot': 'executor_onehot.pt',
    'zero': 'executor_zero.pt',
}
for key, val in best_executors.items():
    best_executors[key] = os.path.join(model_root, val)

# best_executors['zero'] = best_zero_executor


coach_root = '/private/home/hengyuan/minirts/scripts/behavior_clone/new_sweep_coach/coach_repro1'
best_coaches = {
    'rnn50': 'coach_rnn50.pt',
    'rnn250': 'coach_rnn250.pt',
    'rnn500': 'coach_rnn500.pt',
    'bow50': 'coach_bow50.pt',
    'bow250': 'coach_bow250.pt',
    'bow500': 'coach_bow500.pt',
    'onehot50': 'coach_onehot50.pt',
    'onehot250': 'coach_onehot250.pt',
    'onehot500': 'coach_onehot500.pt',
}
for key, val in best_coaches.items():
    best_coaches[key]= os.path.join(model_root, val)


if __name__ == '__main__':
    for k, v in best_executors.items():
        print(v)
        assert os.path.exists(v)

    for k, v in best_coaches.items():
        print(v)
        assert os.path.exists(v)
