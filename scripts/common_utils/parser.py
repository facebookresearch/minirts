# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
from collections import OrderedDict


class Parser:
    def __init__(self):
        self.arg_parsers = OrderedDict()

    def add_parser(self, name, parser):
        if name in self.arg_parsers:
            print('Error: duplicate key for parser:', name)

        self.arg_parsers[name] = parser

    def parse(self):
        args = {}
        remain = sys.argv
        for name, parser in self.arg_parsers.items():
            print('parsing:', name)
            arg, remain = parser.parse_known_args(args=remain)
            args[name] = arg

        if len(remain) > 1:
            print('Warning: extra key for parser:', remain)
            assert False

        return args

    def log(self):
        print(self.arg_parsers)
