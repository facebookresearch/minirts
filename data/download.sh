# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
wget https://dl.fbaipublicfiles.com/minirts/data.tgz
echo "extracting files, this may take a while"
tar -zxf data.tgz
echo "done!"
echo "creating a symlink at visual/public"
pwd=$(pwd)
ln -s $(pwd)/replays_json ../visual/public/data
