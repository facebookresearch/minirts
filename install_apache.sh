# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
rm -rf apache
mkdir apache
cd apache

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH

mkdir apr
mkdir apr_util
mkdir httpd

wget https://www-eu.apache.org/dist//apr/apr-1.7.0.tar.gz
tar -zxf apr-1.7.0.tar.gz
cd $SCRIPTPATH/apr-1.7.0
./configure --prefix=$SCRIPTPATH/apr
make
make install

cd $SCRIPTPATH
wget https://www-eu.apache.org/dist//apr/apr-util-1.6.1.tar.gz
tar -zxf apr-util-1.6.1.tar.gz
cd $SCRIPTPATH/apr-util-1.6.1
./configure --prefix=$SCRIPTPATH/apr_util --with-apr=$SCRIPTPATH/apr
make
make install

cd $SCRIPTPATH
wget https://www-us.apache.org/dist//httpd/httpd-2.4.41.tar.gz
tar -zxf httpd-2.4.41.tar.gz
cd $SCRIPTPATH/httpd-2.4.41
./configure --prefix=$SCRIPTPATH/httpd \
            --with-apr=$SCRIPTPATH/apr \
            --with-apr-util=$SCRIPTPATH/apr_util
make
make install
