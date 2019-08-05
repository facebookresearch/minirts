wget https://dl.fbaipublicfiles.com/minirts/data.tgz
echo "extracting files, this may take a while"
tar -zxf data.tgz
echo "done!"
echo "creating a symlink at visual/public"
pwd=$(pwd)
ln -s $(pwd)/replays_json ../visual/public/data
