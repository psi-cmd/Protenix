export HOME=`pwd ..`/..
export CPATH=/opt/cutlass/include:$CPATH
export CPLUS_INCLUDE_PATH=/opt/cutlass/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/opt/cutlass/lib:$LD_LIBRARY_PATH

ln -s /workspace/data/ ./data
ln -s /workspace/release_data/ ./release_data

pip install -e .
