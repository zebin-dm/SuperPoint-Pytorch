work_path=$(dirname $(dirname $(pwd)))
echo "work path: ${work_path}"
root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"
export PYTHONPATH=${work_path}:${PYTHONPATH}