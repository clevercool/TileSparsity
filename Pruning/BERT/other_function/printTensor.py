from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
s = print_tensors_in_checkpoint_file(file_name='./mrpc_output/model.ckpt-11', tensor_name='', all_tensors=True)
