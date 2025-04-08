
```
AI_framework-dev
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        ├─ nodeids
│        └─ stepwise
├─ dev
│  ├─ .pytest_cache
│  │  ├─ CACHEDIR.TAG
│  │  ├─ README.md
│  │  └─ v
│  │     └─ cache
│  │        ├─ lastfailed
│  │        ├─ nodeids
│  │        └─ stepwise
│  ├─ activations
│  │  ├─ activations.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ activations.cpython-312.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ backend
│  │  ├─ backend_ops
│  │  │  ├─ activations
│  │  │  │  ├─ activations.cpp
│  │  │  │  ├─ activations.h
│  │  │  │  ├─ activations.py
│  │  │  │  ├─ activations.pyd
│  │  │  │  ├─ activations_cuda.cu
│  │  │  │  ├─ binding_activations.cpp
│  │  │  │  ├─ build
│  │  │  │  │  └─ lib.win-amd64-cpython-312
│  │  │  │  │     ├─ activations_cuda.cp312-win_amd64.exp
│  │  │  │  │     ├─ activations_cuda.cp312-win_amd64.lib
│  │  │  │  │     └─ activations_cuda.cp312-win_amd64.pyd
│  │  │  │  ├─ readme.md
│  │  │  │  ├─ setup.py
│  │  │  │  ├─ tests
│  │  │  │  │  ├─ activation_test.py
│  │  │  │  │  ├─ __init__.py
│  │  │  │  │  └─ __pycache__
│  │  │  │  │     ├─ activation_test.cpython-312-pytest-8.3.5.pyc
│  │  │  │  │     └─ __init__.cpython-312.pyc
│  │  │  │  ├─ __init__.py
│  │  │  │  └─ __pycache__
│  │  │  │     ├─ activation_test.cpython-312-pytest-8.3.5.pyc
│  │  │  │     └─ __init__.cpython-312.pyc
│  │  │  ├─ convolution
│  │  │  │  ├─ binding_convolution.cpp
│  │  │  │  ├─ convolution.cpp
│  │  │  │  └─ convolution.pyd
│  │  │  ├─ flatten
│  │  │  │  ├─ binding_flatten.cpp
│  │  │  │  ├─ flatten.cpp
│  │  │  │  └─ flatten.pyd
│  │  │  ├─ losses
│  │  │  │  ├─ binding_losses.cpp
│  │  │  │  ├─ losses.cpp
│  │  │  │  └─ losses.pyd
│  │  │  ├─ metrics
│  │  │  │  ├─ binding_metrics.cpp
│  │  │  │  ├─ metrics.cpp
│  │  │  │  └─ metrics.pyd
│  │  │  ├─ node_no_use
│  │  │  │  ├─ binding_node.cpp
│  │  │  │  ├─ C++ 로 구현한 node 는 잠정 미사용.md
│  │  │  │  ├─ node.h
│  │  │  │  └─ node.pyd
│  │  │  ├─ operaters
│  │  │  │  ├─ binding_operations_matrix.cpp
│  │  │  │  ├─ build
│  │  │  │  │  └─ lib.win-amd64-cpython-312
│  │  │  │  │     ├─ matrix_ops.cp312-win_amd64.exp
│  │  │  │  │     ├─ matrix_ops.cp312-win_amd64.lib
│  │  │  │  │     ├─ matrix_ops.cp312-win_amd64.pyd
│  │  │  │  │     ├─ operations_matrix_cuda.cp312-win_amd64.exp
│  │  │  │  │     ├─ operations_matrix_cuda.cp312-win_amd64.lib
│  │  │  │  │     └─ operations_matrix_cuda.cp312-win_amd64.pyd
│  │  │  │  ├─ matrix_ops.cu
│  │  │  │  ├─ operations_matrix.cpp
│  │  │  │  ├─ operations_matrix.h
│  │  │  │  ├─ operations_matrix.pyd
│  │  │  │  ├─ setup.py
│  │  │  │  ├─ tests
│  │  │  │  │  ├─ operations_matrix_cuda_test.py
│  │  │  │  │  ├─ test.py
│  │  │  │  │  ├─ test2.py
│  │  │  │  │  ├─ test_matrix_ops.py
│  │  │  │  │  ├─ __init__.py
│  │  │  │  │  └─ __pycache__
│  │  │  │  │     ├─ operations_matrix_cuda_test.cpython-312-pytest-8.3.5.pyc
│  │  │  │  │     ├─ test_matrix_ops.cpython-312-pytest-8.3.5.pyc
│  │  │  │  │     └─ __init__.cpython-312.pyc
│  │  │  │  ├─ __init__.py
│  │  │  │  └─ __pycache__
│  │  │  │     └─ __init__.cpython-312.pyc
│  │  │  ├─ optimizers
│  │  │  │  ├─ binding_optimizers.cpp
│  │  │  │  ├─ optimizers.cpp
│  │  │  │  └─ optimizers.pyd
│  │  │  ├─ pooling
│  │  │  │  ├─ binding_pooling.cpp
│  │  │  │  ├─ pooling.cpp
│  │  │  │  └─ pooling.pyd
│  │  │  └─ recurrent
│  │  │     ├─ binding_recurrent.cpp
│  │  │     ├─ recurrent.cpp
│  │  │     └─ recurrent.pyd
│  │  └─ cudabackend
│  │     └─ operaters
│  │        ├─ matrix_operations.cu
│  │        └─ parallel_operations.cu
│  ├─ backpropagation
│  ├─ decorators
│  │  ├─ validation_decorators.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ validation_decorators.cpython-312.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ graph_engine
│  │  ├─ activations.py
│  │  ├─ core_graph.py
│  │  ├─ core_ops.py
│  │  ├─ graph_utils.py
│  │  ├─ node.py
│  │  ├─ tests
│  │  │  ├─ activations_test.py
│  │  │  ├─ cal_graph_test.py
│  │  │  ├─ node_list_link_test.py
│  │  │  ├─ test_core_graph.py
│  │  │  ├─ __init__.py
│  │  │  └─ __pycache__
│  │  │     ├─ activations_test.cpython-312-pytest-8.3.5.pyc
│  │  │     ├─ cal_graph_test.cpython-312-pytest-8.3.5.pyc
│  │  │     ├─ node_list_link_test.cpython-312-pytest-8.3.5.pyc
│  │  │     ├─ test_core_graph.cpython-312-pytest-8.3.5.pyc
│  │  │     └─ __init__.cpython-312.pyc
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ activations.cpython-312.pyc
│  │     ├─ cal_graph.cpython-311.pyc
│  │     ├─ cal_graph.cpython-312.pyc
│  │     ├─ core_graph.cpython-312.pyc
│  │     ├─ core_ops.cpython-312.pyc
│  │     ├─ graph_utils.cpython-312.pyc
│  │     ├─ node.cpython-312.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ layers
│  │  ├─ activations.py
│  │  ├─ Conv2D.py
│  │  ├─ dense.py
│  │  ├─ dense_cuda.py
│  │  ├─ flatten.py
│  │  ├─ input_layer.py
│  │  ├─ layer.py
│  │  ├─ operations.py
│  │  ├─ output_layer.py
│  │  ├─ pooling.py
│  │  ├─ Rnn.py
│  │  ├─ tests
│  │  │  ├─ dense_test.py
│  │  │  ├─ __init__.py
│  │  │  └─ __pycache__
│  │  │     └─ dense_test.cpython-312-pytest-8.3.5.pyc
│  │  └─ __pycache__
│  │     ├─ activations.cpython-312.pyc
│  │     ├─ dense_cuda.cpython-312.pyc
│  │     ├─ flatten.cpython-312.pyc
│  │     ├─ layer.cpython-311.pyc
│  │     ├─ layer.cpython-312.pyc
│  │     └─ pooling.cpython-312.pyc
│  ├─ losses
│  │  ├─ losses.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ losses.cpython-312.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ metrics
│  │  ├─ accuracy_metrics.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ accuracy_metrics.cpython-312.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ models
│  │  ├─ model.py
│  │  ├─ OPL.py
│  │  ├─ sequential.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ model.cpython-311.pyc
│  │     ├─ model.cpython-312.pyc
│  │     ├─ OPL.cpython-312.pyc
│  │     ├─ sequential.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ node
│  │  ├─ node.py
│  │  ├─ node2.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ node.cpython-311.pyc
│  │     ├─ node.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ ops
│  │  ├─ node.py
│  │  └─ operation.py
│  ├─ optimizers
│  │  ├─ sgd.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ sgd.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ pytest.ini
│  ├─ regularizers
│  │  ├─ regularizers.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ regularizers.cpython-311.pyc
│  │     ├─ regularizers.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ runtest.py
│  ├─ saving
│  │  └─ serialization.py
│  ├─ tensor
│  │  └─ tensor.py
│  ├─ tests
│  │  ├─ conftest.py
│  │  ├─ test_setup.py
│  │  ├─ __init__.py
│  │  ├─ __pycache__
│  │  │  ├─ conftest.cpython-312-pytest-8.3.5.pyc
│  │  │  ├─ test_setup.cpython-312-pytest-8.3.5.pyc
│  │  │  ├─ test_setup.cpython-312.pyc
│  │  │  └─ __init__.cpython-312.pyc
│  │  └─ 진단코드.py
│  ├─ trainer
│  │  ├─ data_adapters
│  │  │  ├─ data_adapter_util.py
│  │  │  └─ __pycache__
│  │  │     └─ data_adapter_util.cpython-312.pyc
│  │  ├─ trainer.py
│  │  └─ __pycache__
│  │     ├─ trainer.cpython-311.pyc
│  │     └─ trainer.cpython-312.pyc
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ __init__.cpython-311.pyc
│     └─ __init__.cpython-312.pyc
├─ development_history.md
├─ docs
├─ examples
├─ README.md
├─ tests
│  ├─ ai_test_code
│  │  ├─ keras_debug.py
│  │  ├─ keras_image_sequential.py
│  │  └─ keras_model_config.py
│  ├─ backend_test_code
│  │  ├─ activation_test.py
│  │  ├─ mul_matrix.py
│  │  └─ __pycache__
│  │     └─ activation_test.cpython-312-pytest-8.3.5.pyc
│  ├─ class_test_code
│  │  ├─ new_obj_return.py
│  │  ├─ typing_cast1.py
│  │  └─ typing_cast2.py
│  ├─ cuda_code
│  │  ├─ cuBLAS
│  │  │  ├─ cuBLAS_mat_mul.cu
│  │  │  ├─ cuBLAS_mat_mul.exe
│  │  │  ├─ cuBLAS_mat_mul.exp
│  │  │  └─ cuBLAS_mat_mul.lib
│  │  ├─ cuda_tiling
│  │  │  ├─ cuda_tiling.cu
│  │  │  ├─ cuda_tiling.exe
│  │  │  ├─ cuda_tiling.exp
│  │  │  ├─ cuda_tiling.lib
│  │  │  ├─ cuda_tiling_time_check.cu
│  │  │  ├─ cuda_tiling_time_check.exe
│  │  │  ├─ cuda_tiling_time_check.exp
│  │  │  ├─ cuda_tiling_time_check.lib
│  │  │  ├─ three_time.cu
│  │  │  ├─ three_time.exe
│  │  │  ├─ three_time.exp
│  │  │  └─ three_time.lib
│  │  ├─ cuDNN
│  │  │  └─ cuDNN_con.cu
│  │  ├─ memory_check
│  │  │  ├─ cal_matrix_size
│  │  │  │  ├─ cal_matrix_size.cu
│  │  │  │  ├─ cal_matrix_size.exe
│  │  │  │  ├─ cal_matrix_size.exp
│  │  │  │  └─ cal_matrix_size.lib
│  │  │  └─ matrix_size_checkl
│  │  │     ├─ matrix_size_check.cu
│  │  │     ├─ matrix_size_check.exe
│  │  │     ├─ matrix_size_check.exp
│  │  │     └─ matrix_size_check.lib
│  │  ├─ pybinding_cuda
│  │  │  ├─ build
│  │  │  │  └─ lib.win-amd64-cpython-312
│  │  │  │     ├─ cuda_add.cp312-win_amd64.exp
│  │  │  │     ├─ cuda_add.cp312-win_amd64.lib
│  │  │  │     └─ cuda_add.cp312-win_amd64.pyd
│  │  │  ├─ cuda_add.cp312-win_amd64.pyd
│  │  │  ├─ cuda_add.cu
│  │  │  ├─ setup.py
│  │  │  └─ test.py
│  │  └─ time_check
│  │     ├─ mat_mul_time_check
│  │     │  ├─ mat_mul_time_check.cu
│  │     │  ├─ mat_mul_time_check.exe
│  │     │  ├─ mat_mul_time_check.exp
│  │     │  └─ mat_mul_time_check.lib
│  │     ├─ time_check.cu
│  │     ├─ time_check.exp
│  │     └─ time_check.lib
│  ├─ GPU_test
│  │  ├─ block_size.cu
│  │  ├─ block_size.exe
│  │  ├─ block_size.exp
│  │  ├─ block_size.lib
│  │  ├─ GPU.cu
│  │  ├─ GPU.exe
│  │  ├─ GPU.exp
│  │  └─ GPU.lib
│  ├─ my_framework_code
│  │  ├─ activation_layer_code
│  │  │  └─ model_add_activation.py
│  │  ├─ backend_check_code
│  │  │  ├─ check_back.py
│  │  │  ├─ check_call_method.py
│  │  │  ├─ check_cal_dense.py
│  │  │  ├─ check_config.py
│  │  │  ├─ check_fit.py
│  │  │  ├─ check_loss.py
│  │  │  ├─ check_metrics.py
│  │  │  ├─ check_sys_path.py
│  │  │  ├─ check_weight_update.py
│  │  │  └─ concat_node_list.py
│  │  ├─ batch_size_code
│  │  │  └─ model_add_batch_size.py
│  │  ├─ cnn_code
│  │  │  ├─ keras_cnn.py
│  │  │  ├─ test_cnn.py
│  │  │  └─ __pycache__
│  │  │     └─ test_cnn.cpython-312-pytest-8.3.5.pyc
│  │  ├─ dense_cal_check_code
│  │  │  ├─ cal_dense.py
│  │  │  └─ keras_비교.py
│  │  ├─ OPL_code
│  │  │  ├─ keras_functional_api.py
│  │  │  ├─ OPL_test.py
│  │  │  └─ __pycache__
│  │  │     └─ OPL_test.cpython-312-pytest-8.3.5.pyc
│  │  └─ rnn_code
│  │     ├─ keras_rnn.py
│  │     ├─ test_rnn.py
│  │     └─ __pycache__
│  │        └─ test_rnn.cpython-312-pytest-8.3.5.pyc
│  ├─ object_test_code
│  │  ├─ from_config.py
│  │  └─ get_config.py
│  └─ wheel_build_test
│     └─ setup.py
├─ WhyAIFrameWork.md
├─ __pycache__
│  └─ csv.cpython-312.pyc
└─ 포트폴리오.md

```