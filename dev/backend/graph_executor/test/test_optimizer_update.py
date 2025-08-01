import numpy as np
import cupy as cp
import graph_executor as ge

def test_optimizer_update():
    # 파라미터 및 그래디언트 초기화
    param = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    grad = cp.array([0.1, 0.2, 0.3], dtype=cp.float32)

    # 옵티마이저 타입 설정 (SGD, MOMENTUM, ADAM 가능)
    opt_type = ge.OptimizerType.SGD
    learning_rate = 0.1
    size = param.size

    # 선택적으로 사용하는 velocity, m, v 버퍼 (MOMENTUM/ADAM용)
    velocity = cp.zeros_like(param)  # for momentum
    m = cp.zeros_like(param)        # for adam
    v = cp.zeros_like(param)        # for adam
    timestep = 1                    # for adam

    print("Before update:", cp.asnumpy(param))

    ge.optimizer_update(
        param_ptr=param.data.ptr,
        grad_ptr=grad.data.ptr,
        velocity_ptr=velocity.data.ptr,
        m_ptr=m.data.ptr,
        v_ptr=v.data.ptr,
        lr=learning_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        size=size,
        opt_type=opt_type,
        timestep=timestep
    )

    print("After update :", cp.asnumpy(param))

if __name__ == "__main__":
    test_optimizer_update()
