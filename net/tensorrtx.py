"""
tensorrtx 代码
created by 李龙 2021/1
最终修改 by 李龙 2021/1/15
添加注释 by 林顺喆 2022/12/26
"""

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
from pycuda.compiler import SourceModule


class YoLov5TRT(object):
    """
    YOLOv5类，用于执行TensorRT推理操作
    """

    def __init__(self, engine_file_path):
        # 生成一个pycuda的context对象（使用第0个CUDA设备，即GPU0，一般是独显）
        self.ctx = cuda.Device(0).make_context()
        # 生成cuda流，即一个GPU上的操作队列
        stream = cuda.Stream(0)
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

        # 创建TensorRT的Runtime
        runtime = trt.Runtime(TRT_LOGGER)

        # 反序列化生成engine
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        # IExecutionContext
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        # 遍历binding
        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            # 计算内存空间大小
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size 
            # 内存空间的数据类型
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # 为主机分配 页锁定内存（避免进入磁盘模拟的低速虚拟内存）供numpy对象使用
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配设备内存，与页锁定内存等大
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # 将分配的设备内存记录于bindings列表
            # cuda_mem的类型是pycuda.driver.DeviceAllocation，可通过强制转换其为int类型，以取得在IEContext中的下标
            bindings.append(int(cuda_mem))

            # 根据binding是input或output类型，将内存记录于不同的列表
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = 1

    # 推理方法
    def infer(self, batch_input_image,layer):
        # 激活
        # 该方法将ctx置于context栈的栈顶
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # 将输入的图像传入host主机内存
        np.copyto(host_inputs[0], batch_input_image.ravel())
        # 将主机上存储的输入数据传入GPU
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # 进行推理过程
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # 将推理得到的输入数据从GPU拉回到主机
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        if layer == 3:
            cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
            cuda.memcpy_dtoh_async(host_outputs[2], cuda_outputs[2], stream)
        # 等待所有CUDA操作停止，然后继续
        stream.synchronize()
        # 将self弹出，不再激活 注意pop是static方法
        self.ctx.pop()
        output = host_outputs
        return output



    def __del__(self):
        # 将context栈顶的ICudaContext对象弹出，不再激活
        self.ctx.pop()
        self.stream.is_done()
        self.ctx.detach()



