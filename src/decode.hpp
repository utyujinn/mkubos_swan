#ifndef DECODE_HPP_
#define DECODE_HPP_

#include "context.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#include "tensor_fpga.hpp"

#include <CL/cl2.hpp>
#endif // USE_CPU_ONLY

namespace swan {

void Decode(int tok, int pos, const Tensor1d& ctx_input,
            Tensor3dCache& ctx_k_cache, Tensor3dCache& ctx_v_cache,
            Tensor1d& ctx_final_norm, const Weights& w
#ifndef USE_CPU_ONLY
            ,
            cl::CommandQueue q,
            cl::CommandQueue q1, cl::CommandQueue q2, cl::CommandQueue q3,
            cl::Kernel kernel_matmul_1, cl::Kernel kernel_matmul_2, cl::Kernel kernel_matmul_3,
            cl::Kernel kernel_mul,
            cl::Kernel kernel_rmsnorm, cl::Kernel kernel_softmax,
            cl::Kernel kernel_add, cl::Kernel kernel_rope,
            float* ptr_a, float* ptr_b, float* ptr_c, float* ptr_d, float* ptr_result, float* ptr_result2,
            float* ptr_a_1, float* ptr_b_1, float* ptr_result_1,
            float* ptr_a_2, float* ptr_b_2, float* ptr_result_2,
            float* ptr_a_3, float* ptr_b_3, float* ptr_result_3,
            cl::Buffer buffer_a, cl::Buffer buffer_b,
            cl::Buffer buffer_c, cl::Buffer buffer_d, cl::Buffer buffer_result,
            cl::Buffer buffer_result2,
            cl::Buffer buffer_a_1, cl::Buffer buffer_b_1, cl::Buffer buffer_result_1,
            cl::Buffer buffer_a_2, cl::Buffer buffer_b_2, cl::Buffer buffer_result_2,
            cl::Buffer buffer_a_3, cl::Buffer buffer_b_3, cl::Buffer buffer_result_3
#endif // USE_CPU_ONLY
);

} // namespace swan

#endif // DECODE_HPP_
