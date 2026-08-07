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
            cl::Kernel kernel_matmul_pt_288x,
            float* ptr_a, float* ptr_b, float* ptr_result,
            cl::Buffer buffer_a, cl::Buffer buffer_b, cl::Buffer buffer_result
#endif // USE_CPU_ONLY
);

} // namespace swan

#endif // DECODE_HPP_
