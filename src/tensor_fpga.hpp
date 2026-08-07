#ifndef TENSOR_FPGA_HPP_
#define TENSOR_FPGA_HPP_

#ifndef USE_CPU_ONLY

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

#include "tensor.hpp"
#include "weight.hpp"

namespace swan {

// ---- Weight residency ----
// Per-layer cl::Buffer holding attention/FFN weight matrices resident in
// FPGA memory. Populated once at startup by UploadWeightsFPGA().
struct WeightsFPGA {
  cl::Buffer attn_wq[kNumLayers];   // 288x288
  cl::Buffer attn_wk[kNumLayers];   // 288x288
  cl::Buffer attn_wv[kNumLayers];   // 288x288
  cl::Buffer attn_wo[kNumLayers];   // 288x288
  cl::Buffer ffn_w1[kNumLayers];    // 768x288
  cl::Buffer ffn_w3[kNumLayers];    // 768x288

  // FFN w2 is [288 out][768 in]. Existing kernel needs COLSIZE=288 input,
  // so split columns of w2 into 3 sub-matrices of [288 out][288 col].
  cl::Buffer ffn_w2_c0[kNumLayers];  // w2[:, 0..287]
  cl::Buffer ffn_w2_c1[kNumLayers];  // w2[:, 288..575]
  cl::Buffer ffn_w2_c2[kNumLayers];  // w2[:, 576..767]

  // tok_emb_table is [32000 out][288 in]. Existing kernel produces up to
  // MAX_ROW_SIZE=768 outputs per call, so split into 42 row chunks of 768.
  // Last chunk padded with zeros (32000 = 41*768 + 512, chunk 41 has 256 zero rows).
  static constexpr int kVocabChunks    = 42;
  static constexpr int kVocabChunkRows = 768;
  cl::Buffer tok_emb[kVocabChunks];  // each 768x288
};

void UploadWeightsFPGA(WeightsFPGA& out, const Weights& w,
                       const Tensor2dTok& tok_emb_table,
                       cl::Context context, cl::CommandQueue q);

// FFN w2 on FPGA via column chunking (3 kernel calls + host sum).
void MatmulFFNw2FPGA(Tensor1d& out, const Tensor1dFFNB& in,
                     cl::Buffer buffer_w_c0, cl::Buffer buffer_w_c1,
                     cl::Buffer buffer_w_c2,
                     cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                     float* ptr_a, float* ptr_result,
                     cl::Buffer buffer_a, cl::Buffer buffer_result);

// Final vocab projection on FPGA via row chunking (42 kernel calls).
void MutmulVocabFPGA(Tensor1dLogits& out, const Tensor1d& in,
                     const WeightsFPGA& wfpga,
                     cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                     float* ptr_a, float* ptr_result,
                     cl::Buffer buffer_a, cl::Buffer buffer_result);

void AddFPGA(Tensor1d& out, const Tensor1d& in, float a, cl::CommandQueue q,
             cl::Kernel kernel_add, float* ptr_a, float* ptr_b,
             float* ptr_result, cl::Buffer buffer_a, cl::Buffer buffer_b,
             cl::Buffer buffer_result);
void MulFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a,
             cl::CommandQueue q, cl::Kernel kernel_mul, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result);

void AddFPGA(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs,
             cl::CommandQueue q, cl::Kernel kernel_add, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result);
void MulFPGA(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
             const Tensor1dFFNB& rhs, cl::CommandQueue q, cl::Kernel kernel_mul,
             float* ptr_a, float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result);

void MatmulFPGA(Tensor1d& out, const Tensor1d& in, const Tensor2dAttn& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result);
void MatmulFPGA(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNA& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result);
void MatmulFPGA(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNB& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result);

void MatmulParaFPGA(Tensor1d& out1, const Tensor1d& in1, const Tensor2dAttn& w1,
				Tensor1d& out2, /*const Tensor1d& in2,*/ const Tensor2dAttn& w2,
				Tensor1d& out3, /*const Tensor1d& in3,*/ const Tensor2dAttn& w3,
                cl::CommandQueue& q1, cl::Kernel kernel_matmul_1, float* ptr_a1, float* ptr_b1, float* ptr_result1, cl::Buffer buffer_a1, cl::Buffer buffer_b1, cl::Buffer buffer_result1,
                cl::CommandQueue& q2, cl::Kernel kernel_matmul_2, float* ptr_a2, float* ptr_b2, float* ptr_result2, cl::Buffer buffer_a2, cl::Buffer buffer_b2, cl::Buffer buffer_result2,
                cl::CommandQueue& q3, cl::Kernel kernel_matmul_3, float* ptr_a3, float* ptr_b3, float* ptr_result3, cl::Buffer buffer_a3, cl::Buffer buffer_b3, cl::Buffer buffer_result3);

// Profiling
void PrintMatmulProfile();

// Added: weight-resident variants. Weight is already in FPGA memory
// (see UploadWeightsFPGA); only the input vector is sent each call.
void MatmulPt288x288(Tensor1d& out, const Tensor1d& in,
                cl::Buffer buffer_w,
                cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                float* ptr_a, float* ptr_result,
                cl::Buffer buffer_a, cl::Buffer buffer_result);
void MatmulPt288x768(Tensor1dFFNB& out, const Tensor1d& in,
                cl::Buffer buffer_w,
                cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                float* ptr_a, float* ptr_result,
                cl::Buffer buffer_a, cl::Buffer buffer_result);

void RMSNormFPGA(Tensor1d& out, const Tensor1d& in, const Tensor1d& w,
                 cl::CommandQueue q, cl::Kernel kernel_rmsnorm, float* ptr_a,
                 float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_b, cl::Buffer buffer_result);
void SoftmaxFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, int max_pos,
                 cl::CommandQueue q, cl::Kernel kernel_softmax, float* ptr_a,
                 float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_result);

void RoPEFPGA(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
              const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
              const Tensor1dSinCos& sin_vec, int head_begin, int head_size,
              cl::CommandQueue q, cl::Kernel kernel_rope, float* ptr_a,
              float* ptr_b, float* ptr_c, float* ptr_d, float* ptr_result,
              float* ptr_result2, cl::Buffer buffer_a, cl::Buffer buffer_b,
              cl::Buffer buffer_c, cl::Buffer buffer_d,
              cl::Buffer buffer_result, cl::Buffer buffer_result2);

} // namespace swan

#endif // USE_CPU_ONLY

#endif // TENSOR_FPGA_HPP_
