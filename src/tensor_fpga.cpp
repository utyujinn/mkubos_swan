#include "tensor_fpga.hpp"

#ifndef USE_CPU_ONLY

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace swan {

// ---- Profiling accumulators (nanoseconds) ----
static double g_prof_pack_ns   = 0;
static double g_prof_h2d_ns    = 0;
static double g_prof_kernel_ns = 0;
static double g_prof_d2h_ns    = 0;
static double g_prof_unpack_ns = 0;
static long long g_prof_calls_288 = 0;
static long long g_prof_calls_768 = 0;
static long long g_prof_calls_w2 = 0;
static long long g_prof_calls_vocab = 0;

static inline double ns_since(const std::chrono::high_resolution_clock::time_point& t0) {
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::high_resolution_clock::now() - t0).count();
}

void PrintMatmulProfile() {
  double total = g_prof_pack_ns + g_prof_h2d_ns + g_prof_kernel_ns
               + g_prof_d2h_ns + g_prof_unpack_ns;
  long long calls = g_prof_calls_288 + g_prof_calls_768;
  if (calls == 0 || total <= 0) return;
  auto pct = [&](double x){ return 100.0 * x / total; };
  auto ms  = [](double ns){ return ns / 1e6; };
  printf("\n--- MatmulPt profile ---\n");
  printf("  calls  288x288 / 288x768 / w2 / vocab : %lld / %lld / %lld / %lld\n",
         g_prof_calls_288, g_prof_calls_768,
         g_prof_calls_w2, g_prof_calls_vocab);
  printf("  pack   : %8.2f ms (%5.2f%%)\n", ms(g_prof_pack_ns),   pct(g_prof_pack_ns));
  printf("  h2d    : %8.2f ms (%5.2f%%)  <- enqueueMigrate H->D + finish\n",
         ms(g_prof_h2d_ns),    pct(g_prof_h2d_ns));
  printf("  kernel : %8.2f ms (%5.2f%%)  <- enqueueTask + finish\n",
         ms(g_prof_kernel_ns), pct(g_prof_kernel_ns));
  printf("  d2h    : %8.2f ms (%5.2f%%)  <- enqueueMigrate D->H + finish\n",
         ms(g_prof_d2h_ns),    pct(g_prof_d2h_ns));
  printf("  unpack : %8.2f ms (%5.2f%%)\n", ms(g_prof_unpack_ns), pct(g_prof_unpack_ns));
  printf("  TOTAL  : %8.2f ms  (avg %.3f ms / call)\n", ms(total), ms(total)/calls);
}

/* ---------------------------------  /
      Basic Arithmetic Operations
/  --------------------------------- */

// Add a scalar to each element of the input tensor.
void AddFPGA(Tensor1d& out, const Tensor1d& in, float a, cl::CommandQueue q,
             cl::Kernel kernel_add, float* ptr_a, float* ptr_b,
             float* ptr_result, cl::Buffer buffer_a, cl::Buffer buffer_b,
             cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    ptr_b[i] = a;
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_add.setArg(3, kDim);
  q.enqueueTask(kernel_add);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Multiply each element of the input tensor by a scalar.
void MulFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a,
             cl::CommandQueue q, cl::Kernel kernel_mul, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kSeqLen; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kSeqLen; i++) {
    ptr_b[i] = a;
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_mul.setArg(3, kSeqLen);
  q.enqueueTask(kernel_mul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kSeqLen; i++) {
    out[i] = ptr_result[i];
  }
}

// Add each element of the first input tensor to the corresponding element of
// the second input tensor.
void AddFPGA(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs,
             cl::CommandQueue q, cl::Kernel kernel_add, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = lhs[i];
  }
  for (int i = 0; i < kDim; i++) {
    ptr_b[i] = rhs[i];
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_add.setArg(3, kDim);
  q.enqueueTask(kernel_add);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Multiply each element of the first input tensor by the corresponding element
// of the second input tensor.
void MulFPGA(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
             const Tensor1dFFNB& rhs, cl::CommandQueue q, cl::Kernel kernel_mul,
             float* ptr_a, float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kFFNDim; i++) {
    ptr_a[i] = lhs[i];
  }
  for (int i = 0; i < kFFNDim; i++) {
    ptr_b[i] = rhs[i];
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_mul.setArg(3, kFFNDim);
  q.enqueueTask(kernel_mul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kFFNDim; i++) {
    out[i] = ptr_result[i];
  }
}

/* ---------------------------------  /
           Matrix Operations
/  --------------------------------- */

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dAttn [dim, dim] = Tensor1d [dim]
// out[i] = w[i,j] . in[j]
void MatmulFPGA(Tensor1d& out, const Tensor1d& in, const Tensor2dAttn& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    for (int j = 0; j < kDim; j++) {
      ptr_b[i * kDim + j] = w[i][j];
    }
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_matmul.setArg(3, kDim);
  kernel_matmul.setArg(4, kDim);
  q.enqueueTask(kernel_matmul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1dFFNB [ffn_dim] . Tensor2dFFNA [ffn_dim, dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
void MatmulFPGA(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNA& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kFFNDim; i++) {
    for (int j = 0; j < kDim; j++) {
      ptr_b[i * kDim + j] = w[i][j];
    }
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_matmul.setArg(3, kDim);
  kernel_matmul.setArg(4, kFFNDim);
  q.enqueueTask(kernel_matmul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kFFNDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dFFNB [dim, ffn_dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
void MatmulFPGA(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNB& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kFFNDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    for (int j = 0; j < kFFNDim; j++) {
      ptr_b[i * kFFNDim + j] = w[i][j];
    }
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_matmul.setArg(3, kFFNDim);
  kernel_matmul.setArg(4, kDim);
  q.enqueueTask(kernel_matmul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Matmul parallel 3
// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dFFNB [dim, ffn_dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dAttn [dim, dim] = Tensor1d [dim]
// out[i] = w[i,j] . in[j]
void MatmulParaFPGA(Tensor1d& out1, const Tensor1d& in1, const Tensor2dAttn& w1,
				Tensor1d& out2, /*const Tensor1d& in2,*/ const Tensor2dAttn& w2,
				Tensor1d& out3, /*const Tensor1d& in3,*/ const Tensor2dAttn& w3,
                cl::CommandQueue& q1, cl::Kernel kernel_matmul_1, float* ptr_a1, float* ptr_b1, float* ptr_result1, cl::Buffer buffer_a1, cl::Buffer buffer_b1, cl::Buffer buffer_result1,
                cl::CommandQueue& q2, cl::Kernel kernel_matmul_2, float* ptr_a2, float* ptr_b2, float* ptr_result2, cl::Buffer buffer_a2, cl::Buffer buffer_b2, cl::Buffer buffer_result2,
                cl::CommandQueue& q3, cl::Kernel kernel_matmul_3, float* ptr_a3, float* ptr_b3, float* ptr_result3, cl::Buffer buffer_a3, cl::Buffer buffer_b3, cl::Buffer buffer_result3
                ) {
  for (int i = 0; i < kDim; i++) {
    ptr_a1[i] = in1[i];
    ptr_a2[i] = in1[i];
    ptr_a3[i] = in1[i];
  }
  for (int i = 0; i < kDim; i++) {
    for (int j = 0; j < kDim; j++) {
      ptr_b1[i * kDim + j] = w1[i][j];
      ptr_b2[i * kDim + j] = w2[i][j];
      ptr_b3[i * kDim + j] = w3[i][j];
    }
  }
  q1.enqueueMigrateMemObjects({buffer_a1, buffer_b1}, 0);
  q2.enqueueMigrateMemObjects({buffer_a2, buffer_b2}, 0);
  q3.enqueueMigrateMemObjects({buffer_a3, buffer_b3}, 0);
  kernel_matmul_1.setArg(3, kDim);
  kernel_matmul_1.setArg(4, kDim);
  kernel_matmul_2.setArg(3, kDim);
  kernel_matmul_2.setArg(4, kDim);
  kernel_matmul_3.setArg(3, kDim);
  kernel_matmul_3.setArg(4, kDim);

  cl::Event event1, event2, event3;
  q1.enqueueTask(kernel_matmul_1, nullptr, &event1);
  q2.enqueueTask(kernel_matmul_2, nullptr, &event2);
  q3.enqueueTask(kernel_matmul_3, nullptr, &event3);
  q1.enqueueMigrateMemObjects({buffer_result1}, CL_MIGRATE_MEM_OBJECT_HOST);
  q2.enqueueMigrateMemObjects({buffer_result2}, CL_MIGRATE_MEM_OBJECT_HOST);
  q3.enqueueMigrateMemObjects({buffer_result3}, CL_MIGRATE_MEM_OBJECT_HOST);
  event1.wait();
  event2.wait();
  event3.wait();
  for (int i = 0; i < kDim; i++) {
    out1[i] = ptr_result1[i];
    out2[i] = ptr_result2[i];
    out3[i] = ptr_result3[i];
  }
}

// Upload all weight matrices to FPGA memory once at startup.
// After this, per-call MatmulPt288x* only needs to send the input vector.
void UploadWeightsFPGA(WeightsFPGA& out, const Weights& w,
                       const Tensor2dTok& tok_emb_table,
                       cl::Context context, cl::CommandQueue q) {
  const size_t sz_attn        = sizeof(Tensor2dAttn);  // 288*288*4 = ~324 KB
  const size_t sz_ffn         = sizeof(Tensor2dFFNA);  // 768*288*4 = ~864 KB
  const size_t sz_w2_chunk    = kDim * kDim * sizeof(float);  // 288*288*4
  const size_t sz_vocab_chunk = WeightsFPGA::kVocabChunkRows * kDim * sizeof(float);

  auto upload_one = [&](cl::Buffer& dst, const void* src, size_t bytes) {
    cl_int err;
    dst = cl::Buffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    void* p = q.enqueueMapBuffer(dst, CL_TRUE, CL_MAP_WRITE, 0, bytes,
                                 nullptr, nullptr, &err);
    std::memcpy(p, src, bytes);
    q.enqueueUnmapMemObject(dst, p);
  };

  std::vector<cl::Memory> all;
  std::vector<float> tmp(kDim * kDim);  // scratch for w2 repack

  for (int L = 0; L < kNumLayers; L++) {
    upload_one(out.attn_wq[L], w.attn_wq[L], sz_attn);
    upload_one(out.attn_wk[L], w.attn_wk[L], sz_attn);
    upload_one(out.attn_wv[L], w.attn_wv[L], sz_attn);
    upload_one(out.attn_wo[L], w.attn_wo[L], sz_attn);
    upload_one(out.ffn_w1[L],  w.ffn_w1[L],  sz_ffn);
    upload_one(out.ffn_w3[L],  w.ffn_w3[L],  sz_ffn);
    all.push_back(out.attn_wq[L]);
    all.push_back(out.attn_wk[L]);
    all.push_back(out.attn_wv[L]);
    all.push_back(out.attn_wo[L]);
    all.push_back(out.ffn_w1[L]);
    all.push_back(out.ffn_w3[L]);

    // Repack FFN w2 into 3 column-chunk sub-matrices.
    // w2 is [kDim out][kFFNDim in]. Chunk c contains columns [c*kDim, (c+1)*kDim).
    cl::Buffer* w2_targets[3] = {
        &out.ffn_w2_c0[L], &out.ffn_w2_c1[L], &out.ffn_w2_c2[L]};
    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < kDim; i++) {
        for (int j = 0; j < kDim; j++) {
          tmp[i * kDim + j] = w.ffn_w2[L][i][c * kDim + j];
        }
      }
      upload_one(*w2_targets[c], tmp.data(), sz_w2_chunk);
      all.push_back(*w2_targets[c]);
    }
  }

  // Repack tok_emb_table into 42 row chunks, last chunk zero-padded.
  std::vector<float> chunk_buf(WeightsFPGA::kVocabChunkRows * kDim);
  for (int c = 0; c < WeightsFPGA::kVocabChunks; c++) {
    std::fill(chunk_buf.begin(), chunk_buf.end(), 0.0f);
    for (int r = 0; r < WeightsFPGA::kVocabChunkRows; r++) {
      int global_row = c * WeightsFPGA::kVocabChunkRows + r;
      if (global_row >= kVocabSize) break;
      for (int j = 0; j < kDim; j++) {
        chunk_buf[r * kDim + j] = tok_emb_table[global_row][j];
      }
    }
    upload_one(out.tok_emb[c], chunk_buf.data(), sz_vocab_chunk);
    all.push_back(out.tok_emb[c]);
  }

  q.enqueueMigrateMemObjects(all, 0);
  q.finish();

  size_t total_bytes = kNumLayers * (4 * sz_attn + 2 * sz_ffn + 3 * sz_w2_chunk)
                     + WeightsFPGA::kVocabChunks * sz_vocab_chunk;
  printf("UploadWeightsFPGA: %zu buffers, %.2f MB total\n",
         all.size(), total_bytes / (1024.0 * 1024.0));
}

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dAttn [dim, dim] = Tensor1d [dim]
// out[i] = w[i,j] . in[j]
// Weight is resident in buffer_w (see UploadWeightsFPGA).
void MatmulPt288x288(Tensor1d& out, const Tensor1d& in,
                cl::Buffer buffer_w,
                cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                float* ptr_a, float* ptr_result,
                cl::Buffer buffer_a, cl::Buffer buffer_result) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  g_prof_pack_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueMigrateMemObjects({buffer_a}, 0);
  q.finish();
  g_prof_h2d_ns += ns_since(t0);

  kernel_matmul_pt_288x.setArg(1, buffer_w);
  kernel_matmul_pt_288x.setArg(3, kDim);
  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueTask(kernel_matmul_pt_288x);
  q.finish();
  g_prof_kernel_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  g_prof_d2h_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
  g_prof_unpack_ns += ns_since(t0);
  g_prof_calls_288++;
}
// Compute the matrix multiplication of two input tensors.
// Tensor1dFFNB [ffn_dim] . Tensor2dFFNA [ffn_dim, dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
// Weight is resident in buffer_w (see UploadWeightsFPGA).
void MatmulPt288x768(Tensor1dFFNB& out, const Tensor1d& in,
                cl::Buffer buffer_w,
                cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                float* ptr_a, float* ptr_result,
                cl::Buffer buffer_a, cl::Buffer buffer_result) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  g_prof_pack_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueMigrateMemObjects({buffer_a}, 0);
  q.finish();
  g_prof_h2d_ns += ns_since(t0);

  kernel_matmul_pt_288x.setArg(1, buffer_w);
  kernel_matmul_pt_288x.setArg(3, kFFNDim);
  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueTask(kernel_matmul_pt_288x);
  q.finish();
  g_prof_kernel_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  g_prof_d2h_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kFFNDim; i++) {
    out[i] = ptr_result[i];
  }
  g_prof_unpack_ns += ns_since(t0);
  g_prof_calls_768++;
}

// FFN w2 via column chunking.
// out[288] = w2[288][768] . in[768]
//         = sum_c w2[:, c*288..(c+1)*288-1] . in[c*288..(c+1)*288-1]  (c=0..2)
// 3 kernel launches, each producing a partial [288] which we sum on host.
void MatmulFFNw2FPGA(Tensor1d& out, const Tensor1dFFNB& in,
                     cl::Buffer buffer_w_c0, cl::Buffer buffer_w_c1,
                     cl::Buffer buffer_w_c2,
                     cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                     float* ptr_a, float* ptr_result,
                     cl::Buffer buffer_a, cl::Buffer buffer_result) {
  cl::Buffer* chunks[3] = {&buffer_w_c0, &buffer_w_c1, &buffer_w_c2};
  float acc[kDim];
  for (int i = 0; i < kDim; i++) acc[i] = 0.0f;

  for (int c = 0; c < 3; c++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // Slice c of input: in[c*kDim .. c*kDim + kDim - 1]
    for (int i = 0; i < kDim; i++) {
      ptr_a[i] = in[c * kDim + i];
    }
    g_prof_pack_ns += ns_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    q.enqueueMigrateMemObjects({buffer_a}, 0);
    q.finish();
    g_prof_h2d_ns += ns_since(t0);

    kernel_matmul_pt_288x.setArg(1, *chunks[c]);
    kernel_matmul_pt_288x.setArg(3, kDim);
    t0 = std::chrono::high_resolution_clock::now();
    q.enqueueTask(kernel_matmul_pt_288x);
    q.finish();
    g_prof_kernel_ns += ns_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();
    g_prof_d2h_ns += ns_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kDim; i++) {
      acc[i] += ptr_result[i];
    }
    g_prof_unpack_ns += ns_since(t0);
  }

  for (int i = 0; i < kDim; i++) out[i] = acc[i];
  g_prof_calls_w2++;
}

// Final vocab projection via row chunking.
// out[32000] = tok_emb[32000][288] . in[288]
// Split output rows into 42 chunks of 768. Input sent once, kernel runs 42 times.
void MutmulVocabFPGA(Tensor1dLogits& out, const Tensor1d& in,
                     const WeightsFPGA& wfpga,
                     cl::CommandQueue q, cl::Kernel kernel_matmul_pt_288x,
                     float* ptr_a, float* ptr_result,
                     cl::Buffer buffer_a, cl::Buffer buffer_result) {
  // Input vector: send once.
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  g_prof_pack_ns += ns_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  q.enqueueMigrateMemObjects({buffer_a}, 0);
  q.finish();
  g_prof_h2d_ns += ns_since(t0);

  const int kChunks    = WeightsFPGA::kVocabChunks;
  const int kChunkRows = WeightsFPGA::kVocabChunkRows;

  for (int c = 0; c < kChunks; c++) {
    kernel_matmul_pt_288x.setArg(1, wfpga.tok_emb[c]);
    kernel_matmul_pt_288x.setArg(3, kChunkRows);

    t0 = std::chrono::high_resolution_clock::now();
    q.enqueueTask(kernel_matmul_pt_288x);
    q.finish();
    g_prof_kernel_ns += ns_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();
    g_prof_d2h_ns += ns_since(t0);

    t0 = std::chrono::high_resolution_clock::now();
    int start = c * kChunkRows;
    int end   = start + kChunkRows;
    if (end > kVocabSize) end = kVocabSize;
    for (int r = 0; r < end - start; r++) {
      out[start + r] = ptr_result[r];
    }
    g_prof_unpack_ns += ns_since(t0);
  }

  g_prof_calls_vocab++;
}

/* ---------------------------------  /
      Normalization Operations
/  --------------------------------- */

// Apply the RMS normalization to the input tensor.
// norm = 1 / sum_i..N (in[i]^2) / N
// out[i] = x[i] * norm * w[i]
void RMSNormFPGA(Tensor1d& out, const Tensor1d& in, const Tensor1d& w,
                 cl::CommandQueue q, cl::Kernel kernel_rmsnorm, float* ptr_a,
                 float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    ptr_b[i] = w[i];
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_rmsnorm.setArg(3, kDim);
  q.enqueueTask(kernel_rmsnorm);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Apply the softmax function to the input tensor.
// out[i] = exp(in[i]) / sum(exp(in[i]))
void SoftmaxFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, int in_max_idx,
                 cl::CommandQueue q, cl::Kernel kernel_softmax, float* ptr_a,
                 float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_result) {
  if (in_max_idx == -1) {
    in_max_idx = kSeqLen;
  }

  for (int i = 0; i < in_max_idx; i++) {
    ptr_a[i] = in[i];
  }
  q.enqueueMigrateMemObjects({buffer_a}, 0);
  kernel_softmax.setArg(2, in_max_idx);
  q.enqueueTask(kernel_softmax);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < in_max_idx; i++) {
    out[i] = ptr_result[i];
  }
}

/* ---------------------------------  /
      RoPE: Position Encoding
/  --------------------------------- */

// Apply the rotary position encoding to the input tensor.
// q_out[i] = q_in[i] * cos_vec[i] - q_in[i+1] * sin_vec[i]
// q_out[i+1] = q_in[i] * sin_vec[i] + q_in[i+1] * cos_vec[i]
// k_out[i] = k_in[i] * cos_vec[i] - k_in[i+1] * sin_vec[i]
// k_out[i+1] = k_in[i] * sin_vec[i] + k_in[i+1] * cos_vec[i]
void RoPEFPGA(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
              const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
              const Tensor1dSinCos& sin_vec, int head_begin, int head_dim,
              cl::CommandQueue q, cl::Kernel kernel_rope, float* ptr_a,
              float* ptr_b, float* ptr_c, float* ptr_d, float* ptr_result,
              float* ptr_result2, cl::Buffer buffer_a, cl::Buffer buffer_b,
              cl::Buffer buffer_c, cl::Buffer buffer_d,
              cl::Buffer buffer_result, cl::Buffer buffer_result2) {

  for (int i = 0; i < 288; i++) {
    ptr_a[i] = q_in[i];
    ptr_b[i] = k_in[i];
  }

  for (int i = 0; i < 24; i++) {
    ptr_c[i] = cos_vec[i];
    ptr_d[i] = sin_vec[i];
  }

  q.enqueueMigrateMemObjects({buffer_a, buffer_b, buffer_c, buffer_d}, 0);
  kernel_rope.setArg(6, head_begin);
  q.enqueueTask(kernel_rope);
  q.enqueueMigrateMemObjects({buffer_result, buffer_result2},
                             CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();

  for (int i = 0; i < 288; i++) {
    q_out[i] = ptr_result[i];
    k_out[i] = ptr_result2[i];
  }
}

} // namespace swan

#endif // USE_CPU_ONLY
