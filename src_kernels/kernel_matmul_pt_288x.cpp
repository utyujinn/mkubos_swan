#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <stdint.h>
#include <hls_vector.h>

#define BLK_ROW 96
#define COLSIZE 288
#define MAX_ROW_SIZE 768
#define MAX_COL_SIZE 288
#define MAX_BLK_NUM  MAX_ROW_SIZE/BLK_ROW    // 8
#define MAX_BLK_NUM_HALF MAX_BLK_NUM/2    // 4
#define MAX_BLK_NUM_URAM 1    // 1
#define MAX_BLK_NUM_BRAM (MAX_BLK_NUM-MAX_BLK_NUM_URAM)    // 8-1=6

extern "C" {
void kernel_matmul_pt_288x(float* i_vec, float* i_mat, float* o_vec, int row) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0 // max_read_burst_length=256 max_write_burst_length=256 depth=1024
#pragma HLS INTERFACE m_axi port = i_mat bundle = gmem1 // max_read_burst_length=256 max_write_burst_length=256 depth=1024*1024
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem1 // max_read_burst_length=256 max_write_burst_length=256 depth=1024
#pragma HLS INTERFACE s_axilite port=row  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    int blk_num = row / BLK_ROW;    // 3 or 8
    // Buffers
    float vec_buf[MAX_BLK_NUM][MAX_COL_SIZE];
    float split_buf[MAX_BLK_NUM][BLK_ROW][MAX_COL_SIZE];
    float out_buf[BLK_ROW][MAX_BLK_NUM];
//
#pragma HLS BIND_STORAGE variable=vec_buf type=RAM_T2P impl=uram
#pragma HLS BIND_STORAGE variable=out_buf type=RAM_T2P impl=uram

    SWEEP_VEC:
    // Load Vector (replicate)
    for (int i=0; i<blk_num; i++){
//#pragma HLS DATAFLOW
        // ------ Read Vector ------
        LOAD_VEC:
        //for (int v_i=0; v_i<col; v_i++){
        for (int v_i=0; v_i<COLSIZE; v_i++){
#pragma HLS PIPELINE
            vec_buf[i][v_i] = i_vec[v_i];
        }
        // Load Matrix (Split)
        // ------ Read Matrix ------
        READ_MAT:
        for (int mr_i=0; mr_i<BLK_ROW; mr_i++){
            for (int m_i=0; m_i<COLSIZE; m_i++){
#pragma HLS PIPELINE
                split_buf[i][mr_i][m_i] = i_mat[(i*BLK_ROW + mr_i)*COLSIZE + m_i];
            }
        }
        // Compute multiply
        // ------ Compute ------
        COMPUTE_MULT:
        for (int c_i=0; c_i<BLK_ROW; c_i++){
            float c_tmp = 0;
#pragma HLS PIPELINE
            COMPUTE_ELE:
            for (int c_j=0; c_j<COLSIZE; c_j++){
#pragma HLS UNROLL
                c_tmp += split_buf[i][c_i][c_j] * vec_buf[i][c_j];
            }
            out_buf[c_i][i] = c_tmp;
        }
        // Output
        // ------ Write result ------
        WRITE_RESULT:
        for (int w_i=0; w_i<BLK_ROW; w_i++){
#pragma HLS PIPELINE
            int new_col = i*BLK_ROW + w_i;
            o_vec[new_col] = out_buf[w_i][i];
        }
    }
}
}

#endif // USE_CPU_ONLY
