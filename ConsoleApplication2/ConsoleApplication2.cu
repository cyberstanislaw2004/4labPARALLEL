#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

typedef vector<vector<long long>> Matrix;

__global__ void matrixMultiplyKernel(long long* A, long long* B, long long* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        long long sum = 0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * p + col];
        C[row * p + col] = sum;
    }
}

Matrix read_matrix(const string& filename) {
    ifstream file(filename);
    Matrix matrix;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<long long> row;
        long long num;
        while (iss >> num) row.push_back(num);
        matrix.push_back(row);
    }
    return matrix;
}

void write_matrix(const Matrix& matrix, const string& filename) {
    ofstream file(filename);
    for (const auto& row : matrix) {
        for (long long val : row)
            file << val << " ";
        file << "\n";
    }
}

void log_time(const string& log_path, const string& matrix_name, long long time_ms, long long m, long long n, long long p) {
    ofstream log_file(log_path, ios::app);
    log_file << matrix_name << ": Time (ms): " << time_ms
             << ", Size: A(" << m << "x" << n << "), B(" << n << "x" << p << "), C(" << m << "x" << p << ")\n";
}

void gpu_matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    size_t sizeA = m * n * sizeof(long long);
    size_t sizeB = n * p * sizeof(long long);
    size_t sizeC = m * p * sizeof(long long);

    long long *h_A = new long long[m * n];
    long long *h_B = new long long[n * p];
    long long *h_C = new long long[m * p];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            h_A[i * n + j] = A[i][j];

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            h_B[i * p + j] = B[i][j];

    long long *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + 15) / 16, (m + 15) / 16);

    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            C[i][j] = h_C[i * p + j];

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    vector<string> matrices = {
        "10x10", "100x100", "500x500", "1000x1000", "2000x2000", "3000x3000"
    };

    for (const auto& name : matrices) {
        Matrix A = read_matrix("matrix/" + name + ".txt");
        Matrix B = read_matrix("matrix/" + name + ".txt");

        int m = A.size();
        int n = B.size();
        int p = B[0].size();

        Matrix C(m, vector<long long>(p, 0));

        auto start = high_resolution_clock::now();
        gpu_matrix_multiply(A, B, C, m, n, p);
        auto end = high_resolution_clock::now();

        long long time_ms = duration_cast<milliseconds>(end - start).count();

        write_matrix(C, name + "result.txt");
        log_time("timings.txt", name, time_ms, m, n, p);
    }

    return 0;
}