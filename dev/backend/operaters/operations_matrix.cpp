#include <vector>

// matrix_add
std::vector<std::vector<double>> matrix_add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B){
    size_t rows = A.size();
    size_t cols = A[0].size();

    // result vector init
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j <cols; ++j){
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B){
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner_dim = B.size();
    
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            for (size_t k = 0; k < inner_dim; ++k){
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}