#pragma once

#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

namespace NMatrix {

template <typename T = double>
class TMatrix {
private:
    struct ProxyRow {
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;
        using reverse_iterator = typename std::vector<T>::reverse_iterator;
        using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

        constexpr ProxyRow(size_t cols, const T& value = T()) : ProxyData(cols, value) {}
        constexpr ProxyRow(const ProxyRow&) = default;
        // constexpr ProxyRow(ProxyRow&&) noexcept = default;
        constexpr ProxyRow& operator=(const ProxyRow&) = default;
        constexpr ProxyRow& operator=(ProxyRow&&) noexcept = default;
        constexpr bool operator==(const ProxyRow& other) const;
        

        constexpr ProxyRow(const std::vector<T>& other) : ProxyData(other) {}
        constexpr ProxyRow(std::vector<T>&& other) : ProxyData(std::move(other)) {}
        constexpr ProxyRow& operator=(const std::vector<T>& other);
        constexpr ProxyRow& operator=(std::vector<T>&& other);
        constexpr bool operator==(const std::vector<T>& other) const;

        ProxyRow(std::initializer_list<T> list) : ProxyData(list) {}

        operator std::vector<T>() const; 

        friend bool operator==(const std::vector<T>& vec, const ProxyRow& row) {
            return vec == row.ProxyData;
        }

        constexpr size_t size() const noexcept;

        constexpr iterator begin() noexcept { return ProxyData.begin(); }
        constexpr const_iterator begin() const noexcept { return ProxyData.begin(); }
        constexpr iterator end() noexcept { return ProxyData.end(); }
        constexpr const_iterator end() const noexcept { return ProxyData.end(); }

        constexpr const_iterator cbegin() const noexcept { return ProxyData.cbegin(); }
        constexpr const_iterator cend() const noexcept { return ProxyData.cend(); }

        constexpr reverse_iterator rbegin() noexcept { return ProxyData.rbegin(); }
        constexpr const_reverse_iterator rbegin() const noexcept { return ProxyData.rbegin(); }
        constexpr reverse_iterator rend() noexcept { return ProxyData.rend(); }
        constexpr const_reverse_iterator rend() const noexcept { return ProxyData.rend(); }

        constexpr const_reverse_iterator crbegin() const noexcept { return ProxyData.crbegin(); }
        constexpr const_reverse_iterator crend() const noexcept { return ProxyData.crend(); }

        T& operator[](size_t col);
        const T& operator[](size_t col) const;

        std::vector<T> ProxyData;
    };
public:
    using value_type = T;
    TMatrix() = default;
    constexpr TMatrix(size_t rows, size_t cols, const T& value = T()) : Data_(rows, ProxyRow(cols, value)) {}
    constexpr TMatrix(const TMatrix&) = default;
    constexpr TMatrix(TMatrix&&) noexcept = default;
    constexpr TMatrix& operator=(const TMatrix&) = default;
    constexpr TMatrix& operator=(TMatrix&&) noexcept = default;

    constexpr TMatrix(const std::vector<ProxyRow>& data) : Data_(data) {}
    constexpr TMatrix(std::initializer_list<std::initializer_list<T>> list);
    constexpr TMatrix(std::initializer_list<std::vector<T>> list);
    // TODO: add construction from any combinations of vectors and initializer_list
    // need to add conversion from initializer_list to ProxyRow, some developments:
    // template <typename... Rows>
    // constexpr TMatrix(const Rows&... rows) : Data_(std::vector<ProxyRow>(rows.begin(), rows.end())...) {
    //     auto size = Data_.front().size();
    //     for (const auto& current : Data_) {
    //         if (current.size() != static_cast<decltype(current.size())>(size)) {
    //             throw(std::invalid_argument("All rows should be same size"));
    //         }
    //     }
    // }

    size_t Rows() const noexcept;
    size_t Cols() const noexcept;

    ProxyRow& operator[](size_t row);
    const ProxyRow& operator[](size_t row) const;

    TMatrix operator-() const;

    TMatrix& operator+=(const TMatrix&);
    TMatrix& operator-=(const TMatrix&);
    TMatrix& operator*=(const TMatrix&);
    // TMatrix& operator/=(const TMatrix&);

    TMatrix operator+(const TMatrix&) const;
    TMatrix operator-(const TMatrix&) const;
    TMatrix operator*(const TMatrix&) const;

    TMatrix& operator+=(const T&);
    TMatrix& operator-=(const T&);
    TMatrix& operator*=(const T&);
    TMatrix& operator/=(const T&);

    TMatrix operator+(const T&) const;
    TMatrix operator-(const T&) const;
    TMatrix operator*(const T&) const;
    TMatrix operator/(const T&) const;
    
    // TODO: operator std::vector<std::vector<T>>() const;

    TMatrix Transpose() const;
    std::vector<T> GetRow(int row) const;
    std::vector<T> GetColumn(int column) const;

    bool operator==(const TMatrix&) const;
    // TODO: add concepts
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const TMatrix<U>&);

private:
    // TODO: use pimpl
    std::vector<ProxyRow> Data_;
};

template <typename T>
TMatrix<T>::ProxyRow::operator std::vector<T>() const {
    return ProxyData;
} 

// template <typename T>
// constexpr typename TMatrix<T>::ProxyRow& TMatrix<T>::ProxyRow::operator=(const std::vector<T>& other) {
//     ProxyData = other;
//     return *this;
// }

// template <typename T>
// constexpr typename TMatrix<T>::ProxyRow& TMatrix<T>::ProxyRow::operator=(std::vector<T>&& other) {
//     ProxyData = std::move(other);
//     return *this;
// }

template <typename T>
constexpr bool TMatrix<T>::ProxyRow::operator==(const ProxyRow& other) const {
    return ProxyData == other.ProxyData;
}

// template <typename T>
// constexpr bool TMatrix<T>::ProxyRow::operator==(const std::vector<T>& other) const {
//     return ProxyData == other;
// }

template <typename T>
constexpr size_t TMatrix<T>::ProxyRow::size() const noexcept {
    return ProxyData.size();
} 

template <typename T>
T& TMatrix<T>::ProxyRow::operator[](size_t col) {
    if (col >= ProxyData.size()) {
        throw std::out_of_range("Index out of range");
    }
    return ProxyData[col];
}
template <typename T>
const T& TMatrix<T>::ProxyRow::operator[](size_t col) const {
    if (col >= ProxyData.size()) {
        throw std::out_of_range("Index out of range");
    }
    return ProxyData[col];
}


template <typename T>
constexpr TMatrix<T>::TMatrix(std::initializer_list<std::initializer_list<T>> list) {
    size_t rows_ = list.size();
    size_t cols_ = list.begin()->size();

    Data_.reserve(rows_);
    for (const auto& row : list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        Data_.emplace_back(row);
    }
}

template <typename T>
constexpr TMatrix<T>::TMatrix(std::initializer_list<std::vector<T>> list) {
    size_t rows_ = list.size();
    size_t cols_ = list.begin()->size();

    Data_.reserve(rows_);
    for (const auto& row : list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        Data_.emplace_back(row);
    }
}

template <typename T>
size_t TMatrix<T>::Rows() const noexcept { return Data_.size(); }


template <typename T>
size_t TMatrix<T>::Cols() const noexcept { return Data_.empty() ? 0 : Data_.front().size(); }


template <typename T>
typename TMatrix<T>::ProxyRow& TMatrix<T>::operator[](size_t row) {
    if (row >= Rows()) {
        throw std::out_of_range("Index out of range");
    }
    return Data_[row];
}

template <typename T>
const typename TMatrix<T>::ProxyRow& TMatrix<T>::operator[](size_t row) const {
    if (row >= Rows()) {
        throw std::out_of_range("Index out of range");
    }
    return Data_[row];
}

template <typename T>
TMatrix<T> TMatrix<T>::operator-() const {
    TMatrix<T> result(*this);
    for (size_t i = 0; i < Rows(); ++i) {
        for (size_t j = 0; j < Cols(); ++j) {
            result[i][j] = -result[i][j];
        }
    }
    return result;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator+=(const TMatrix<T>& other) {
    if (Rows() != other.Rows() || Cols() != other.Cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    for (size_t i = 0; i < Rows(); ++i) {
        for (size_t j = 0; j < Cols(); ++j) {
            (*this)[i][j] += other[i][j];
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator-=(const TMatrix<T>& other) {
    if (Rows() != other.Rows() || Cols() != other.Cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    for (size_t i = 0; i < Rows(); ++i) {
        for (size_t j = 0; j < Cols(); ++j) {
            (*this)[i][j] -= other[i][j];
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator*=(const TMatrix<T>& other) {
    if (Cols() != other.Rows()) {
        throw std::invalid_argument("Matrix dimensions are not suitable for multiplication");
    }

    *this = BlockAndParallelMultiply(*this, other);

    return *this;
}


template <typename T>
TMatrix<T> TMatrix<T>::operator+(const TMatrix<T>& other) const {
    TMatrix<T> result(*this);
    return result += other;
}

template <typename T>
TMatrix<T> TMatrix<T>::operator-(const TMatrix<T>& other) const {
    TMatrix<T> result(*this);
    return result -= other;
}

template <typename T>
TMatrix<T> TMatrix<T>::operator*(const TMatrix<T>& other) const {
    TMatrix<T> result(*this);
    return result *= other;
}


template <typename T>
TMatrix<T>& TMatrix<T>::operator+=(const T& scalar) {
    for (auto& rows : Data_) {
        for (auto& element : rows) {
            element += scalar;
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator-=(const T& scalar) {
    for (auto& rows : Data_) {
        for (auto& element : rows) {
            element -= scalar;
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator*=(const T& scalar) {
    for (auto& rows : Data_) {
        for (auto& element : rows) {
            element *= scalar;
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator/=(const T& scalar) {
    T zero = T(0.0);
    if (scalar == zero) {
        throw std::runtime_error("Division by zero!");
    }
    for (auto& rows : Data_) {
        for (auto& element : rows) {
            element /= scalar;
        }
    }
    return *this;
}

template <typename T>
TMatrix<T> TMatrix<T>::operator+(const T& scalar) const {
    TMatrix<T> result(*this);
    return result += scalar;
}

template <typename T>
TMatrix<T> TMatrix<T>::operator-(const T& scalar) const {
    TMatrix<T> result(*this);
    return result -= scalar;
}

template <typename T>
TMatrix<T> TMatrix<T>::operator*(const T& scalar) const {
    TMatrix<T> result(*this);
    return result *= scalar;
}

template <typename T>
TMatrix<T> TMatrix<T>::operator/(const T& scalar) const {
    TMatrix<T> result(*this);
    return result /= scalar;
}

template <typename T>
TMatrix<T> TMatrix<T>::Transpose() const {
    TMatrix<T> result(Cols(), Rows());
    for (size_t i = 0; i < Rows(); ++i) {
        for (size_t j = 0; j < Cols(); ++j) {
            result[j][i] = (*this)[i][j];
        }
    }
    return result;
}

template <typename T>
bool TMatrix<T>::operator==(const TMatrix<T>& other) const {
    return Data_ == other.Data_;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const TMatrix<T>& matrix) {
    for (const auto& row : matrix.Data_) {
        for (const auto& elem : row) {
            out << elem << " ";
        }
        out << "\n";
    }
    return out;
}

template <typename T>
std::vector<T> TMatrix<T>::GetRow(int row) const {
    if (!(0 <= row && row < static_cast<int>(Rows()))) {
        throw std::out_of_range("Index out of range");
    }
    return static_cast<std::vector<T>>((*this)[row]);
}

template <typename T>
std::vector<T> TMatrix<T>::GetColumn(int column) const {
    if (!(0 <= column && column < static_cast<int>(Cols()))) {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> result;
    int size = static_cast<int>(this->Rows());
    result.reserve(size);
    for (int row = 0, end = size; row < end; ++row) {
        result.push_back((*this)[row][column]);
    }
    return result;
}

template <typename T>
concept AVX2Supported = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept AVX2IntegerSupported = std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long>;

template <AVX2Supported T>
inline T sum256(std::conditional_t<std::is_same_v<T, float>, __m256, __m256d> vec) {
    if constexpr (std::is_same_v<T, float>) {
        __m128 hi = _mm256_extractf128_ps(vec, 1);
        __m128 lo = _mm256_castps256_ps128(vec);
        __m128 sum = _mm_add_ps(hi, lo);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    } else {
        __m128d hi = _mm256_extractf128_pd(vec, 1);
        __m128d lo = _mm256_castpd256_pd128(vec);
        __m128d sum = _mm_add_pd(hi, lo);
        sum = _mm_hadd_pd(sum, sum);
        return _mm_cvtsd_f64(sum);
    }
}

template <typename T>
void avx2_multiply(
    const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, TMatrix<T>& result,
    size_t ib, size_t jb, size_t kb, size_t block_size
) 
requires AVX2Supported<T>
{
    size_t N = result.Rows();
    size_t M = result.Cols();
    size_t K = matrix1.Cols();

    size_t iend = std::min(ib + block_size, N);
    size_t jend = std::min(jb + block_size, M);
    size_t kend = std::min(kb + block_size, K);

    for (size_t i = ib; i < iend; ++i) {
        for (size_t j = jb; j < jend; ++j) {
            T sum = 0;

            if constexpr (std::is_same_v<T, float>) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t k = kb; k + 16 <= kend; k += 16) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(&matrix1[i][k]), _mm256_loadu_ps(&matrix2[k][j]), acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(&matrix1[i][k + 8]), _mm256_loadu_ps(&matrix2[k + 8][j]), acc);
                }
                sum += sum256<float>(acc);
            } else {
                __m256d acc = _mm256_setzero_pd();
                for (size_t k = kb; k + 8 <= kend; k += 8) {
                    acc = _mm256_fmadd_pd(_mm256_loadu_pd(&matrix1[i][k]), _mm256_loadu_pd(&matrix2[k][j]), acc);
                    acc = _mm256_fmadd_pd(_mm256_loadu_pd(&matrix1[i][k + 4]), _mm256_loadu_pd(&matrix2[k + 4][j]), acc);
                }
                sum += sum256<double>(acc);
            }

            for (size_t k = (kend / (sizeof(T) == 4 ? 16 : 8)) * (sizeof(T) == 4 ? 16 : 8); k < kend; ++k) {
                sum += matrix1[i][k] * matrix2[k][j];
            }

            result[i][j] += sum;
        }
    }
}

template <typename T>
void avx2_integer_multiply(
    const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, TMatrix<T>& result,
    size_t ib, size_t jb, size_t kb, size_t block_size
) 
requires AVX2IntegerSupported<T>
{
    size_t N = result.Rows();
    size_t M = result.Cols();
    size_t K = matrix1.Cols();

    size_t iend = std::min(ib + block_size, N);
    size_t jend = std::min(jb + block_size, M);
    size_t kend = std::min(kb + block_size, K);

    for (size_t i = ib; i < iend; ++i) {
        for (size_t j = jb; j < jend; ++j) {
            T sum = 0;

            __m256i acc = _mm256_setzero_si256();
            for (size_t k = kb; k + 8 <= kend; k += 8) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix1[i][k]));
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j]));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(a, b));
            }

            alignas(32) int res[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(res), acc);
            for (int r : res) sum += r;

            for (size_t k = (kend / 8) * 8; k < kend; ++k) {
                sum += matrix1[i][k] * matrix2[k][j];
            }

            result[i][j] += sum;
        }
    }
}

template <typename T>
void scalar_multiply(
    const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, TMatrix<T>& result,
    size_t ib, size_t jb, size_t kb, size_t block_size
) 
requires (!AVX2Supported<T> && !AVX2IntegerSupported<T>)
{
    size_t N = result.Rows();
    size_t M = result.Cols();
    size_t K = matrix1.Cols();

    size_t iend = std::min(ib + block_size, N);
    size_t jend = std::min(jb + block_size, M);
    size_t kend = std::min(kb + block_size, K);

    for (size_t i = ib; i < iend; ++i) {
        for (size_t j = jb; j < jend; ++j) {
            T sum = 0;
            for (size_t k = kb; k < kend; ++k) {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            result[i][j] += sum;
        }
    }
}

template <typename T>
TMatrix<T> BlockAndParallelMultiply(const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, const size_t block_size = 256) {
    if (matrix1.Cols() != matrix2.Rows()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    size_t N = matrix1.Rows();
    size_t M = matrix2.Cols();
    size_t K = matrix1.Cols();

    TMatrix<T> result(N, M, T{});

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t ib = 0; ib < N; ib += block_size) {
        for (size_t jb = 0; jb < M; jb += block_size) {
            for (size_t kb = 0; kb < K; kb += block_size) {
                if constexpr (AVX2Supported<T>) {
                    avx2_multiply(matrix1, matrix2, result, ib, jb, kb, block_size);
                } else if constexpr (AVX2IntegerSupported<T>) {
                    avx2_integer_multiply(matrix1, matrix2, result, ib, jb, kb, block_size);
                } else {
                    scalar_multiply(matrix1, matrix2, result, ib, jb, kb, block_size);
                }
            }
        }
    }

    return result;
}

// TODO: Rewrite this, may be write TMatrixView

// The half-interview
// [beginRow, endRow)
// [beginCol, endCol)
template<typename T>
TMatrix<T> GetSubMatrix(const TMatrix<T>& matrix,
                        int beginRow, int endRow,
                        int beginCol, int endCol) 
{
    if (beginRow > endRow) {
        std::swap(beginRow, endRow);
    }
    if (beginCol > endCol) {
        std::swap(beginCol, endCol);
    }
    if (!(0 <= beginRow && endRow <= static_cast<int>(matrix.Rows()))
        || !(0 <= beginCol && endCol <= static_cast<int>(matrix.Cols()))
    ) {
        throw std::out_of_range("Some index out of range");
    }

    TMatrix<T> result(endRow - beginRow, endCol - beginCol);
    for (int row = beginRow; row < endRow; ++row) {
        for (int col = beginCol; col < endCol; ++col) {
            result[row - beginRow][col - beginCol] = matrix[row][col];
        }
    }
    return result;
}

} // namespace NMatrix
