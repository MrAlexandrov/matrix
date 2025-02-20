#pragma once

#include <initializer_list>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

#include <thread>

namespace NMatrix {

template<typename T>
concept AVX2Supported =
    std::is_trivially_copyable_v<T> && 
    std::is_standard_layout_v<T> &&
    std::is_arithmetic_v<T> &&
    (sizeof(T) == 4 || sizeof(T) == 8) &&
    (std::is_convertible_v<T, float> || 
     std::is_convertible_v<T, double> ||
     std::is_convertible_v<T, int32_t> ||
     std::is_convertible_v<T, int64_t>);


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

        std::vector<T> ProxyData{};
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
    std::vector<ProxyRow> Data_{};
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

    if constexpr (AVX2Supported<T>) {
        *this = BestMultiplyMultithread(*this, other);
    } else {
        *this = BlockMultiplyWithTranspose(*this, other);
    }

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
    T zero = static_cast<T>(0);
    if (scalar == zero) {
        throw std::runtime_error("Division by zero!");
    }
    assert(scalar != zero);
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

// template <typename T>
// TMatrix<T> TMatrix<T>::Transpose() const {
//     TMatrix<T> result(Cols(), Rows());
//     for (size_t i = 0; i < Rows(); ++i) {
//         for (size_t j = 0; j < Cols(); ++j) {
//             result[j][i] = (*this)[i][j];
//         }
//     }
//     return result;
// }

template <typename T>
TMatrix<T> TMatrix<T>::Transpose() const {
    size_t rows = Rows();
    size_t cols = Cols();
    TMatrix<T> transposed(cols, rows);

    const size_t block_size = 64; // Размер блока
    for (size_t i = 0; i < rows; i += block_size) {
        for (size_t j = 0; j < cols; j += block_size) {
            for (size_t ib = i, end = std::min(i + block_size, rows); ib < end; ++ib) {
                for (size_t jb = j, end = std::min(j + block_size, cols); jb < end; ++jb) {
                    transposed[jb][ib] = (*this)[ib][jb];
                }
            }
        }
    }

    return transposed;
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
TMatrix<T> BlockMultiplyWithTranspose(const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, size_t block_size = 64) {
    if (matrix1.Cols() != matrix2.Rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    size_t N = matrix1.Rows();
    size_t M = matrix2.Cols();
    size_t K = matrix1.Cols();

    TMatrix<T> result(N, M, T{0});
    TMatrix<T> transposed_matrix2 = matrix2.Transpose();

    for (size_t ib = 0; ib < N; ib += block_size) {
        for (size_t jb = 0; jb < M; jb += block_size) {
            for (size_t kb = 0; kb < K; kb += block_size) {
                for (size_t i = ib, end_i = std::min(ib + block_size, N); i < end_i; ++i) {
                    for (size_t j = jb, end_j = std::min(jb + block_size, M); j < end_j; ++j) {
                        T sum = 0;
                        for (size_t k = kb, end_k = std::min(kb + block_size, K); k < end_k; ++k) {
                            sum += matrix1[i][k] * transposed_matrix2[j][k];
                        }
                        result[i][j] += sum;
                    }
                }
            }
        }
    }

    return result;
}

// template <typename T>
// requires AVX2Supported<T>
// struct AVX2Traits;

// template <>
// struct AVX2Traits<float> {
//     using VectorType = __m256;
//     static constexpr size_t VectorSize = 8; // 8 float в __m256

//     static VectorType setZero() { return _mm256_setzero_ps(); }
//     static VectorType load(const float* ptr) { return _mm256_loadu_ps(ptr); }
//     static VectorType fmadd(VectorType a, VectorType b, VectorType c) { return _mm256_fmadd_ps(a, b, c); }
//     static float sum(VectorType v) {
//         float temp[8];
//         _mm256_storeu_ps(temp, v);
//         return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
//     }
// };

// template <>
// struct AVX2Traits<double> {
//     using VectorType = __m256d;
//     static constexpr size_t VectorSize = 4; // 4 double в __m256d

//     static VectorType setZero() { return _mm256_setzero_pd(); }
//     static VectorType load(const double* ptr) { return _mm256_loadu_pd(ptr); }
//     static VectorType fmadd(VectorType a, VectorType b, VectorType c) { return _mm256_fmadd_pd(a, b, c); }
//     static double sum(VectorType v) {
//         double temp[4];
//         _mm256_storeu_pd(temp, v);
//         return temp[0] + temp[1] + temp[2] + temp[3];
//     }
// };

// template <>
// struct AVX2Traits<int32_t> {
//     using VectorType = __m256i;
//     static constexpr size_t VectorSize = 8; // 8 int32_t в __m256i

//     static VectorType setZero() { return _mm256_setzero_si256(); }
//     static VectorType load(const int32_t* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }
//     static VectorType multiply(VectorType a, VectorType b) { return _mm256_mullo_epi32(a, b); }
//     static int32_t sum(VectorType v) {
//         alignas(32) int32_t temp[8];
//         _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v);
//         return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
//     }
// };

// template <>
// struct AVX2Traits<long> {
//     using VectorType = __m256i;
//     static constexpr size_t VectorSize = 8; // 8 int32_t в __m256i

//     static VectorType setZero() { return _mm256_setzero_si256(); }
//     static VectorType load(const int32_t* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }
//     static VectorType multiply(VectorType a, VectorType b) { return _mm256_mullo_epi32(a, b); }
//     static int32_t sum(VectorType v) {
//         alignas(32) int32_t temp[8];
//         _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v);
//         return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
//     }
// };

// // template <>
// // struct AVX2Traits<int64_t> {
// //     using VectorType = __m256i;
// //     static constexpr size_t VectorSize = 4; // 4 int64_t в __m256i

// //     static VectorType setZero() { return _mm256_setzero_si256(); }
// //     static VectorType load(const int64_t* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }
// //     static VectorType multiply(VectorType a, VectorType b) { return _mm256_mul_epi32(a, b); }
// //     static int64_t sum(VectorType v) {
// //         alignas(32) int64_t temp[4];
// //         _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v);
// //         return temp[0] + temp[1] + temp[2] + temp[3];
// //     }
// // };

// template <>
// struct AVX2Traits<long long> {
//     using VectorType = __m256i;
//     static constexpr size_t VectorSize = 4; // 4 int64_t в __m256i

//     static VectorType setZero() { return _mm256_setzero_si256(); }
//     static VectorType load(const int64_t* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }
//     static VectorType multiply(VectorType a, VectorType b) { return _mm256_mul_epi32(a, b); }
//     static int64_t sum(VectorType v) {
//         alignas(32) int64_t temp[4];
//         _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v);
//         return temp[0] + temp[1] + temp[2] + temp[3];
//     }
// };

template <typename T>
struct AVX2Traits {
    using VectorType = std::conditional_t<
    std::is_same_v<T, float>, __m256,
    std::conditional_t<std::is_same_v<T, double>, __m256d, __m256i>>;

    static constexpr size_t VectorSize = sizeof(VectorType) / sizeof(T);

    static VectorType setZero() {
        if constexpr (std::is_same_v<T, float>) {
            return _mm256_setzero_ps();
        } else if constexpr (std::is_same_v<T, double>) {
            return _mm256_setzero_pd();
        } else {
            return _mm256_setzero_si256();
        }
    }

    static VectorType load(const T* ptr) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm256_loadu_ps(ptr);
        } else if constexpr (std::is_same_v<T, double>) {
            return _mm256_loadu_pd(ptr);
        } else {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        }
    }    

    static T sum(VectorType v) {
        alignas(32) T temp[VectorSize];
    
        if constexpr (std::is_same_v<T, float>) {
            _mm256_storeu_ps(temp, v);
        } else if constexpr (std::is_same_v<T, double>) {
            _mm256_storeu_pd(temp, v);
        } else if constexpr (sizeof(T) == 4) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), v);
        } else if constexpr (sizeof(T) == 8) {
            __m128i low128 = _mm256_extracti128_si256(v, 0);
            __m128i high128 = _mm256_extracti128_si256(v, 1);
            __m128i sum128 = _mm_add_epi64(low128, high128);
            temp = _mm_cvtsi128_si64(sum128) + _mm_extract_epi64(sum128, 1);
        } else {
            static_assert(false, "Unreachable");
        }
    
        T result = T{0};
        for (size_t i = 0; i < VectorSize; ++i) {
            result += temp[i];
        }
        return result;
    }    
};

// Специализация для целых чисел (размер 4 байта)
template <typename T>
requires (std::is_integral_v<T> && sizeof(T) == 4)
struct AVX2Traits<T> {
    using VectorType = __m256i;
    static constexpr size_t VectorSize = 8;

    static VectorType setZero() { return _mm256_setzero_si256(); }
    static VectorType load(const T* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }
    static VectorType multiply(VectorType a, VectorType b) { return _mm256_mullo_epi32(a, b); }
    static T sum(VectorType v) {
        alignas(32) T temp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v);
        return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }
};

// TODO: rewrite, do note work well
// Специализация для целых чисел (размер 8 байт)
template <typename T>
requires (std::is_integral_v<T> && sizeof(T) == 8)
struct AVX2Traits<T> {
    using VectorType = __m256i;
    static constexpr size_t VectorSize = 4;

    static VectorType setZero() { return _mm256_setzero_si256(); }
    static VectorType load(const T* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }

    static VectorType multiply(VectorType a, VectorType b) {
        const __m256i b_swap        = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1));   // swap H<->L
        const __m256i crossprod     = _mm256_mullo_epi32(a, b_swap);                 // 32-bit L*H and H*L cross-products
    
        const __m256i prodlh        = _mm256_slli_epi64(crossprod, 32);          // bring the low half up to the top of each 64-bit chunk 
        const __m256i prodhl        = _mm256_and_si256(crossprod, _mm256_set1_epi64x(0xFFFFFFFF00000000)); // isolate the other, also into the high half were it needs to eventually be
        const __m256i sumcross      = _mm256_add_epi32(prodlh, prodhl);       // the sum of the cross products, with the low half of each u64 being 0.
    
        const __m256i prodll        = _mm256_mul_epu32(a,b);                  // widening 32x32 => 64-bit  low x low products
        const __m256i prod          = _mm256_add_epi32(prodll, sumcross);     // add the cross products into the high half of the result
        return prod;
    }
    static T sum(VectorType v) {
        __m128i low128 = _mm256_extracti128_si256(v, 0);
        __m128i high128 = _mm256_extracti128_si256(v, 1);
    
        __m128i sum128 = _mm_add_epi64(low128, high128);
    
        return _mm_cvtsi128_si64(sum128) + _mm_extract_epi64(sum128, 1);
    }
};

// Специализация для float
template <>
struct AVX2Traits<float> {
    using VectorType = __m256;
    static constexpr size_t VectorSize = 8;

    static VectorType setZero() { return _mm256_setzero_ps(); }
    static VectorType load(const float* ptr) { return _mm256_loadu_ps(ptr); }
    static VectorType fmadd(VectorType a, VectorType b, VectorType c) { return _mm256_fmadd_ps(a, b, c); }
    static float sum(VectorType v) {
        float temp[8];
        _mm256_storeu_ps(temp, v);
        return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }
};

// Специализация для double
template <>
struct AVX2Traits<double> {
    using VectorType = __m256d;
    static constexpr size_t VectorSize = 4;

    static VectorType setZero() { return _mm256_setzero_pd(); }
    static VectorType load(const double* ptr) { return _mm256_loadu_pd(ptr); }
    static VectorType fmadd(VectorType a, VectorType b, VectorType c) { return _mm256_fmadd_pd(a, b, c); }
    static double sum(VectorType v) {
        double temp[4];
        _mm256_storeu_pd(temp, v);
        return temp[0] + temp[1] + temp[2] + temp[3];
    }
};


template <typename T>
T avx2_dot_product(const T* a, const T* b, size_t size) {
    using Traits = AVX2Traits<T>;
    using VectorType = typename Traits::VectorType;
    VectorType acc = Traits::setZero();
    size_t vecSize = Traits::VectorSize;
    size_t k = 0;

    for (; k + vecSize <= size; k += vecSize) {
        VectorType va = Traits::load(&a[k]);
        VectorType vb = Traits::load(&b[k]);
        if constexpr (std::is_floating_point_v<T>) {
            acc = Traits::fmadd(va, vb, acc);
        } else if constexpr (std::is_integral_v<T>) {
            VectorType product = Traits::multiply(va, vb);
            if constexpr (sizeof(T) == 4) {
                acc = _mm256_add_epi32(acc, product);
            } else if constexpr (sizeof(T) == 8) {
                acc = _mm256_add_epi64(acc, product);
            } else {
                static_assert(false, "Unreachable");
            }
        }
    }

    T result = Traits::sum(acc);
    for (; k < size; ++k) {
        result += a[k] * b[k];
    }
    return result;
}

// template <typename T>
// requires AVX2Supported<T>
// TMatrix<T> BestMultiply(const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, size_t block_size = 64) {
//     if (matrix1.Cols() != matrix2.Rows()) {
//         throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
//     }
//     size_t N = matrix1.Rows();
//     size_t M = matrix2.Cols();
//     size_t K = matrix1.Cols();
//     TMatrix<T> result(N, M, T{0});
//     TMatrix<T> transposed_matrix2 = matrix2.Transpose();

//     for (size_t ib = 0; ib < N; ib += block_size) {
//         for (size_t jb = 0; jb < M; jb += block_size) {
//             for (size_t kb = 0; kb < K; kb += block_size) {
//                 for (size_t i = ib, end_i = std::min(ib + block_size, N); i < end_i; ++i) {
//                     for (size_t j = jb, end_j = std::min(jb + block_size, M); j < end_j; ++j) {
//                         size_t count = std::min(block_size, K - kb);
//                         result[i][j] += avx2_dot_product(&matrix1[i][kb], &transposed_matrix2[j][kb], count);
//                     }
//                 }
//             }
//         }
//     }

//     return result;
// }

template <typename T>
requires AVX2Supported<T>
TMatrix<T> BestMultiplyMultithread(const TMatrix<T>& matrix1, const TMatrix<T>& matrix2, size_t block_size = 64) {
    if (matrix1.Cols() != matrix2.Rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), matrix1.Rows());

    size_t N = matrix1.Rows();
    size_t M = matrix2.Cols();
    size_t K = matrix1.Cols();
    TMatrix<T> result(N, M, T{0});
    TMatrix<T> transposed_matrix2 = matrix2.Transpose();

    {
        std::vector<std::jthread> threads;

        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.emplace_back([&, thread_id]() {
                for (size_t ib = thread_id * block_size; ib < N; ib += num_threads * block_size) {
                    for (size_t jb = 0; jb < M; jb += block_size) {
                        for (size_t kb = 0; kb < K; kb += block_size) {
                            for (size_t i = ib, end_i = std::min(ib + block_size, N); i < end_i; ++i) {
                                for (size_t j = jb, end_j = std::min(jb + block_size, M); j < end_j; ++j) {
                                    size_t count = std::min(block_size, K - kb);
                                    result[i][j] += avx2_dot_product(&(matrix1[i][kb]), &(transposed_matrix2[j][kb]), count);;
                                }
                            }
                        }
                    }
                }
            });
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

template <typename T>
TMatrix<T> InvertMatrix(TMatrix<T> matrix) {
    size_t n = matrix.Rows();
    if (n != matrix.Cols()) {
        throw std::invalid_argument("Matrix must be square to invert.");
    }

    TMatrix<T> inverse(n, n, 1);

    for (size_t i = 0; i < n; ++i) {
        T pivot = matrix[i][i];
        if (pivot == 0) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }

        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            matrix[i][j] /= pivot;
            inverse[i][j] /= pivot;
        }

        #pragma omp parallel for
        for (size_t row = 0; row < n; ++row) {
            if (row != i) {
                T factor = matrix[row][i];
                for (size_t col = 0; col < n; ++col) {
                    matrix[row][col] -= factor * matrix[i][col];
                    inverse[row][col] -= factor * inverse[i][col];
                }
            }
        }
    }

    return inverse;
}

template <typename T>
TMatrix<T> FastPower(const TMatrix<T>& matrix, unsigned degree) {
    if (matrix.Rows() != matrix.Cols()) {
        throw std::invalid_argument("Matrix should be square for exponentiation.");
    }
    if (degree == 0) {
        size_t n = matrix.Rows();
        TMatrix<T> identity(n, n, T{0});
        for (size_t i = 0; i < n; ++i) {
            identity[i][i] = T{1};
        }
        return identity;
    }

    TMatrix<T> result = TMatrix<T>(matrix);
    TMatrix<T> base = matrix;
    unsigned power = degree - 1;
    
    while (power > 0) {
        // same as (power % 2 == 1)
        if (static_cast<bool>(power & 1)) {
            result *= base;
        }
        base *= base;
        power >>= 1;
    }
    
    return result;
}

template <typename T>
TMatrix<T> SlowPower(const TMatrix<T>& matrix, unsigned degree) {
    if (matrix.Rows() != matrix.Cols()) {
        throw std::invalid_argument("Matrix should be square for exponentiation.");
    }

    if (degree == 0) {
        size_t n = matrix.Rows();
        TMatrix<T> identity(n, n, T{0});
        for (size_t i = 0; i < n; ++i) {
            identity[i][i] = T{1};
        }
        return identity;
    }

    TMatrix<T> result = matrix;

    for (unsigned i = 1; i < degree; ++i) {
        result = result * matrix;
    }

    return result;
}

} // namespace NMatrix
