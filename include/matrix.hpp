#pragma once

#include <cstdint>
#include <vector>
#include <iostream>

namespace NMatrix {

template <typename T = double>
class TMatrix {
public:
    typedef T value_type;
    TMatrix() = default;
    constexpr TMatrix(size_t rows, size_t cols, const T& value = T()) : data_(rows, std::vector<T>(cols, value)) {}
    constexpr TMatrix(const std::vector<std::vector<T>>& data) : data_(data) {}
	constexpr TMatrix& operator=(const TMatrix&) = default;
    constexpr TMatrix(const TMatrix&) = default;
    constexpr TMatrix(TMatrix&&) = default;
	constexpr TMatrix& operator=(TMatrix&&) = default;
    TMatrix(std::initializer_list<std::initializer_list<T>> list);
    
    const size_t rows() const noexcept;
    const size_t cols() const noexcept;

	std::vector<T>& operator[](size_t);
	const std::vector<T>& operator[](size_t) const;

	TMatrix& operator+=(const TMatrix&);
	TMatrix& operator-=(const TMatrix&);
	TMatrix& operator*=(const TMatrix&);
	// TMatrix& operator/=(const TMatrix&);

    TMatrix operator+(const TMatrix&) const;
    TMatrix operator-(const TMatrix&) const;
    TMatrix operator*(const TMatrix&) const;
	
    TMatrix transpose() const;

    bool operator==(const TMatrix&) const;
    // TODO: add concepts
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const TMatrix<U>&);

private:
    // TODO: use pimpl
    std::vector<std::vector<T>> data_;
};

template <typename T>
TMatrix<T>::TMatrix(std::initializer_list<std::initializer_list<T>> list) {
    for (const auto& row : list) {
        data_.emplace_back(row);
    }
}

template <typename T>
const size_t TMatrix<T>::rows() const noexcept { return data_.size(); }


template <typename T>
const size_t TMatrix<T>::cols() const noexcept { return data_.empty() ? 0 : data_.front().size(); }


template <typename T>
std::vector<T>& TMatrix<T>::operator[](size_t row) {
    if (row >= rows()) {
        throw std::out_of_range("Index out of range");
    }
    return data_[row];
}

template <typename T>
const std::vector<T>& TMatrix<T>::operator[](size_t row) const {
    if (row >= rows()) {
        throw std::out_of_range("Index out of range");
    }
    return data_[row];
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator+=(const TMatrix<T>& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            (*this)[i][j] += other[i][j];
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator-=(const TMatrix<T>& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            (*this)[i][j] -= other[i][j];
        }
    }
    return *this;
}

template <typename T>
TMatrix<T>& TMatrix<T>::operator*=(const TMatrix<T>& other) {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions are not suitable for multiplication");
    }

    TMatrix<T> result(rows(), other.cols(), T{});

    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            for (size_t k = 0; k < cols(); ++k) {
                result[i][j] += (*this)[i][k] * other[k][j];
            }
        }
    }

    *this = std::move(result);
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
TMatrix<T> TMatrix<T>::transpose() const {
	TMatrix<T> result(cols(), rows());
	for (size_t i = 0; i < rows(); ++i) {
		for (size_t j = 0; j < cols(); ++j) {
			result[j][i] = (*this)[i][j];
		}
	}
	return result;
}

template <typename T>
bool TMatrix<T>::operator==(const TMatrix<T>& other) const {
	return data_ == other.data_;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const TMatrix<T>& matrix) {
	for (const auto& row : matrix.data_) {
		for (const auto& elem : row) {
			out << elem << ' ';
		}
		out << '\n';
	}
	return out;
}

} // namespace NMatrix