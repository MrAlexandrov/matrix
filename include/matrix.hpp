#pragma once

#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace NMatrix {

template <typename T = double>
class TMatrix {
private:
    struct ProxyRow {
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;
        using reverse_iterator = typename std::vector<T>::reverse_iterator;
        using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

        constexpr ProxyRow(size_t cols, const T& value = T()) : ProxyData_(cols, value) {}
        constexpr ProxyRow(const ProxyRow&) = default;
        constexpr ProxyRow(ProxyRow&&) = default;
        constexpr ProxyRow& operator=(const ProxyRow&) = default;
        constexpr ProxyRow& operator=(ProxyRow&&) = default;
        constexpr bool operator==(const ProxyRow& other) const;
        

        constexpr ProxyRow(const std::vector<T>& other) : ProxyData_(other) {}
        constexpr ProxyRow(std::vector<T>&& other) : ProxyData_(std::move(other)) {}
        constexpr ProxyRow& operator=(const std::vector<T>& other);
        constexpr ProxyRow& operator=(std::vector<T>&& other);
        constexpr bool operator==(const std::vector<T>& other) const;
        
        ProxyRow(std::initializer_list<T> list) : ProxyData_(list) {}
        
        operator std::vector<T>() const; 

        friend bool operator==(const std::vector<T>& vec, const ProxyRow& row) {
            return vec == row.ProxyData_;
        }

        constexpr size_t size() const noexcept;

        constexpr iterator begin() noexcept { return ProxyData_.begin(); }
        constexpr const_iterator begin() const noexcept { return ProxyData_.begin(); }
        constexpr iterator end() noexcept { return ProxyData_.end(); }
        constexpr const_iterator end() const noexcept { return ProxyData_.end(); }

        constexpr const_iterator cbegin() const noexcept { return ProxyData_.cbegin(); }
        constexpr const_iterator cend() const noexcept { return ProxyData_.cend(); }

        constexpr reverse_iterator rbegin() noexcept { return ProxyData_.rbegin(); }
        constexpr const_reverse_iterator rbegin() const noexcept { return ProxyData_.rbegin(); }
        constexpr reverse_iterator rend() noexcept { return ProxyData_.rend(); }
        constexpr const_reverse_iterator rend() const noexcept { return ProxyData_.rend(); }

        constexpr const_reverse_iterator crbegin() const noexcept { return ProxyData_.crbegin(); }
        constexpr const_reverse_iterator crend() const noexcept { return ProxyData_.crend(); }

        T& operator[](size_t col);
        const T& operator[](size_t col) const;

        std::vector<T> ProxyData_;
    };
public:
    using value_type = T;
    TMatrix() = default;
    constexpr TMatrix(size_t rows, size_t cols, const T& value = T()) : Data_(rows, ProxyRow(cols, value)) {}
    constexpr TMatrix(const TMatrix&) = default;
    constexpr TMatrix(TMatrix&&) = default;
    constexpr TMatrix& operator=(const TMatrix&) = default;
    constexpr TMatrix& operator=(TMatrix&&) = default;

    constexpr TMatrix(const std::vector<ProxyRow>& data) : Data_(data) {}
    TMatrix(std::initializer_list<std::initializer_list<T>> list);
    
    const size_t Rows() const noexcept;
    const size_t Cols() const noexcept;

	ProxyRow& operator[](size_t row);
	const ProxyRow& operator[](size_t row) const;

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
    return ProxyData_;
} 

template <typename T>
constexpr typename TMatrix<T>::ProxyRow& TMatrix<T>::ProxyRow::operator=(const std::vector<T>& other) {
    ProxyData_ = other;
    return *this;
}

template <typename T>
constexpr typename TMatrix<T>::ProxyRow& TMatrix<T>::ProxyRow::operator=(std::vector<T>&& other) {
    ProxyData_ = std::move(other);
    return *this;
}

template <typename T>
constexpr bool TMatrix<T>::ProxyRow::operator==(const ProxyRow& other) const {
    return ProxyData_ == other.ProxyData_;
}

template <typename T>
constexpr bool TMatrix<T>::ProxyRow::operator==(const std::vector<T>& other) const {
    return ProxyData_ == other;
}

template <typename T>
constexpr size_t TMatrix<T>::ProxyRow::size() const noexcept {
    return ProxyData_.size();
} 

template <typename T>
T& TMatrix<T>::ProxyRow::operator[](size_t col) {
    if (col >= ProxyData_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return ProxyData_[col];
}
template <typename T>
const T& TMatrix<T>::ProxyRow::operator[](size_t col) const {
    if (col >= ProxyData_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return ProxyData_[col];
}


template <typename T>
TMatrix<T>::TMatrix(std::initializer_list<std::initializer_list<T>> list) {
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
const size_t TMatrix<T>::Rows() const noexcept { return Data_.size(); }


template <typename T>
const size_t TMatrix<T>::Cols() const noexcept { return Data_.empty() ? 0 : Data_.front().size(); }


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

    TMatrix<T> result(Rows(), other.Cols(), T{});

    for (size_t i = 0; i < Rows(); ++i) {
        for (size_t j = 0; j < other.Cols(); ++j) {
            for (size_t k = 0; k < Cols(); ++k) {
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
			out << elem << ' ';
		}
		out << '\n';
	}
	return out;
}

} // namespace NMatrix