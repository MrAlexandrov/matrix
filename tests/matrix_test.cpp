#include "gtest/gtest.h"
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "../include/matrix.hpp"

namespace NMatrix {

using TestTypes = ::testing::Types<short, float, double, long double, int, long, long long>;

template <typename T>
class MatrixConstructorTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};

template <typename T>
class MatrixAccessTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};

template <typename T>
class MatrixOperationsTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};

template <typename T>
class MatrixSpecialFunctionsTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};


TYPED_TEST_SUITE(MatrixConstructorTest, TestTypes);
TYPED_TEST_SUITE(MatrixAccessTest, TestTypes);
TYPED_TEST_SUITE(MatrixOperationsTest, TestTypes);
TYPED_TEST_SUITE(MatrixSpecialFunctionsTest, TestTypes);


TYPED_TEST(MatrixConstructorTest, DefaultConstructor) {
    using T = TypeParam;
    const TMatrix<T> matrix;
    EXPECT_EQ(matrix.Rows(), 0.0);
    EXPECT_EQ(matrix.Cols(), 0.0);
}

TYPED_TEST(MatrixConstructorTest, ParameterizedConstructor) {
    using T = TypeParam;
    TMatrix<T> matrix(1, 3);

    const TMatrix<T> expected{
        {0, 0, 0}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixConstructorTest, ParameterizedConstructorWithDefaultValue) {
    using T = TypeParam;
    const TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_EQ(matrix.Rows(), 3);
    EXPECT_EQ(matrix.Cols(), 3);

    const TMatrix<T> expected{
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixConstructorTest, InitializerListIncorrect) {
    using T = TypeParam;
    EXPECT_THROW(
        TMatrix<T> matrix({
            {1},
            {2, 3}
            }
        ), 
        std::invalid_argument);
}

TYPED_TEST(MatrixConstructorTest, VectorsCorrect) {
    using T = TypeParam;
    const std::vector<T> row = {1, 2, 3};
    const TMatrix<T> matrix {
        row,
        row
    };
    const TMatrix<T> expected_output = {
        {1, 2, 3},
        {1, 2, 3}
    };
    EXPECT_EQ(matrix, expected_output);
}

TYPED_TEST(MatrixConstructorTest, VectorsInorrect) {
    using T = TypeParam;
    const std::vector<T> row1 = {1, 2, 3};
    const std::vector<T> row2 = {1, 2};

    EXPECT_THROW(
        TMatrix<T> matrix({
                row1,
                row2
            }
        ), 
        std::invalid_argument
    );
}


TYPED_TEST(MatrixAccessTest, AccessRowsOutOfBounds) {
    using T = TypeParam;
    TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, ConstAccessRowsOutOfBounds) {
    using T = TypeParam;
    const TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, AccessColsOutOfBounds) {
    using T = TypeParam;
    TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, ConstAccessColsOutOfBounds) {
    using T = TypeParam;
    const TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, Rows) {
    using T = TypeParam;
    const TMatrix<T> matrix(1, 1);
    EXPECT_EQ(matrix.Rows(), 1);
}

TYPED_TEST(MatrixAccessTest, Cols) {
    using T = TypeParam;
    const TMatrix<T> matrix(1, 1);
    EXPECT_EQ(matrix.Cols(), 1);
}

TYPED_TEST(MatrixOperationsTest, UnaryMinus) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 2, 1);

    const TMatrix<T> expected(2, 2, -1);
    EXPECT_EQ(-matrix, expected);
}


TYPED_TEST(MatrixOperationsTest, AdditionCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1(2, 2, 1.0);
    const TMatrix<T> matrix2(2, 2, 2.0);
    const TMatrix<T> result = matrix1 + matrix2;

    const TMatrix<T> expected{
        {3, 3},
        {3, 3}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, AdditionIncorrectRows) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1);
    const TMatrix<T> matrix2(2, 1);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TYPED_TEST(MatrixOperationsTest, AdditionIncorrectCols) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1);
    const TMatrix<T> matrix2(1, 2);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TYPED_TEST(MatrixOperationsTest, SubtractionCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1(2, 2, 3.0);
    const TMatrix<T> matrix2(2, 2, 2.0);
    const TMatrix<T> result = matrix1 - matrix2;

    const TMatrix<T> expected{
        {1, 1},
        {1, 1}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, SubtractionInorrectRows) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1, 3.0);
    const TMatrix<T> matrix2(2, 1, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TYPED_TEST(MatrixOperationsTest, SubtractionInorrectCols) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1, 3.0);
    const TMatrix<T> matrix2(1, 2, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1{
        {2, -3, 1},
        {5, 4, -2}
    };
    const TMatrix<T> matrix2{
        {-7, 5},
        {2, -1},
        {4, 3}
    };
    const TMatrix<T> result = matrix1 * matrix2;

    const TMatrix<T> expected{
        {-16, 16},
        {-35, 15}
    };
    EXPECT_EQ(result, expected);
}


TYPED_TEST(MatrixOperationsTest, MultiplicationIncorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1{
        {1},
        {2}
    };
    const TMatrix<T> matrix2{
        {2, 0},
        {1, 2}
    };

    EXPECT_THROW(matrix1 * matrix2, std::invalid_argument);
    
    const TMatrix<T> result = matrix2 * matrix1;
    const TMatrix<T> expected{
        {2},
        {5}
    };
    EXPECT_EQ(result, expected);
}


TYPED_TEST(MatrixOperationsTest, AdittionEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix += 1;

    const TMatrix<T> expected{
        {2, 3, 4},
        {5, 6, 7}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, SubtractionEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix -= 1;

    const TMatrix<T> expected{
        {0, 1, 2},
        {3, 4, 5}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix *= 2;

    const TMatrix<T> expected{
        {2, 4, 6},
        {8, 10, 12}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {2, 4, 6},
        {8, 10, 12}
    };
    matrix /= 2;

    const TMatrix<T> expected{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionEqualScalarIncorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_THROW(matrix /= 0, std::runtime_error);
}

TYPED_TEST(MatrixOperationsTest, AdittionScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix + 1;

    const TMatrix<T> expected{
        {2, 3, 4},
        {5, 6, 7}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, SubtractionScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix - 1;

    const TMatrix<T> expected{
        {0, 1, 2},
        {3, 4, 5}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix * 2;

    const TMatrix<T> expected{
        {2, 4, 6},
        {8, 10, 12}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {2, 4, 6},
        {8, 10, 12}
    };
    const TMatrix<T> result = matrix / 2;

    const TMatrix<T> expected{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionScalarIncorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_THROW(TMatrix<T> result = matrix / 0, std::runtime_error);
}

TYPED_TEST(MatrixSpecialFunctionsTest, Transpose) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix.Transpose();

    const TMatrix<T> expected{
        {1, 4},
        {2, 5},
        {3, 6}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixSpecialFunctionsTest, Equality) {
    using T = TypeParam;
    const TMatrix<T> matrix1{
        {1, 2},
        {3, 4}
    };
    const TMatrix<T> matrix2{
        {1, 2},
        {3, 4}
    };
    const TMatrix<T> matrix3{
        {4, 3},
        {2, 1}
    };

    EXPECT_TRUE(matrix1 == matrix2);
    EXPECT_FALSE(matrix1 == matrix3);
}

TYPED_TEST(MatrixSpecialFunctionsTest, Printing) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);

    std::stringstream output;
    output << matrix;

    const std::string expected_output =
        "1 1 1 \n"
        "1 1 1 \n";

    EXPECT_EQ(output.str(), expected_output);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetRowOutOfRange) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);

    EXPECT_THROW(matrix.GetRow(static_cast<int>(matrix.Rows() + 1)), std::out_of_range);
    EXPECT_THROW(matrix.GetRow(static_cast<int>(- matrix.Rows() - 1)), std::out_of_range);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetRow) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);
    const auto result = matrix.GetRow(0);

    const std::vector<T> expected_output = {
        1, 1, 1
    };

    EXPECT_EQ(result, expected_output);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetColumnOutOfRange) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);

    EXPECT_THROW(matrix.GetColumn(static_cast<int>(matrix.Cols() + 1)), std::out_of_range);
    EXPECT_THROW(matrix.GetColumn(static_cast<int>(- matrix.Cols() - 1)), std::out_of_range);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetColumn) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);
    const auto result = matrix.GetColumn(0);

    const std::vector<T> expected_output = {
        1, 1
    };

    EXPECT_EQ(result, expected_output);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetSubMatrixOutOfRange) {
    using T = TypeParam;
    const TMatrix<T> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    auto submatrix = [&](int beginRow, int endRow, int beginCol, int endCol) {
        using T = TypeParam;
        return GetSubMatrix(matrix, beginRow, endRow, beginCol, endCol);
    };
    EXPECT_THROW(submatrix(-1, 0, 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, -1, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, static_cast<int>(matrix.Rows() + 1), 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, 0, static_cast<int>(matrix.Cols() + 1)), std::out_of_range);

    EXPECT_THROW(submatrix(0, -1, 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, 0, -1), std::out_of_range);
    EXPECT_THROW(submatrix(static_cast<int>(matrix.Rows() + 1), 0, 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, 0, static_cast<int>(matrix.Cols() + 1)), std::out_of_range);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetSubMatrix) {
    using T = TypeParam;
    const TMatrix<T> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = GetSubMatrix(matrix, 0, 3, 1, 3);
    const TMatrix<T> expected_output = {
        {2, 3},
        {5, 6},
        {8, 9}
    };

    EXPECT_EQ(result, expected_output);
}

} // namespace NMatrix

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
