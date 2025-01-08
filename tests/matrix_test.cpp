#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "../include/matrix.hpp"

namespace NMatrix {

TEST(MatrixConstructorTest, DefaultConstructor) {
    const TMatrix<> matrix;
    EXPECT_EQ(matrix.Rows(), 0.0);
    EXPECT_EQ(matrix.Cols(), 0.0);
}

TEST(MatrixConstructorTest, ParameterizedConstructor) {
    TMatrix<> matrix(1, 3);

    const TMatrix<> expected{
        {0, 0, 0}
    };
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixConstructorTest, ParameterizedConstructorWithDefaultValue) {
    const TMatrix<> matrix(3, 3, 1.0);
    EXPECT_EQ(matrix.Rows(), 3);
    EXPECT_EQ(matrix.Cols(), 3);

    const TMatrix<> expected{
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixConstructorTest, InitializerListIncorrect) {
    EXPECT_THROW(
        TMatrix<> matrix({
            {1},
            {2, 3}
            }
        ), 
        std::invalid_argument);
}

TEST(MatrixConstructorTest, VectorsCorrect) {
    const std::vector<long double> row = {1, 2, 3};
    const TMatrix<> matrix {
        row,
        row
    };
    const TMatrix<> expected_output = {
        {1, 2, 3},
        {1, 2, 3}
    };
    EXPECT_EQ(matrix, expected_output);
}

TEST(MatrixConstructorTest, VectorsInorrect) {
    const std::vector<long double> row1 = {1, 2, 3};
    const std::vector<long double> row2 = {1, 2};

    EXPECT_THROW(
        TMatrix<> matrix({
                row1,
                row2
            }
        ), 
        std::invalid_argument
    );
}


TEST(MatrixAccessTest, AccessRowsOutOfBounds) {
    TMatrix<> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TEST(MatrixAccessTest, ConstAccessRowsOutOfBounds) {
    const TMatrix<> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TEST(MatrixAccessTest, AccessColsOutOfBounds) {
    TMatrix<> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TEST(MatrixAccessTest, ConstAccessColsOutOfBounds) {
    const TMatrix<> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TEST(MatrixAccessTest, Rows) {
    const TMatrix<> matrix(1, 1);
    EXPECT_EQ(matrix.Rows(), 1);
}

TEST(MatrixAccessTest, Cols) {
    const TMatrix<> matrix(1, 1);
    EXPECT_EQ(matrix.Cols(), 1);
}

TEST(MatrixOperationsTest, UnaryMinus) {
    const TMatrix<> matrix(2, 2, 1);

    const TMatrix<> expected(2, 2, -1);
    EXPECT_EQ(-matrix, expected);
}


TEST(MatrixOperationsTest, AdditionCorrect) {
    const TMatrix<> matrix1(2, 2, 1.0);
    const TMatrix<> matrix2(2, 2, 2.0);
    const TMatrix<> result = matrix1 + matrix2;

    const TMatrix<> expected{
        {3, 3},
        {3, 3}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, AdditionIncorrectRows) {
    const TMatrix<> matrix1(1, 1);
    const TMatrix<> matrix2(2, 1);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TEST(MatrixOperationsTest, AdditionIncorrectCols) {
    const TMatrix<> matrix1(1, 1);
    const TMatrix<> matrix2(1, 2);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TEST(MatrixOperationsTest, SubtractionCorrect) {
    const TMatrix<> matrix1(2, 2, 3.0);
    const TMatrix<> matrix2(2, 2, 2.0);
    const TMatrix<> result = matrix1 - matrix2;

    const TMatrix<> expected{
        {1, 1},
        {1, 1}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, SubtractionInorrectRows) {
    const TMatrix<> matrix1(1, 1, 3.0);
    const TMatrix<> matrix2(2, 1, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TEST(MatrixOperationsTest, SubtractionInorrectCols) {
    const TMatrix<> matrix1(1, 1, 3.0);
    const TMatrix<> matrix2(1, 2, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TEST(MatrixOperationsTest, MultiplicationCorrect) {
    const TMatrix<> matrix1{
        {2, -3, 1},
        {5, 4, -2}
    };
    const TMatrix<> matrix2{
        {-7, 5},
        {2, -1},
        {4, 3}
    };
    const TMatrix<> result = matrix1 * matrix2;

    const TMatrix<> expected{
        {-16, 16},
        {-35, 15}
    };
    EXPECT_EQ(result, expected);
}


TEST(MatrixOperationsTest, MultiplicationIncorrect) {
    const TMatrix<> matrix1{
        {1},
        {2}
    };
    const TMatrix<> matrix2{
        {2, 0},
        {1, 2}
    };

    EXPECT_THROW(matrix1 * matrix2, std::invalid_argument);
    
    const TMatrix<> result = matrix2 * matrix1;
    const TMatrix<> expected{
        {2},
        {5}
    };
    EXPECT_EQ(result, expected);
}


TEST(MatrixOperationsTest, AdittionEqualScalarCorrect) {
    TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix += 1;

    const TMatrix<> expected{
        {2, 3, 4},
        {5, 6, 7}
    };
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, SubtractionEqualScalarCorrect) {
    TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix -= 1;

    const TMatrix<> expected{
        {0, 1, 2},
        {3, 4, 5}
    };
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, MultiplicationEqualScalarCorrect) {
    TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix *= 2;

    const TMatrix<> expected{
        {2, 4, 6},
        {8, 10, 12}
    };
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, DivisionEqualScalarCorrect) {
    TMatrix<> matrix{
        {2, 4, 6},
        {8, 10, 12}
    };
    matrix /= 2;

    const TMatrix<> expected{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, DivisionEqualScalarIncorrect) {
    TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_THROW(matrix /= 0, std::runtime_error);
}

TEST(MatrixOperationsTest, AdittionScalarCorrect) {
    const TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<> result = matrix + 1;

    const TMatrix<> expected{
        {2, 3, 4},
        {5, 6, 7}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, SubtractionScalarCorrect) {
    const TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<> result = matrix - 1;

    const TMatrix<> expected{
        {0, 1, 2},
        {3, 4, 5}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, MultiplicationScalarCorrect) {
    const TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<> result = matrix * 2;

    const TMatrix<> expected{
        {2, 4, 6},
        {8, 10, 12}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, DivisionScalarCorrect) {
    const TMatrix<> matrix{
        {2, 4, 6},
        {8, 10, 12}
    };
    const TMatrix<> result = matrix / 2;

    const TMatrix<> expected{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, DivisionScalarIncorrect) {
    const TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_THROW(TMatrix<> result = matrix / 0, std::runtime_error);
}

TEST(MatrixSpecialFunctionsTest, Transpose) {
    const TMatrix<> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<> result = matrix.Transpose();

    const TMatrix<> expected{
        {1, 4},
        {2, 5},
        {3, 6}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixSpecialFunctionsTest, Equality) {
    const TMatrix<> matrix1{
        {1, 2},
        {3, 4}
    };
    const TMatrix<> matrix2{
        {1, 2},
        {3, 4}
    };
    const TMatrix<> matrix3{
        {4, 3},
        {2, 1}
    };

    EXPECT_TRUE(matrix1 == matrix2);
    EXPECT_FALSE(matrix1 == matrix3);
}

TEST(MatrixSpecialFunctionsTest, Printing) {
    const TMatrix<> matrix(2, 3, 1);

    std::stringstream output;
    output << matrix;

    const std::string expected_output =
        "1 1 1 \n"
        "1 1 1 \n";

    EXPECT_EQ(output.str(), expected_output);
}

TEST(MatrixSpecialFunctionsTest, GetRowOutOfRange) {
    const TMatrix<> matrix(2, 3, 1);

    EXPECT_THROW(matrix.GetRow(static_cast<int>(matrix.Rows() + 1)), std::out_of_range);
    EXPECT_THROW(matrix.GetRow(static_cast<int>(- matrix.Rows() - 1)), std::out_of_range);
}

TEST(MatrixSpecialFunctionsTest, GetRow) {
    const TMatrix<> matrix(2, 3, 1);
    const auto result = matrix.GetRow(0);

    const std::vector<long double> expected_output = {
        1, 1, 1
    };

    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, GetColumnOutOfRange) {
    const TMatrix<> matrix(2, 3, 1);

    EXPECT_THROW(matrix.GetColumn(static_cast<int>(matrix.Cols() + 1)), std::out_of_range);
    EXPECT_THROW(matrix.GetColumn(static_cast<int>(- matrix.Cols() - 1)), std::out_of_range);
}

TEST(MatrixSpecialFunctionsTest, GetColumn) {
    const TMatrix<> matrix(2, 3, 1);
    const auto result = matrix.GetColumn(0);

    const std::vector<long double> expected_output = {
        1, 1
    };

    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, GetSubMatrixOutOfRange) {
    const TMatrix<> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    auto submatrix = [&](int beginRow, int endRow, int beginCol, int endCol) {
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

TEST(MatrixSpecialFunctionsTest, GetSubMatrix) {
    const TMatrix<> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = GetSubMatrix(matrix, 0, 3, 1, 3);
    const TMatrix<> expected_output = {
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
