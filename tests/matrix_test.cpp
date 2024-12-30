#include <gtest/gtest.h>
#include <stdexcept>
#include "../include/matrix.hpp"

namespace NMatrix {

TEST(MatrixConstructorTest, DefaultConstructor) {
    NMatrix::TMatrix<> matrix;
    EXPECT_EQ(matrix.Rows(), 0.0);
    EXPECT_EQ(matrix.Cols(), 0.0);
}

TEST(MatrixConstructorTest, ParameterizedConstructor) {
    TMatrix<> matrix(1, 3);

    TMatrix<> expected{{0, 0, 0}};
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixConstructorTest, ParameterizedConstructorWithDefaultValue) {
    TMatrix<> matrix(3, 3, 1.0);
    EXPECT_EQ(matrix.Rows(), 3);
    EXPECT_EQ(matrix.Cols(), 3);

    TMatrix<> expected{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixAccessTest, OutOfBoundsAccessRows) {
    TMatrix<> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TEST(MatrixAccessTest, OutOfBoundsAccessCols) {
    TMatrix<> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TEST(MatrixAccessTest, Rows) {
    TMatrix<> matrix(1, 1);
    EXPECT_EQ(matrix.Rows(), 1);
}

TEST(MatrixAccessTest, Cols) {
    TMatrix<> matrix(1, 1);
    EXPECT_EQ(matrix.Cols(), 1);
}

TEST(MatrixOperationsTest, AdditionCorrect) {
    TMatrix<> matrix1(2, 2, 1.0);
    TMatrix<> matrix2(2, 2, 2.0);
    TMatrix<> result = matrix1 + matrix2;

    TMatrix<> expected{{3, 3}, {3, 3}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, AdditionIncorrect) {
    TMatrix<> matrix1(1, 1);
    TMatrix<> matrix2(1, 2);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TEST(MatrixOperationsTest, SubtractionCorrect) {
    TMatrix<> matrix1(2, 2, 3.0);
    TMatrix<> matrix2(2, 2, 2.0);
    TMatrix<> result = matrix1 - matrix2;

    TMatrix<> expected{{1, 1}, {1, 1}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, SubtractionInorrect) {
    TMatrix<> matrix1(1, 1, 3.0);
    TMatrix<> matrix2(2, 2, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TEST(MatrixOperationsTest, MultiplicationCorrect) {
    TMatrix<> matrix1{{2, -3, 1}, {5, 4, -2}};
    TMatrix<> matrix2{{-7, 5}, {2, -1}, {4, 3}};
    TMatrix<> result = matrix1 * matrix2;

    TMatrix<> expected{{-16, 16}, {-35, 15}};
    EXPECT_EQ(result, expected);
}


TEST(MatrixOperationsTest, MultiplicationIncorrect) {
    TMatrix<> matrix1{{1}, {2}};
    TMatrix<> matrix2{{2, 0}, {1, 2}};

    EXPECT_THROW(matrix1 * matrix2, std::invalid_argument);
    
    TMatrix<> result = matrix2 * matrix1;
    TMatrix<> expected{{2}, {5}};
    EXPECT_EQ(result, expected);
}


TEST(MatrixOperationsTest, AdittionEqualScalarCorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    matrix += 1;

    TMatrix<> expected{{2, 3, 4}, {5, 6, 7}};
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, SubtractionEqualScalarCorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    matrix -= 1;

    TMatrix<> expected{{0, 1, 2}, {3, 4, 5}};
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, MultiplicationEqualScalarCorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    matrix *= 2;

    TMatrix<> expected{{2, 4, 6}, {8, 10, 12}};
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, DivisionEqualScalarCorrect) {
    TMatrix<> matrix{{2, 4, 6}, {8, 10, 12}};
    matrix /= 2;

    TMatrix<> expected{{1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(matrix, expected);
}

TEST(MatrixOperationsTest, DivisionEqualScalarIncorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    EXPECT_THROW(matrix /= 0, std::runtime_error);
}

TEST(MatrixOperationsTest, AdittionScalarCorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    TMatrix<> result = matrix + 1;

    TMatrix<> expected{{2, 3, 4}, {5, 6, 7}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, SubtractionScalarCorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    TMatrix<> result = matrix - 1;

    TMatrix<> expected{{0, 1, 2}, {3, 4, 5}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, MultiplicationScalarCorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    TMatrix<> result = matrix * 2;

    TMatrix<> expected{{2, 4, 6}, {8, 10, 12}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, DivisionScalarCorrect) {
    TMatrix<> matrix{{2, 4, 6}, {8, 10, 12}};
    TMatrix<> result = matrix / 2;

    TMatrix<> expected{{1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, DivisionScalarIncorrect) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};

    EXPECT_THROW(TMatrix<> result = matrix / 0, std::runtime_error);
}

TEST(MatrixSpecialFunctionsTest, Transpose) {
    TMatrix<> matrix{{1, 2, 3}, {4, 5, 6}};
    TMatrix<> result = matrix.Transpose();

    TMatrix<> expected{{1, 4}, {2, 5}, {3, 6}};
    EXPECT_EQ(result, expected);
}

TEST(MatrixSpecialFunctionsTest, Equality) {
    TMatrix<> matrix1{{1, 2}, {3, 4}};
    TMatrix<> matrix2{{1, 2}, {3, 4}};
    TMatrix<> matrix3{{4, 3}, {2, 1}};

    EXPECT_TRUE(matrix1 == matrix2);
    EXPECT_FALSE(matrix1 == matrix3);
}

TEST(MatrixSpecialFunctionsTest, Printing) {
    NMatrix::TMatrix<> matrix(2, 3, 1);

    std::stringstream output;
    output << matrix;

    std::string expected_output = 
        "1 1 1 \n"
        "1 1 1 \n";

    EXPECT_EQ(output.str(), expected_output);
}

} // namespace NMatrix

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
