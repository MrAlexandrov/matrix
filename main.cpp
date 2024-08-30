#include "include/matrix.hpp"

using namespace NMatrix;

int main() {
    int n, m;
    std::cin >> n >> m;
    TMatrix<> a(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cin >> a[i][j];
        }
    }
    std::cout << n << ' ' << m << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << a[i][j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
    return 0;
}