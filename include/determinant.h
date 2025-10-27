#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <memory>
#include <string>
#include <cstddef>

namespace LinearAlgebra 
{

class Matrix 
{
private:
    std::unique_ptr<long double[]> data;
    size_t size;
    
    size_t index(size_t i, size_t j) const;
    
public:
    Matrix(size_t n);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix() = default;
    
    long double& operator()(size_t i, size_t j);
    const long double& operator()(size_t i, size_t j) const;
    
    size_t getSize() const;
    void swapRows(size_t i, size_t j);
    Matrix copy() const;
};

namespace DeterminantCalculator 
{
    long double calculateDeterminant(Matrix& matrix);
}

namespace MatrixReader 
{
    Matrix readFromFile(const std::string& filename);
    Matrix readFromUserInput();
}

void printUsage(const std::string& programName);

} // namespace LinearAlgebra

#endif // DETERMINANT_H
