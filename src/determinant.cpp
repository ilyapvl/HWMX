#include "determinant.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>

namespace LinearAlgebra 
{

size_t Matrix::index(size_t i, size_t j) const 
{ 
    return i * size + j; 
}

Matrix::Matrix(size_t n) : size(n), data(std::make_unique<long double[]>(n * n)) 
{
}

Matrix::Matrix(const Matrix& other) : size(other.size), data(std::make_unique<long double[]>(other.size * other.size)) 
{
    std::copy(other.data.get(), other.data.get() + size * size, data.get());
}

Matrix::Matrix(Matrix&& other) noexcept : size(other.size), data(std::move(other.data)) 
{
    other.size = 0;
}

Matrix& Matrix::operator=(const Matrix& other) 
{
    if (this != &other) 
    {
        size = other.size;
        data = std::make_unique<long double[]>(size * size);
        std::copy(other.data.get(), other.data.get() + size * size, data.get());
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept 
{
    if (this != &other) 
    {
        size = other.size;
        data = std::move(other.data);
        other.size = 0;
    }
    return *this;
}

long double& Matrix::operator()(size_t i, size_t j) 
{ 
    return data[index(i, j)]; 
}

const long double& Matrix::operator()(size_t i, size_t j) const 
{ 
    return data[index(i, j)]; 
}

size_t Matrix::getSize() const 
{ 
    return size; 
}

void Matrix::swapRows(size_t i, size_t j) 
{
    if (i == j) return;
    for (size_t k = 0; k < size; ++k) 
    {
        std::swap((*this)(i, k), (*this)(j, k));
    }
}

Matrix Matrix::copy() const 
{
    return Matrix(*this);
}

long double DeterminantCalculator::calculateDeterminant(Matrix& matrix) 
{
    const size_t n = matrix.getSize();
    
    if (n == 0) return 1.0L;
    if (n == 1) return matrix(0, 0);
    
    std::unique_ptr<size_t[]> pivot = std::make_unique<size_t[]>(n);
    for (size_t i = 0; i < n; ++i) 
    {
        pivot[i] = i;
    }
    
    long double det = 1.0L;
    int sign = 1;
    
    // LU decomposition with partial pivoting
    for (size_t k = 0; k < n; ++k) 
    {
        // Find pivot row
        size_t pivot_row = k;
        long double max_val = std::fabsl(matrix(k, k));
        
        for (size_t i = k + 1; i < n; ++i) 
        {
            long double val = std::fabsl(matrix(i, k));
            if (val > max_val) 
            {
                max_val = val;
                pivot_row = i;
            }
        }
        
        // Swap rows if necessary
        if (pivot_row != k) 
        {
            matrix.swapRows(k, pivot_row);
            std::swap(pivot[k], pivot[pivot_row]);
            sign = -sign;
        }
        
        long double pivot_val = matrix(k, k);
        
        // Check for singular matrix
        if (std::fabsl(pivot_val) < 1e-15L) 
        {
            return 0.0L;
        }
        
        det *= pivot_val;
        
        // Eliminate below diagonal
        for (size_t i = k + 1; i < n; ++i) 
        {
            long double factor = matrix(i, k) / pivot_val;
            matrix(i, k) = factor;
            
            for (size_t j = k + 1; j < n; ++j) 
            {
                matrix(i, j) -= factor * matrix(k, j);
            }
        }
    }
    
    return det * static_cast<long double>(sign);
}

Matrix MatrixReader::readFromFile(const std::string& filename) 
{
    std::ifstream file(filename);
    if (!file.is_open()) 
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    size_t size = 0;
    std::string line;
    std::getline(file, line);
    
    std::istringstream first_line(line);
    long double value;
    while (first_line >> value) 
    {
        ++size;
    }
    
    file.clear();
    file.seekg(0);
    
    Matrix matrix(size);
    
    for (size_t i = 0; i < size; ++i) 
    {
        if (!std::getline(file, line)) 
        {
            throw std::runtime_error("Invalid matrix format: not enough rows");
        }
        
        std::istringstream iss(line);
        for (size_t j = 0; j < size; ++j) 
        {
            if (!(iss >> matrix(i, j))) 
            {
                throw std::runtime_error("Invalid matrix format: not enough columns");
            }
        }
    }
    
    return matrix;
}

Matrix MatrixReader::readFromUserInput() 
{
    std::cout << "Enter matrix size N: ";
    size_t size;
    std::cin >> size;
    
    if (size == 0) 
    {
        return Matrix(0);
    }
    
    Matrix matrix(size);
    
    std::cout << "Enter " << size << "x" << size << " matrix elements row by row:" << std::endl;
    std::cin.ignore();
    
    for (size_t i = 0; i < size; ++i) 
    {
        std::string line;
        std::cout << "Row " << (i + 1) << ": ";
        std::getline(std::cin, line);
        
        std::istringstream iss(line);
        for (size_t j = 0; j < size; ++j) 
        {
            if (!(iss >> matrix(i, j))) 
            {
                throw std::runtime_error("Invalid input: not enough numbers in row " + std::to_string(i + 1));
            }
        }
        
        long double extra;
        if (iss >> extra) 
        {
            throw std::runtime_error("Invalid input: too many numbers in row " + std::to_string(i + 1));
        }
    }
    
    return matrix;
}

void printUsage(const std::string& programName) 
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << programName << " <matrix_file.txt>  - Calculate determinant from file" << std::endl;
    std::cout << "  " << programName << "                   - Enter matrix manually" << std::endl;
    std::cout << "Using long double precision with partial pivoting LU decomposition" << std::endl;
}

} // namespace LinearAlgebra
