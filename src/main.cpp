#include <iostream>
#include <chrono>
#include "determinant.h"

using namespace LinearAlgebra;

int main(int argc, char* argv[]) 
{
    try 
    {
        Matrix matrix(0);
        
        if (argc == 2) 
        {
            // Read from file
            matrix = MatrixReader::readFromFile(argv[1]);
            auto read_time = std::chrono::high_resolution_clock::now();
            
            double determinant = DeterminantCalculator::calculateDeterminant(matrix);
            auto calc_time = std::chrono::high_resolution_clock::now();
            
            std::cout << determinant << std::endl;
            
            auto calc_duration = std::chrono::duration_cast<std::chrono::microseconds>(calc_time - read_time);
            
            std::cerr << "Calculation time: " << calc_duration.count() << " μs" << std::endl;
        }
        else if (argc == 1) 
        {
            matrix = MatrixReader::readFromUserInput();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            double determinant = DeterminantCalculator::calculateDeterminant(matrix);
            auto calc_time = std::chrono::high_resolution_clock::now();
            
            std::cout << "Determinant: " << determinant << std::endl;
            
            auto calc_duration = std::chrono::duration_cast<std::chrono::microseconds>(calc_time - start_time);
            std::cerr << "Calculation time: " << calc_duration.count() << " μs" << std::endl;
        }
        else 
        {
            printUsage(argv[0]);
            return 1;
        }
        
        std::cerr << "Matrix size: " << matrix.getSize() << "x" << matrix.getSize() << std::endl;
        
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
