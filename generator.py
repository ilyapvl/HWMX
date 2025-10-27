import numpy as np
from pathlib import Path
from typing import Tuple

class MatrixGenerator:

    def __init__(self):
        self.rng = np.random.default_rng()
    
    def generate_matrix_with_determinant(self, N: int, D: float) -> np.ndarray:

        if N == 0:
            return np.array([[]], dtype=float)
        
        if N == 1:
            return np.array([[D]], dtype=float)
        
        matrix = self._create_initial_matrix(N, D)
        
        matrix = self._apply_random_transformations(matrix, D)
        
        return matrix
    
    def _create_initial_matrix(self, N: int, D: float) -> np.ndarray:

        matrix = np.eye(N, dtype=float)
        
        if abs(D) < 1e-15:
            matrix[0, 0] = 0.0
        else:

            base = abs(D) ** (1.0 / N)
            signs = self.rng.choice([1, -1], size=N)

            if np.prod(signs) != np.sign(D):
                signs[0] *= -1
            
            for i in range(N):
                matrix[i, i] = base * signs[i]
        
        return matrix
    
    def _apply_random_transformations(self, matrix: np.ndarray, target_det: float) -> np.ndarray:

        N = matrix.shape[0]
        num_operations = max(15, N * 2)
        
        for _ in range(num_operations):
            op_type = self.rng.choice(['add', 'scale_compensated', 'orthogonal'])
            
            if op_type == 'add' and N >= 2:
                matrix = self._apply_row_addition(matrix)
            
            elif op_type == 'scale_compensated' and N >= 2:
                matrix = self._apply_compensated_scaling(matrix)
            
            elif op_type == 'orthogonal' and N >= 2:
                matrix = self._apply_orthogonal_transformation(matrix)
        
        return self._adjust_to_exact_determinant(matrix, target_det)
    
    def _apply_row_addition(self, matrix: np.ndarray) -> np.ndarray:

        N = matrix.shape[0]
        i, j = self.rng.choice(N, size=2, replace=False)
        scalar = self.rng.uniform(-2.0, 2.0)
        
        matrix = matrix.copy()
        matrix[i] += scalar * matrix[j]
        return matrix
    
    def _apply_compensated_scaling(self, matrix: np.ndarray) -> np.ndarray:

        N = matrix.shape[0]
        i, j = self.rng.choice(N, size=2, replace=False)
        scalar = self.rng.uniform(0.3, 3.0)
        
        matrix = matrix.copy()

        matrix[i] *= scalar
        matrix[j] *= (1.0 / scalar)
        return matrix
    
    def _apply_orthogonal_transformation(self, matrix: np.ndarray) -> np.ndarray:

        N = matrix.shape[0]
        
        random_matrix = self.rng.standard_normal((N, N))
        Q, _ = np.linalg.qr(random_matrix)
        
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        
        return Q @ matrix
    
    def _adjust_to_exact_determinant(self, matrix: np.ndarray, target_det: float) -> np.ndarray:

        current_det = np.linalg.det(matrix)
        
        if abs(current_det) < 1e-15 and abs(target_det) < 1e-15:
            return matrix
        
        if abs(current_det) < 1e-15:
            return self._create_initial_matrix(matrix.shape[0], target_det)
        
        scale_factor = (target_det / current_det) ** (1.0 / matrix.shape[0])
        return matrix * scale_factor
    
    def verify_determinant(self, matrix: np.ndarray, expected_det: float, tolerance: float = 1e-7) -> bool:
        if matrix.size == 0:
            return abs(expected_det) < tolerance
        
        actual_det = np.linalg.det(matrix)
        return abs(actual_det - expected_det) < tolerance


def save_matrix_to_file(matrix: np.ndarray, filename: str) -> None:

    file_path = Path("./data/" + filename)
 
    with file_path.open('w', encoding='utf-8') as f:
        if matrix.size == 0:
            return
        
        for i in range(matrix.shape[0]):
            row_str = ' '.join(f'{x}' for x in matrix[i])
            f.write(row_str)
            if i < matrix.shape[0] - 1:
                f.write('\n')


def get_user_input() -> Tuple[int, float]:

    try:
        N = int(input("Enter the size of the matrix N: ").strip())
        
        D = float(input("Enter the determinant D: ").strip())
        
        return N, D
        
    except ValueError as e:
        print(f"Invalid input: {e}")
        raise


def main():
    N, D = get_user_input()
        
    print(f"\nGenerating {N}x{N} matrix with determinant {D}...")
    
    generator = MatrixGenerator()
    matrix = generator.generate_matrix_with_determinant(N, D)
    
    if generator.verify_determinant(matrix, D):
        print("Success: Matrix generated")
        actual_det = np.linalg.det(matrix)
        print(actual_det)
    else:
        print("Error: Determinant verification failed")
        actual_det = np.linalg.det(matrix)
        print(f"Expected: {D}, Got: {actual_det}")
        return
    
    filename = f"matrix_{N}_{D:.2f}.txt"
    save_matrix_to_file(matrix, filename)


if __name__ == "__main__":
    main()
