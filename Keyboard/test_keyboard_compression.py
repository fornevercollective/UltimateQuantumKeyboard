import io
import json
import zlib
from keyboard_compression import QuantumKeyboard

def test_project_orthogonal_3d_path():
    """Test the project_orthogonal_3d_path method for consistency in return types."""
    keyboard = QuantumKeyboard('qwerty')
    word_positions = keyboard.get_word_positions("test")
    projections = keyboard.project_orthogonal_3d_path(word_positions)
    
    print("Testing project_orthogonal_3d_path:")
    for direction, projection in projections.items():
        print(f"  {direction}: {projection}")
        # Check if all projections have consistent dimensions
        dimensions = [len(point) for point in projection]
        print(f"  Dimensions: {dimensions}")
    
    # The method's return type annotation suggests all projections should be 2D
    # But some are actually 3D, which is inconsistent

def test_gzip_compression():
    """Test the gzip compression and decompression methods."""
    keyboard = QuantumKeyboard('qwerty')
    word_positions = keyboard.get_word_positions("test")
    
    print("\nTesting gzip_compress_path:")
    try:
        compressed = keyboard.gzip_compress_path(word_positions)
        print(f"  Compressed size: {len(compressed)} bytes")
    except Exception as e:
        print(f"  Error in compression: {e}")
    
    print("\nTesting gzip_decompress_path:")
    try:
        if 'compressed' in locals():
            decompressed = keyboard.gzip_decompress_path(compressed)
            print(f"  Decompressed: {decompressed}")
            print(f"  Original: {word_positions}")
            print(f"  Match: {decompressed == word_positions}")
    except Exception as e:
        print(f"  Error in decompression: {e}")

if __name__ == "__main__":
    test_project_orthogonal_3d_path()
    test_gzip_compression()