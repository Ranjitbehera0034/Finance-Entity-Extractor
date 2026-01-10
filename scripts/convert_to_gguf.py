"""
GGUF Conversion Script.

Converts the PyTorch model to GGUF format for llama.cpp compatibility.
This enables deployment on any platform (Linux, Windows, Mac) with CPU or GPU.

Author: Ranjit Behera
"""

import subprocess
import argparse
import sys
from pathlib import Path
import shutil
import tempfile
import os


def check_llama_cpp():
    """Check if llama.cpp is available, download if needed."""
    llama_cpp_path = Path("tools/llama.cpp")
    
    if not llama_cpp_path.exists():
        print("📥 llama.cpp not found. Cloning repository...")
        llama_cpp_path.parent.mkdir(parents=True, exist_ok=True)
        
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_cpp_path)
        ], check=True)
        
        print("✅ llama.cpp cloned successfully")
    
    return llama_cpp_path


def convert_to_gguf(model_path: Path, output_path: Path, quantization: str = "q4_k_m"):
    """
    Convert PyTorch/Safetensors model to GGUF format.
    
    Args:
        model_path: Path to the PyTorch model directory
        output_path: Output path for GGUF file
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16, f32)
    """
    llama_cpp = check_llama_cpp()
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        # Try alternative script name
        convert_script = llama_cpp / "convert-hf-to-gguf.py"
    
    if not convert_script.exists():
        print("❌ Conversion script not found in llama.cpp")
        print("   Trying pip-installed llama-cpp-python converter...")
        return convert_with_pip_package(model_path, output_path, quantization)
    
    # Step 1: Convert to GGUF (F16)
    print(f"🔄 Converting {model_path} to GGUF (F16)...")
    
    f16_output = output_path.parent / f"{output_path.stem}-f16.gguf"
    
    cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(f16_output),
        "--outtype", "f16"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ F16 GGUF created: {f16_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        return None
    
    # Step 2: Quantize if needed
    if quantization != "f16":
        print(f"🔄 Quantizing to {quantization}...")
        
        quantize_bin = llama_cpp / "quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_cpp / "build" / "bin" / "quantize"
        
        if not quantize_bin.exists():
            print("⚠️ Quantize binary not found. Using F16 output.")
            shutil.move(f16_output, output_path)
            return output_path
        
        cmd = [str(quantize_bin), str(f16_output), str(output_path), quantization]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Quantized GGUF created: {output_path}")
            f16_output.unlink()  # Remove intermediate F16 file
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Quantization failed, using F16: {e}")
            shutil.move(f16_output, output_path)
    else:
        shutil.move(f16_output, output_path)
    
    return output_path


def convert_with_pip_package(model_path: Path, output_path: Path, quantization: str):
    """
    Alternative conversion using transformers + gguf library.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("❌ transformers not installed. Run: pip install transformers torch")
        return None
    
    print("🔄 Loading model for conversion...")
    
    # Check if gguf conversion is available via transformers
    try:
        # Try using the convert script from transformers
        cmd = [
            sys.executable, "-m", "transformers.gguf",
            "--model", str(model_path),
            "--output", str(output_path)
        ]
        subprocess.run(cmd, check=True)
        return output_path
    except Exception as e:
        print(f"⚠️ Direct conversion not available: {e}")
        print("\n📋 Manual GGUF conversion instructions:")
        print("=" * 50)
        print("1. Clone llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp && make")
        print("")
        print("2. Convert the model:")
        print(f"   python convert_hf_to_gguf.py {model_path} --outfile {output_path}")
        print("")
        print("3. (Optional) Quantize:")
        print(f"   ./quantize {output_path} {output_path.stem}-q4_k_m.gguf q4_k_m")
        print("=" * 50)
        return None


def create_gguf_readme(output_dir: Path):
    """Create a README for GGUF usage."""
    readme = """# GGUF Model for llama.cpp

## Quick Start

### Using llama.cpp CLI
```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Run inference
./main -m finance-extractor-v8-q4_k_m.gguf -p "Extract financial entities from: Rs.500 debited from A/c 1234 on 01-01-25"
```

### Using Python (llama-cpp-python)
```bash
pip install llama-cpp-python
```

```python
from llama_cpp import Llama

llm = Llama(model_path="finance-extractor-v8-q4_k_m.gguf")

output = llm(
    "Extract financial entities from: Rs.500 debited from A/c 1234 on 01-01-25",
    max_tokens=200,
    stop=["\\n\\n"]
)

print(output["choices"][0]["text"])
```

## Quantization Variants

| File | Size | Quality | Speed |
|------|------|---------|-------|
| `*-f16.gguf` | ~7.6GB | Highest | Slowest |
| `*-q8_0.gguf` | ~4GB | Very High | Fast |
| `*-q4_k_m.gguf` | ~2GB | Good | Fastest |

## Compatibility

- ✅ Linux (CPU, NVIDIA GPU, AMD GPU)
- ✅ Windows (CPU, NVIDIA GPU)
- ✅ macOS (CPU, Metal)
- ✅ Any llama.cpp compatible tool
"""
    
    with open(output_dir / "GGUF_README.md", "w") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF")
    parser.add_argument(
        "--model", 
        default="models/released/finance-extractor-v8-pytorch",
        help="Path to PyTorch model directory"
    )
    parser.add_argument(
        "--output",
        default="models/released/finance-extractor-v8-q4_k_m.gguf",
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        choices=["f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0"],
        help="Quantization type"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = convert_to_gguf(model_path, output_path, args.quantization)
    
    if result:
        print(f"\n🎉 GGUF conversion complete!")
        print(f"   Output: {result}")
        print(f"   Size: {result.stat().st_size / (1024**3):.2f} GB")
        create_gguf_readme(result.parent)
    else:
        print("\n⚠️ GGUF conversion requires manual steps (see instructions above)")


if __name__ == "__main__":
    main()
