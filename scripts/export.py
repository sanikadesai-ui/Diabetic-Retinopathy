#!/usr/bin/env python
"""
Export script for DR-SAFE pipeline.

Export models to ONNX, TorchScript, or other formats for deployment.

Usage:
    python scripts/export.py --checkpoint outputs/checkpoints/best.pth --format onnx
    python scripts/export.py --checkpoint outputs/checkpoints/best.pth --format torchscript
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
import torch
import yaml
from typing import Optional, Tuple

app = typer.Typer(help="Export DR-SAFE models for deployment")


@app.command()
def export(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
    ),
    format: str = typer.Option(
        "onnx",
        "--format", "-f",
        help="Export format: onnx, torchscript, or jit",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output path (default: based on checkpoint name)",
    ),
    image_size: int = typer.Option(
        512,
        "--image-size",
        help="Input image size",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size", "-b",
        help="Batch size for export (1 for variable batch)",
    ),
    dynamic_batch: bool = typer.Option(
        True,
        "--dynamic-batch/--static-batch",
        help="Allow dynamic batch size",
    ),
    opset_version: int = typer.Option(
        14,
        "--opset",
        help="ONNX opset version",
    ),
    simplify: bool = typer.Option(
        True,
        "--simplify/--no-simplify",
        help="Simplify ONNX model (requires onnx-simplifier)",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify exported model",
    ),
    device: str = typer.Option(
        "cpu",
        "--device", "-d",
        help="Device for export (cpu recommended for ONNX)",
    ),
):
    """
    Export a trained model for deployment.
    
    Example:
        python scripts/export.py --checkpoint model.pth --format onnx --output model.onnx
    """
    from drsafe.utils.logging import setup_logger
    from drsafe.utils.io import load_checkpoint
    from drsafe.models.model import load_model_from_checkpoint
    
    logger = setup_logger("drsafe")
    
    # Determine output path
    checkpoint_path = Path(checkpoint)
    if output is None:
        if format == "onnx":
            output = checkpoint_path.with_suffix(".onnx")
        elif format in ["torchscript", "jit"]:
            output = checkpoint_path.with_suffix(".pt")
        else:
            output = checkpoint_path.with_suffix(f".{format}")
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from: {checkpoint}")
    logger.info(f"Export format: {format}")
    logger.info(f"Output path: {output_path}")
    
    # Load model
    device = torch.device(device)
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    if format == "onnx":
        export_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            image_size=image_size,
            dynamic_batch=dynamic_batch,
            opset_version=opset_version,
            simplify=simplify,
            verify=verify,
        )
    elif format in ["torchscript", "jit"]:
        export_torchscript(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            verify=verify,
        )
    else:
        raise ValueError(f"Unknown export format: {format}")
    
    logger.info(f"Model exported successfully to {output_path}")
    
    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported model size: {file_size_mb:.2f} MB")


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    image_size: int,
    dynamic_batch: bool,
    opset_version: int,
    simplify: bool,
    verify: bool,
):
    """Export model to ONNX format."""
    from drsafe.utils.logging import get_logger
    
    logger = get_logger()
    
    # Define input/output names
    input_names = ["input"]
    output_names = ["severity_logits", "referable_logits"]
    
    # Define dynamic axes if needed
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "severity_logits": {0: "batch_size"},
            "referable_logits": {0: "batch_size"},
        }
    else:
        dynamic_axes = None
    
    # Export
    logger.info("Exporting to ONNX...")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    logger.info("ONNX export complete")
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            logger.info("Simplifying ONNX model...")
            
            onnx_model = onnx.load(str(output_path))
            simplified_model, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, str(output_path))
                logger.info("ONNX simplification complete")
            else:
                logger.warning("ONNX simplification check failed")
                
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")
    
    # Verify if requested
    if verify:
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np
            
            logger.info("Verifying ONNX model...")
            
            # Check model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Run inference
            ort_session = ort.InferenceSession(str(output_path))
            
            # Compare outputs
            with torch.no_grad():
                pytorch_outputs = model(dummy_input)
            
            ort_inputs = {
                "input": dummy_input.cpu().numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Check severity output
            np.testing.assert_allclose(
                pytorch_outputs[0].cpu().numpy(),
                ort_outputs[0],
                rtol=1e-3,
                atol=1e-5,
            )
            
            # Check referable output
            np.testing.assert_allclose(
                pytorch_outputs[1].cpu().numpy(),
                ort_outputs[1],
                rtol=1e-3,
                atol=1e-5,
            )
            
            logger.info("ONNX verification passed ✓")
            
        except ImportError:
            logger.warning("onnxruntime not installed, skipping verification")
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")


def export_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    verify: bool,
):
    """Export model to TorchScript format."""
    from drsafe.utils.logging import get_logger
    
    logger = get_logger()
    
    logger.info("Exporting to TorchScript...")
    
    # Use tracing
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save
    traced_model.save(str(output_path))
    
    logger.info("TorchScript export complete")
    
    # Verify if requested
    if verify:
        logger.info("Verifying TorchScript model...")
        
        # Load and run
        loaded_model = torch.jit.load(str(output_path))
        
        with torch.no_grad():
            pytorch_outputs = model(dummy_input)
            jit_outputs = loaded_model(dummy_input)
        
        # Compare
        import numpy as np
        
        np.testing.assert_allclose(
            pytorch_outputs[0].cpu().numpy(),
            jit_outputs[0].cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        
        np.testing.assert_allclose(
            pytorch_outputs[1].cpu().numpy(),
            jit_outputs[1].cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        
        logger.info("TorchScript verification passed ✓")


@app.command()
def benchmark(
    model_path: str = typer.Option(
        ...,
        "--model", "-m",
        help="Path to exported model (ONNX or TorchScript)",
    ),
    image_size: int = typer.Option(
        512,
        "--image-size",
        help="Input image size",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size", "-b",
        help="Batch size for benchmarking",
    ),
    num_iterations: int = typer.Option(
        100,
        "--iterations", "-n",
        help="Number of iterations",
    ),
    warmup: int = typer.Option(
        10,
        "--warmup",
        help="Number of warmup iterations",
    ),
):
    """
    Benchmark exported model inference speed.
    
    Example:
        python scripts/export.py benchmark --model model.onnx --batch-size 1
    """
    import time
    import numpy as np
    
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    
    print(f"Benchmarking: {model_path}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")
    
    # Create dummy input
    dummy_input = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    
    if suffix == ".onnx":
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            times.append(time.perf_counter() - start)
            
    elif suffix == ".pt":
        model = torch.jit.load(str(model_path))
        model.eval()
        
        dummy_tensor = torch.from_numpy(dummy_input)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                model(dummy_tensor)
                times.append(time.perf_counter() - start)
    else:
        raise ValueError(f"Unknown model format: {suffix}")
    
    # Calculate statistics
    times = np.array(times) * 1000  # Convert to ms
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\nLatency (ms):")
    print(f"  Mean: {np.mean(times):.2f}")
    print(f"  Std: {np.std(times):.2f}")
    print(f"  Min: {np.min(times):.2f}")
    print(f"  Max: {np.max(times):.2f}")
    print(f"  P50: {np.percentile(times, 50):.2f}")
    print(f"  P95: {np.percentile(times, 95):.2f}")
    print(f"  P99: {np.percentile(times, 99):.2f}")
    print(f"\nThroughput: {1000 / np.mean(times) * batch_size:.1f} images/sec")


if __name__ == "__main__":
    app()
