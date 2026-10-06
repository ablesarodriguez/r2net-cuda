"""Round-trip checks: compress an image, decompress it and compare with the original.

    python -m unittest discover tests

The CPU coders always run. The CUDA coders run only when CuPy and a GPU are available.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coder_original import compress_v1, decompress_v1
from coder_parallel_cpu import compress_parallel_cpu, decompress_parallel_cpu
from coder_wavefront_cpu import compress_wf_cpu, decompress_wf_cpu
from utils.metrics import calculate_metrics
from utils.normalize import denormalize_min_max_values, normalize_min_max_values

try:
    import cupy
    cupy.cuda.Device(0).compute_capability  # raises if there is no usable GPU
    from coder_parallel_cuda import compress_parallel_cuda, decompress_parallel_cuda
    from coder_wavefront_cuda import compress_wf_cuda, decompress_wf_cuda
    HAVE_GPU = True
except Exception:
    HAVE_GPU = False

CONTEXT_SIZE = 16
BLOCK_SIZES = (4, 8, 16)
LEARNING_RATES = (0.05, 0.1, 0.5, 0.7)
DEFAULT_LEARNING_RATE = 0.5


def smooth_image(height, width):
    """Synthetic image with gradients and waves, similar in spirit to a natural image."""
    y, x = np.mgrid[0:height, 0:width]
    values = 127 + 60 * np.sin(x / 9.0) + 50 * np.cos(y / 7.0) + 0.3 * x
    return values.clip(0, 255).astype(np.uint8)


def noise_image(height, width):
    """Uniform noise: the worst case for a predictor."""
    return np.random.default_rng(0).integers(0, 256, (height, width), dtype=np.uint8)


IMAGES = {
    'smooth, multiple of the block size': smooth_image(64, 96),
    'smooth, needs padding': smooth_image(50, 70),
    'noise, needs padding': noise_image(50, 70),
    'flat colour': np.full((32, 32), 7, dtype=np.uint8),
}


def roundtrip_sequential(norm, block, lr, shape):
    residual = compress_v1(norm, block, lr, CONTEXT_SIZE)
    return residual, decompress_v1(residual, block, lr, CONTEXT_SIZE, original_shape=shape)


def roundtrip_wavefront_cpu(norm, block, lr, shape):
    residual = compress_wf_cpu(norm, block, lr, CONTEXT_SIZE)
    return residual, decompress_wf_cpu(residual, block, lr, CONTEXT_SIZE, original_shape=shape)


def roundtrip_parallel_cpu(norm, block, lr, shape):
    residual, context = compress_parallel_cpu(norm, block, lr, CONTEXT_SIZE)
    return residual, decompress_parallel_cpu(residual, context, block, lr, CONTEXT_SIZE, original_shape=shape)


def roundtrip_wavefront_cuda(norm, block, lr, shape):
    residual = compress_wf_cuda(norm, block, lr, CONTEXT_SIZE)
    return residual, decompress_wf_cuda(residual, block, lr, CONTEXT_SIZE, original_shape=shape)


def roundtrip_parallel_cuda(norm, block, lr, shape):
    residual = compress_parallel_cuda(norm, block, lr, CONTEXT_SIZE)
    return residual, decompress_parallel_cuda(residual, block, lr, CONTEXT_SIZE, original_shape=shape)


CPU_CODERS = {
    'sequential': roundtrip_sequential,
    'wavefront cpu': roundtrip_wavefront_cpu,
    'parallel cpu': roundtrip_parallel_cpu,
}
GPU_CODERS = {
    'wavefront cuda': roundtrip_wavefront_cuda,
    'parallel cuda': roundtrip_parallel_cuda,
}


def run(coder, image, dtype, block, lr):
    """Returns the residual and the metrics of the reconstructed image."""
    low, high, norm = normalize_min_max_values(image, dtype)
    residual, reconstructed = coder(norm, block, dtype(lr), image.shape)
    restored = denormalize_min_max_values(reconstructed, low, high)
    return residual, restored, calculate_metrics(image, restored, bits=8)


class RoundTripMixin:
    """The same checks for any group of coders. Subclasses set CODERS."""

    CODERS = {}

    def cases(self, learning_rates=LEARNING_RATES):
        return [(coder_name, coder, image_name, image, block, lr)
                for coder_name, coder in self.CODERS.items()
                for image_name, image in IMAGES.items()
                for block in BLOCK_SIZES
                for lr in learning_rates]

    def test_float64_is_lossless(self):
        for coder_name, coder, image_name, image, block, lr in self.cases():
            with self.subTest(coder=coder_name, image=image_name, block=block, lr=lr):
                _, restored, metrics = run(coder, image, np.float64, block, lr)
                self.assertEqual(restored.shape, image.shape)
                self.assertEqual(metrics['PAE'], 0)
                self.assertTrue(np.array_equal(restored, image))

    def test_float32_is_near_lossless_at_the_default_learning_rate(self):
        # float32 is only checked at the default learning rate: with small ones (0.05, 0.1)
        # the single-precision reconstruction loses accuracy quickly.
        for coder_name, coder, image_name, image, block, lr in self.cases([DEFAULT_LEARNING_RATE]):
            with self.subTest(coder=coder_name, image=image_name, block=block):
                _, _, metrics = run(coder, image, np.float32, block, lr)
                self.assertGreater(metrics['PSNR'], 40.0)

    def test_residual_is_finite_and_keeps_the_precision(self):
        for coder_name, coder, image_name, image, block, lr in self.cases([DEFAULT_LEARNING_RATE]):
            for dtype in (np.float64, np.float32):
                with self.subTest(coder=coder_name, image=image_name, block=block, dtype=dtype.__name__):
                    residual, _, _ = run(coder, image, dtype, block, lr)
                    self.assertEqual(residual.dtype, dtype)
                    self.assertTrue(np.isfinite(residual).all())

    def test_residual_covers_the_padded_image(self):
        image = IMAGES['smooth, needs padding']  # 50 x 70
        for coder_name, coder in self.CODERS.items():
            for block, padded in ((4, (52, 72)), (8, (56, 72)), (16, (64, 80))):
                with self.subTest(coder=coder_name, block=block):
                    residual, _, _ = run(coder, image, np.float64, block, DEFAULT_LEARNING_RATE)
                    self.assertEqual(residual.shape, padded)


class CpuCodersTest(RoundTripMixin, unittest.TestCase):
    CODERS = CPU_CODERS


@unittest.skipUnless(HAVE_GPU, 'CuPy with a CUDA GPU is required')
class GpuCodersTest(RoundTripMixin, unittest.TestCase):
    CODERS = GPU_CODERS


if __name__ == '__main__':
    unittest.main()
