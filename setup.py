import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists("./build"):
    os.makedirs("build")

# Generate the FMHA kernels
os.system(f"{sys.executable} csrc/composable_kernel/example/ck_tile/01_fmha/generate.py -d fwd --output_dir build --receipt 2")
os.system(f"{sys.executable} csrc/composable_kernel/example/ck_tile/01_fmha/generate.py -d bwd --output_dir build --receipt 2")

sources = [
    "csrc/flash_attn_ck/flash_api.cpp",
    "csrc/flash_attn_ck/mha_bwd.cpp",
    "csrc/flash_attn_ck/mha_fwd.cpp",
    "csrc/flash_attn_ck/mha_varlen_bwd.cpp",
    "csrc/flash_attn_ck/mha_varlen_fwd.cpp"
] + [os.path.join("build", f) for f in os.listdir("build") if f.startswith("fmha_") and f.endswith(".cpp")]

for source in sources:
    os.system(f"cp {source} {os.path.splitext(source)[0]}.hip")

sources = [os.path.splitext(source)[0] + ".hip" for source in sources]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3", "-std=c++17",
        "-mllvm", "-enable-post-misched=0",
        "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
        "-fgpu-flush-denormals-to-zero",
        "-DCK_ENABLE_BF16", "-DCK_ENABLE_FP16", "-DCK_ENABLE_FP32",
        "-DCK_USE_XDL",
        "-DUSE_PROF_API=1",
        "-D__HIP_PLATFORM_HCC__=1",
        "--offload-arch=gfx942"
    ],
}

include_dirs = [
    os.path.join(this_dir, "csrc", "composable_kernel", "include"),
    os.path.join(this_dir, "csrc", "composable_kernel", "library", "include"),
    os.path.join(this_dir, "csrc", "composable_kernel", "example", "ck_tile", "01_fmha"),
    os.path.join(this_dir, "csrc", "flash_attn_ck"),
    os.path.join(this_dir, "build"),
]

ext_modules = [
    CUDAExtension(
        name="flash_attn_2_cuda",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    )
]

setup(
    name="flash_attn",
    version="2.6.1",
    author="Tri Dao",
    author_email="trid@cs.stanford.edu",
    description="Flash Attention 2 for ROCm",
    long_description="ROCm-specific implementation of Flash Attention 2",
    long_description_content_type="text/markdown",
    url="https://github.com/ROCm/flash-attention/tree/ck_tile",
    packages=find_packages(exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=1.9.0",
        "einops",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: AMD GPGPU",
    ],
    platforms=["linux_x86_64"],
)