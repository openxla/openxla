"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "c54616ea481aa8fb48e113f4832b6df8ca8b2a99"
    LLVM_SHA256 = "1bf83c67a1826e7fc01310ba6544a065591069ef90539705a0b08cbe69047507"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = [
            "//third_party/llvm:generated.patch",  # Autogenerated, don't remove.
            "//third_party/llvm:build.patch",
            "//third_party/llvm:mathextras.patch",
            "//third_party/llvm:toolchains.patch",
            "//third_party/llvm:zstd.patch",
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
