#!/usr/bin/env python3
import io
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check-dependency-graph.py"
RULES = REPO_ROOT / "config" / "include_rules.json"
PRODUCT_SOURCE_ROOTS = ("cpp", "python/hikoboshi")
PRODUCT_SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".py", ".pyi"}
LEGACY_SYMBOL_PATTERNS = {
    "batch_pairwise": re.compile(r"\bbatch_pairwise\b"),
    "compute_distances": re.compile(r"\bcompute_distances\b"),
    "distance_matrix": re.compile(r"\bdistance_matrix\b"),
    "MSA": re.compile(r"\b(?:MSA|msa)\b"),
    "tree": re.compile(r"\b(?:tree|build_tree|guide_tree)\b"),
    "profile-merge": re.compile(r"\bprofile_merge\b|\bprofile-merge\b"),
    "_align_cpp": re.compile(r"\b_align_cpp\b"),
}
C_LIKE_COMMENT_OR_STRING_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def prepare_fixture(root: Path) -> None:
    (root / "config").mkdir(parents=True)
    shutil.copyfile(RULES, root / "config" / "include_rules.json")


def write_source(root: Path, relpath: str, include_line: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"// fixture\n\n{include_line}\n")


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def require_failure(relpath: str, include_line: str, expected: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare_fixture(root)
        write_source(root, relpath, include_line)
        result = run_checker(root)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            raise SystemExit(f"expected dependency graph failure for {relpath}")
        if expected not in output:
            raise SystemExit(
                f"missing expected diagnostic for {relpath}\n"
                f"expected: {expected}\n"
                f"output:\n{output}"
            )


def require_success(relpath: str, include_line: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare_fixture(root)
        write_source(root, relpath, include_line)
        result = run_checker(root)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise SystemExit(f"expected dependency graph success for {relpath}\n{output}")


def require_failures(cases: list[tuple[str, str, str]]) -> None:
    for relpath, include_line, expected in cases:
        require_failure(relpath, include_line, expected)


def blank_preserving_newlines(match: re.Match[str]) -> str:
    return "\n" * match.group(0).count("\n")


def searchable_source_text(path: Path) -> str:
    text = path.read_text(errors="ignore")
    if path.suffix in {".py", ".pyi"}:
        try:
            tokens = tokenize.generate_tokens(io.StringIO(text).readline)
            return " ".join(
                token.string
                for token in tokens
                if token.type not in {tokenize.COMMENT, tokenize.STRING}
            )
        except tokenize.TokenError:
            return text
    return C_LIKE_COMMENT_OR_STRING_RE.sub(blank_preserving_newlines, text)


def iter_product_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    for source_root in PRODUCT_SOURCE_ROOTS:
        base = root / source_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in PRODUCT_SOURCE_SUFFIXES:
                sources.append(path)
    return sources


def legacy_symbol_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for path in iter_product_sources(root):
        searchable = searchable_source_text(path)
        relpath = path.relative_to(root).as_posix()
        for label, pattern in LEGACY_SYMBOL_PATTERNS.items():
            if pattern.search(searchable):
                violations.append(f"{relpath}: legacy symbol reappeared: {label}")
    return violations


def require_no_legacy_symbols(root: Path = REPO_ROOT) -> None:
    violations = legacy_symbol_violations(root)
    if violations:
        raise SystemExit("legacy symbol check failed\n" + "\n".join(violations))


def require_legacy_symbol_failure(relpath: str, body: str, expected: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        output = "\n".join(legacy_symbol_violations(root))
        if expected not in output:
            raise SystemExit(
                f"missing expected legacy symbol diagnostic for {relpath}\n"
                f"expected: {expected}\n"
                f"output:\n{output}"
            )


def require_python_import_failure(relpath: str, body: str, expected: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare_fixture(root)
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        result = run_checker(root)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            raise SystemExit(f"expected dependency graph failure for {relpath}")
        if expected not in output:
            raise SystemExit(
                f"missing expected Python import diagnostic for {relpath}\n"
                f"expected: {expected}\n"
                f"output:\n{output}"
            )


def require_success_with_python_cpp_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare_fixture(root)
        write_source(root, "python/hikoboshi/native.cpp", "#include <hikoboshi/io/parser.hpp>")
        result = run_checker(root)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise SystemExit(f"python/hikoboshi C++ file should not be layer-owned\n{output}")


def require_empty_skeleton_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare_fixture(root)
        result = run_checker(root)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise SystemExit(f"empty source skeleton should pass\n{output}")


def main() -> int:
    require_empty_skeleton_success()
    require_success_with_python_cpp_ignored()
    require_success("cpp/dispatch/cuda/good.cpp", "#include <cuda_runtime.h>")
    require_success("cpp/dispatch/hip/good.cpp", "#include <hip/hip_runtime.h>")
    require_success("cpp/dispatch/metal/good.mm", "#include <Metal/Metal.h>")
    require_success("cpp/dispatch/vulkan/good.cpp", "#include <vulkan/vulkan.h>")
    require_success("cpp/dispatch/opencl/good.cpp", "#include <CL/cl.h>")
    require_success("cpp/dispatch/simd/good_x86.cpp", "#include <immintrin.h>")
    require_success("cpp/dispatch/simd/good_arm.cpp", "#include <arm_neon.h>")
    require_no_legacy_symbols()
    for symbol in LEGACY_SYMBOL_PATTERNS:
        require_legacy_symbol_failure(
            f"python/hikoboshi/{symbol.replace('-', '_')}.py",
            f"{symbol.replace('-', '_')} = object()\n",
            f"python/hikoboshi/{symbol.replace('-', '_')}.py: legacy symbol reappeared: {symbol}",
        )
    require_failures(
        [
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/io/parser.hpp>",
                "cpp/api/bad.cpp:3: forbidden include edge api -> io: "
                "hikoboshi/io/parser.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/io/fasta_writer.hpp>",
                "cpp/api/bad.cpp:3: forbidden include edge api -> io: "
                "hikoboshi/io/fasta_writer.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/io/structure_loader.hpp>",
                "cpp/api/bad.cpp:3: forbidden include edge api -> io: "
                "hikoboshi/io/structure_loader.hpp",
            ),
            (
                "cpp/universal/bad.cpp",
                "#include <hikoboshi/api/all_vs_all.hpp>",
                "cpp/universal/bad.cpp:3: forbidden include edge universal -> api: "
                "hikoboshi/api/all_vs_all.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/weights/generated/mpnn64.hpp>",
                "cpp/api/bad.cpp:3: forbidden generated weight include from api: "
                "hikoboshi/weights/generated/mpnn64.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <cpp/weights/generated/mpnn_d64_blob.hpp>",
                "cpp/api/bad.cpp:3: forbidden generated weight include from api: "
                "cpp/weights/generated/mpnn_d64_blob.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                '#include "../weights/generated/mpnn_d64_blob.hpp"',
                "cpp/api/bad.cpp:3: forbidden include edge api -> weights: "
                "../weights/generated/mpnn_d64_blob.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/weights/provider.hpp>",
                "cpp/api/bad.cpp:3: forbidden include edge api -> weights: "
                "hikoboshi/weights/provider.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                '#include "generated_mpnn64_weights.hpp"',
                "cpp/api/bad.cpp:3: forbidden generated weight include from api: "
                "generated_mpnn64_weights.hpp",
            ),
            (
                "cpp/algorithms/bad.cpp",
                "#include <hikoboshi/weights/provider.hpp>",
                "cpp/algorithms/bad.cpp:3: forbidden include edge algorithms -> weights: "
                "hikoboshi/weights/provider.hpp",
            ),
            (
                "cpp/algorithms/all_vs_all/bad.cpp",
                "#include <hikoboshi/primitives/alignment/smith_waterman.hpp>",
                "cpp/algorithms/all_vs_all/bad.cpp:3: forbidden include edge "
                "algorithms -> primitives: hikoboshi/primitives/alignment/smith_waterman.hpp",
            ),
            (
                "cpp/algorithms/all_vs_all/bad.cpp",
                '#include "../../primitives/linalg/gemm.hpp"',
                "cpp/algorithms/all_vs_all/bad.cpp:3: forbidden include edge "
                "algorithms -> primitives: ../../primitives/linalg/gemm.hpp",
            ),
            (
                "cpp/modules/bad.cpp",
                "#include <hikoboshi/algorithms/workflow.hpp>",
                "cpp/modules/bad.cpp:3: forbidden include edge modules -> algorithms: "
                "hikoboshi/algorithms/workflow.hpp",
            ),
            (
                "cpp/runtime/bad.cpp",
                "#include <hikoboshi/modules/mpnn.hpp>",
                "cpp/runtime/bad.cpp:3: forbidden include edge runtime -> modules: "
                "hikoboshi/modules/mpnn.hpp",
            ),
            (
                "cpp/runtime/bad.cpp",
                "#include <hikoboshi/algorithms/pairwise.hpp>",
                "cpp/runtime/bad.cpp:3: forbidden include edge runtime -> algorithms: "
                "hikoboshi/algorithms/pairwise.hpp",
            ),
            (
                "cpp/runtime/bad.cpp",
                "#include <hikoboshi/api/engine.hpp>",
                "cpp/runtime/bad.cpp:3: forbidden include edge runtime -> api: "
                "hikoboshi/api/engine.hpp",
            ),
            (
                "cpp/runtime/bad.cpp",
                "#include <hikoboshi/io/structure_loader.hpp>",
                "cpp/runtime/bad.cpp:3: forbidden include edge runtime -> io: "
                "hikoboshi/io/structure_loader.hpp",
            ),
            (
                "cpp/io/bad.cpp",
                "#include <hikoboshi/primitives/linalg/gemm.hpp>",
                "cpp/io/bad.cpp:3: forbidden include edge io -> primitives: "
                "hikoboshi/primitives/linalg/gemm.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/primitives/kernel.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> primitives: "
                "hikoboshi/primitives/kernel.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/dispatch/scalar_forward.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> dispatch: "
                "hikoboshi/dispatch/scalar_forward.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/modules/mpnn.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> modules: "
                "hikoboshi/modules/mpnn.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/algorithms/all_vs_all.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> algorithms: "
                "hikoboshi/algorithms/all_vs_all.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/universal/alignment_path.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> universal: "
                "hikoboshi/universal/alignment_path.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/primitives/alignment/traceback.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> primitives: "
                "hikoboshi/primitives/alignment/traceback.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/dispatch/backend_tag.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> dispatch: "
                "hikoboshi/dispatch/backend_tag.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/modules/similarity.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> modules: "
                "hikoboshi/modules/similarity.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/algorithms/pairwise.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> algorithms: "
                "hikoboshi/algorithms/pairwise.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/errors/format.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> errors: "
                "hikoboshi/errors/format.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                '#include "../algorithms/workflow.hpp"',
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> algorithms: "
                "../algorithms/workflow.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/bindings/_align_cpp.hpp>",
                "cpp/api/bad.cpp:3: forbidden include edge api -> bindings: "
                "hikoboshi/bindings/_align_cpp.hpp",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hikoboshi/io/distance_matrix.hpp>",
                "cpp/api/bad.cpp:3: forbidden include edge api -> io: "
                "hikoboshi/io/distance_matrix.hpp",
            ),
            (
                "cpp/universal/bad.cpp",
                "#include <hikoboshi/api/batch_pairwise.hpp>",
                "cpp/universal/bad.cpp:3: forbidden include edge universal -> api: "
                "hikoboshi/api/batch_pairwise.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/algorithms/compute_distances.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> algorithms: "
                "hikoboshi/algorithms/compute_distances.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/algorithms/msa.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> algorithms: "
                "hikoboshi/algorithms/msa.hpp",
            ),
            (
                "cpp/bindings/bad.cpp",
                "#include <hikoboshi/algorithms/tree.hpp>",
                "cpp/bindings/bad.cpp:3: forbidden include edge bindings -> algorithms: "
                "hikoboshi/algorithms/tree.hpp",
            ),
            (
                "cpp/cli/bad.cpp",
                "#include <hikoboshi/modules/profile_merge.hpp>",
                "cpp/cli/bad.cpp:3: forbidden include edge cli -> modules: "
                "hikoboshi/modules/profile_merge.hpp",
            ),
        ]
    )
    require_failures(
        [
            (
                "cpp/api/bad.cpp",
                "#include <cuda_runtime.h>",
                "cpp/api/bad.cpp:3: forbidden CUDA SDK include outside "
                "cpp/dispatch/cuda: cuda_runtime.h",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <hip/hip_runtime.h>",
                "cpp/api/bad.cpp:3: forbidden HIP SDK include outside "
                "cpp/dispatch/hip: hip/hip_runtime.h",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <Metal/Metal.h>",
                "cpp/api/bad.cpp:3: forbidden Metal SDK include outside "
                "cpp/dispatch/metal: Metal/Metal.h",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <vulkan/vulkan.h>",
                "cpp/api/bad.cpp:3: forbidden Vulkan SDK include outside "
                "cpp/dispatch/vulkan: vulkan/vulkan.h",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <CL/cl.h>",
                "cpp/api/bad.cpp:3: forbidden OpenCL SDK include outside "
                "cpp/dispatch/opencl: CL/cl.h",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <immintrin.h>",
                "cpp/api/bad.cpp:3: forbidden x86 SIMD SDK include outside "
                "cpp/dispatch/simd: immintrin.h",
            ),
            (
                "cpp/api/bad.cpp",
                "#include <arm_neon.h>",
                "cpp/api/bad.cpp:3: forbidden ARM SIMD SDK include outside "
                "cpp/dispatch/simd: arm_neon.h",
            ),
            (
                "cpp/dispatch/bad.cpp",
                "#include <cuda_runtime.h>",
                "cpp/dispatch/bad.cpp:3: forbidden CUDA SDK include outside "
                "cpp/dispatch/cuda: cuda_runtime.h",
            ),
        ]
    )
    for layer in (
        "api",
        "universal",
        "algorithms",
        "modules",
        "primitives",
        "io",
        "errors",
        "cli",
        "bindings",
    ):
        require_failure(
            f"cpp/{layer}/bad.cpp",
            "#include <hikoboshi/dispatch/cuda/kernel.hpp>",
            f"cpp/{layer}/bad.cpp:3: forbidden backend dispatch include from "
            f"{layer}: hikoboshi/dispatch/cuda/kernel.hpp",
        )
    require_failure(
        "cpp/modules/bad.cpp",
        '#include "../dispatch/cuda/kernel.hpp"',
        "cpp/modules/bad.cpp:3: forbidden backend dispatch include from "
        "modules: ../dispatch/cuda/kernel.hpp",
    )
    require_python_import_failure(
        "python/hikoboshi/bad.py",
        "from hikoboshi.dispatch.cuda import kernel\n",
        "python/hikoboshi/bad.py:1: forbidden backend dispatch import from "
        "python: hikoboshi.dispatch.cuda",
    )
    print("dependency graph negative fixture: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
