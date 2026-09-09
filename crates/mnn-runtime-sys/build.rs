use std::{
    env,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
};

mod build_support;
mod prebuilt;
use build_support::{cuda_side_library, cuda_target_supported, ios_sdk, static_cpp_libraries};
use prebuilt::{
    select_prebuilt, PrebuiltArtifact, PrebuiltFeatures, MNN_SOURCE_REVISION, MNN_SOURCE_SHA256,
    MNN_SOURCE_VERSION, PREBUILT_REPOSITORY, PREBUILT_TAG,
};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, PartialEq, Eq)]
enum LinkMode {
    Static,
    Dynamic,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=prebuilt.rs");
    println!("cargo:rerun-if-changed=cpp/mnn_runtime_bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/mnn_runtime_bridge.h");
    for variable in [
        "MNN_INCLUDE_DIR",
        "MNN_LIB_DIR",
        "MNN_PREBUILT_CACHE_DIR",
        "MNN_PREBUILT_BASE_URL",
        "MNN_PREBUILT_TAG",
        "MNN_PREBUILT_SHA256",
        "MNN_REQUIRE_PREBUILT",
        "MNN_SOURCE_DIR",
        "MNN_OPENMP_LIB",
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_TOOLKIT_ROOT_DIR",
        "MNN_CUDA_ARCHS",
        "DOCS_RS",
        "ANDROID_NDK_ROOT",
        "ANDROID_NDK_HOME",
        "ANDROID_NDK",
        "NDK_HOME",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    if cfg!(feature = "openmp") && cfg!(feature = "mnn-threadpool") {
        panic!("features `openmp` and `mnn-threadpool` are mutually exclusive");
    }

    let link_mode = match (cfg!(feature = "static"), cfg!(feature = "dynamic")) {
        (true, true) => panic!("features `static` and `dynamic` are mutually exclusive"),
        (false, true) => LinkMode::Dynamic,
        _ => LinkMode::Static,
    };
    let target_os = required_env("CARGO_CFG_TARGET_OS");
    let target_arch = required_env("CARGO_CFG_TARGET_ARCH");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target = required_env("TARGET");

    if cfg!(feature = "crt-static") && !target_crt_static() {
        panic!(
            "feature `crt-static` requires RUSTFLAGS='-C target-feature=+crt-static' so Rust and C++ use the same CRT"
        );
    }
    if cfg!(feature = "openmp") && matches!(target_os.as_str(), "macos" | "ios") {
        panic!("this MNN baseline does not support OpenMP on Apple targets");
    }
    if cfg!(feature = "cuda") && !cuda_target_supported(&target_os, &target_env) {
        panic!("feature `cuda` requires Linux or Windows MSVC and the NVIDIA CUDA toolkit");
    }
    let artifact = select_prebuilt(
        &target_os,
        &target_arch,
        &target_env,
        &target,
        PrebuiltFeatures {
            metal: cfg!(feature = "metal"),
            coreml: cfg!(feature = "coreml"),
            opencl: cfg!(feature = "opencl"),
            opengl: cfg!(feature = "opengl"),
            vulkan: cfg!(feature = "vulkan"),
            cuda: cfg!(feature = "cuda"),
            threadpool: cfg!(feature = "mnn-threadpool"),
            openmp: cfg!(feature = "openmp"),
            crt_static: target_crt_static(),
            dynamic: link_mode == LinkMode::Dynamic,
        },
    );
    let (include_dir, lib_dir, prebuilt) = if let Some(external) = external_installation() {
        (external.0, external.1, None)
    } else if let Some(artifact) =
        artifact.filter(|_| cfg!(feature = "prebuilt") && !cfg!(feature = "build-from-source"))
    {
        if artifact.has_backend("cuda") {
            validate_cuda_prebuilt();
        }
        let installation = acquire_prebuilt(artifact, &target_os);
        (installation.0, installation.1, Some(artifact))
    } else {
        assert!(
            env::var("MNN_REQUIRE_PREBUILT").as_deref() != Ok("1"),
            "no compatible prebuilt for `{target}` with the selected features/CRT/threading; source fallback disabled by MNN_REQUIRE_PREBUILT=1"
        );
        println!("cargo:warning=No compatible prebuilt selected; compiling MNN from source");
        let installation = build_from_source(&target_os, &target_arch, &target_env, &target);
        (installation.0, installation.1, None)
    };

    build_bridge(&include_dir, &target_env, link_mode);
    link_mnn(&lib_dir, &target_os, &target_env, link_mode, prebuilt);
    println!("cargo:metadata=include={}", include_dir.display());
}

fn external_installation() -> Option<(PathBuf, PathBuf)> {
    let include = env::var_os("MNN_INCLUDE_DIR").map(PathBuf::from);
    let library = env::var_os("MNN_LIB_DIR").map(PathBuf::from);
    match (include, library) {
        (None, None) => None,
        (Some(include), Some(library)) => {
            validate_directory(&include, "MNN_INCLUDE_DIR");
            validate_directory(&library, "MNN_LIB_DIR");
            Some((include, library))
        }
        _ => panic!("MNN_INCLUDE_DIR and MNN_LIB_DIR must be set together"),
    }
}

fn acquire_prebuilt(artifact: &PrebuiltArtifact, os: &str) -> (PathBuf, PathBuf) {
    let tag = env::var("MNN_PREBUILT_TAG").unwrap_or_else(|_| PREBUILT_TAG.to_owned());
    let asset = format!("{tag}-{}", artifact.suffix);
    let extension = if os == "windows" { "zip" } else { "tar.gz" };
    let checksum = env::var("MNN_PREBUILT_SHA256")
        .ok()
        .or_else(|| (tag == PREBUILT_TAG).then(|| artifact.sha256.to_owned()))
        .unwrap_or_else(|| {
            panic!("MNN_PREBUILT_SHA256 is required when MNN_PREBUILT_TAG is not `{PREBUILT_TAG}`")
        });

    let out_dir = PathBuf::from(required_env("OUT_DIR"));
    let cache = env::var_os("MNN_PREBUILT_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| out_dir.join("downloads"));
    fs::create_dir_all(&cache).expect("failed to create MNN prebuilt cache directory");
    let archive = cache.join(format!("{asset}.{extension}"));
    let base_url = env::var("MNN_PREBUILT_BASE_URL").unwrap_or_else(|_| {
        format!("https://github.com/{PREBUILT_REPOSITORY}/releases/download/{tag}")
    });
    let url = format!("{base_url}/{asset}.{extension}");
    ensure_download(&url, &archive, &checksum);

    let extraction = out_dir.join("prebuilt");
    let root = extraction.join(&asset);
    let stamp = extraction.join("archive.sha256");
    if !root.join("include/MNN/Interpreter.hpp").is_file()
        || fs::read_to_string(&stamp).ok().as_deref() != Some(&checksum)
    {
        if extraction.exists() {
            fs::remove_dir_all(&extraction).expect("failed to clear partial MNN extraction");
        }
        fs::create_dir_all(&extraction).expect("failed to create MNN extraction directory");
        if os == "windows" {
            extract_zip(&archive, &extraction);
        } else {
            extract_tar_gz(&archive, &extraction);
        }
        fs::write(stamp, &checksum).expect("failed to record extracted archive checksum");
    }

    println!("cargo:warning=Using verified MNN prebuilt `{asset}` from {PREBUILT_REPOSITORY}");
    (root.join("include"), root.join("lib"))
}

fn build_from_source(os: &str, arch: &str, target_env: &str, target: &str) -> (PathBuf, PathBuf) {
    let source = acquire_source();
    let mut config = cmake::Config::new(&source);
    config
        .profile("Release")
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.10")
        .define(
            "MNN_BUILD_SHARED_LIBS",
            if cfg!(feature = "dynamic") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_BUILD_DEMO", "OFF")
        .define("MNN_BUILD_TEST", "OFF")
        .define("MNN_BUILD_BENCHMARK", "OFF")
        .define("MNN_BUILD_QUANTOOLS", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_USE_SYSTEM_LIB", "OFF")
        .define("MNN_KLEIDIAI", "OFF")
        .define(
            "MNN_USE_THREAD_POOL",
            if cfg!(feature = "mnn-threadpool") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define(
            "MNN_OPENMP",
            if cfg!(feature = "openmp") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define(
            "MNN_METAL",
            if cfg!(feature = "metal") { "ON" } else { "OFF" },
        )
        .define(
            "MNN_COREML",
            if cfg!(feature = "coreml") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define(
            "MNN_OPENCL",
            if cfg!(feature = "opencl") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define(
            "MNN_OPENGL",
            if cfg!(feature = "opengl") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define(
            "MNN_VULKAN",
            if cfg!(feature = "vulkan") {
                "ON"
            } else {
                "OFF"
            },
        );

    config.define(
        "MNN_CUDA",
        if cfg!(feature = "cuda") { "ON" } else { "OFF" },
    );
    if cfg!(feature = "cuda") {
        if let Some(root) = cuda_root() {
            config.define("CUDA_TOOLKIT_ROOT_DIR", root);
        }
        if let Ok(architectures) = env::var("MNN_CUDA_ARCHS") {
            config.define("CUDA_ARCHS", architectures);
        }
    }
    if arch == "x86_64" && !matches!(os, "android" | "ios") {
        config.define("MNN_USE_SSE", "ON");
    } else {
        config
            .define("MNN_USE_SSE", "OFF")
            .define("MNN_USE_AVX", "OFF")
            .define("MNN_USE_AVX2", "OFF")
            .define("MNN_USE_AVX512", "OFF");
    }
    if os == "windows" && target_env == "msvc" {
        config.define(
            "MNN_WIN_RUNTIME_MT",
            if target_crt_static() { "ON" } else { "OFF" },
        );
    }
    if os == "android" {
        let ndk = [
            "ANDROID_NDK_ROOT",
            "ANDROID_NDK_HOME",
            "ANDROID_NDK",
            "NDK_HOME",
        ]
        .into_iter()
        .find_map(env::var_os)
        .map(PathBuf::from)
        .expect("Android source builds require ANDROID_NDK_ROOT");
        let abi = match arch {
            "aarch64" => "arm64-v8a",
            "arm" => "armeabi-v7a",
            "x86_64" => "x86_64",
            "x86" => "x86",
            _ => panic!("unsupported Android architecture `{arch}`"),
        };
        config
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                ndk.join("build/cmake/android.toolchain.cmake"),
            )
            .define("ANDROID_ABI", abi)
            .define("ANDROID_STL", "c++_static")
            .define("ANDROID_NATIVE_API_LEVEL", "android-21")
            .define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON");
    }
    if os == "ios" {
        config
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("MNN_BUILD_FOR_IOS", "ON")
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0")
            .define(
                "CMAKE_OSX_ARCHITECTURES",
                if arch == "aarch64" { "arm64" } else { "x86_64" },
            )
            .define(
                "CMAKE_OSX_SYSROOT",
                ios_sdk(target).expect("Mac Catalyst needs an external MNN installation via MNN_INCLUDE_DIR/MNN_LIB_DIR"),
            );
    }

    println!("cargo:warning=Building MNN {MNN_SOURCE_VERSION} from source for `{target}`");
    let installation = config.build();
    if let Some(library) = cuda_side_library(os, cfg!(feature = "cuda")) {
        let filename = format!("lib{library}.so");
        let source = installation
            .join("build/source/backend/cuda")
            .join(&filename);
        let destination = installation.join("lib").join(&filename);
        fs::copy(&source, &destination).unwrap_or_else(|error| {
            panic!(
                "failed to install CUDA side library {}: {error}",
                source.display()
            )
        });
    }
    (installation.join("include"), installation.join("lib"))
}

fn acquire_source() -> PathBuf {
    if let Some(source) = env::var_os("MNN_SOURCE_DIR").map(PathBuf::from) {
        if source.join("CMakeLists.txt").is_file() {
            return source;
        }
        panic!(
            "MNN_SOURCE_DIR does not contain CMakeLists.txt: {}",
            source.display()
        );
    }

    let out_dir = PathBuf::from(required_env("OUT_DIR"));
    let root = out_dir
        .join("source")
        .join(format!("MNN-{MNN_SOURCE_REVISION}"));
    if root.join("CMakeLists.txt").is_file() {
        return root;
    }
    let archive = out_dir.join(format!("MNN-{MNN_SOURCE_REVISION}.tar.gz"));
    let url = format!("https://codeload.github.com/alibaba/MNN/tar.gz/{MNN_SOURCE_REVISION}");
    ensure_download(&url, &archive, MNN_SOURCE_SHA256);
    let destination = out_dir.join("source");
    fs::create_dir_all(&destination).expect("failed to create MNN source directory");
    extract_tar_gz(&archive, &destination);
    root
}

fn build_bridge(include: &Path, target_env: &str, link_mode: LinkMode) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .cpp_link_stdlib(None::<&str>)
        .include(include)
        .include("cpp")
        .file("cpp/mnn_runtime_bridge.cpp")
        .warnings(true);
    if target_env == "msvc" {
        build
            .flag_if_supported("/std:c++17")
            .flag_if_supported("/EHsc");
        build.static_crt(target_crt_static());
        if link_mode == LinkMode::Dynamic {
            build.define("USING_MNN_DLL", None);
        }
    } else {
        build
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-fvisibility=hidden");
    }
    build.compile("mnn_runtime_bridge");
}

fn link_mnn(
    lib: &Path,
    os: &str,
    target_env: &str,
    mode: LinkMode,
    prebuilt: Option<&PrebuiltArtifact>,
) {
    println!("cargo:rustc-link-search=native={}", lib.display());
    let whole_archive = prebuilt.is_some()
        || cfg!(feature = "cuda")
        || cfg!(feature = "metal")
        || cfg!(feature = "coreml")
        || cfg!(feature = "opencl")
        || cfg!(feature = "opengl")
        || cfg!(feature = "vulkan");
    let static_kind = if whole_archive {
        "static:+whole-archive"
    } else {
        "static"
    };
    match (os, mode) {
        ("windows", LinkMode::Static) if lib.join("MNN_static.lib").is_file() => {
            println!("cargo:rustc-link-lib={static_kind}=MNN_static")
        }
        (_, LinkMode::Static) => println!("cargo:rustc-link-lib={static_kind}=MNN"),
        (_, LinkMode::Dynamic) => println!("cargo:rustc-link-lib=dylib=MNN"),
    }

    match (os, target_env) {
        ("macos" | "ios", _) => println!("cargo:rustc-link-lib=dylib=c++"),
        ("linux", _) => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        }
        ("windows", "gnu") => {
            let libraries =
                static_cpp_libraries(os, target_env, cfg!(feature = "static-cpp-runtime"));
            if libraries.is_empty() {
                println!("cargo:rustc-link-lib=dylib=stdc++");
            } else {
                for library in libraries {
                    link_toolchain_static_library(library);
                }
            }
        }
        ("android", _) => {
            link_toolchain_static_library("c++_static");
            link_toolchain_static_library("c++abi");
            println!("cargo:rustc-link-lib=log");
            println!("cargo:rustc-link-lib=android");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
        }
        _ => {}
    }
    if matches!(os, "macos" | "ios") && (prebuilt.is_some() || cfg!(feature = "metal")) {
        for framework in [
            "Foundation",
            "CoreFoundation",
            "CoreGraphics",
            "Metal",
            "MetalPerformanceShaders",
        ] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=objc");
        if os == "ios" {
            println!("cargo:rustc-link-lib=framework=UIKit");
        }
    }
    if cfg!(feature = "coreml") && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
    }
    if cfg!(feature = "cuda") {
        if let Some(root) = cuda_root() {
            for suffix in ["lib64", "lib/x64", "lib"] {
                let directory = root.join(suffix);
                if directory.is_dir() {
                    println!("cargo:rustc-link-search=native={}", directory.display());
                }
            }
        }
        if let Some(library) = cuda_side_library(os, true) {
            let companion = if mode == LinkMode::Static
                && (prebuilt.is_some() || lib.join("cuda-static").is_dir())
            {
                lib.join("cuda-static")
            } else {
                lib.to_owned()
            };
            assert!(
                companion.join(format!("lib{library}.so")).is_file(),
                "CUDA companion lib{library}.so is missing from {}",
                companion.display()
            );
            println!("cargo:rustc-link-search=native={}", companion.display());
            println!("cargo:rustc-link-lib=dylib={library}");
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
    }
    // Vulkan/OpenCL profiles use MNN's dynamic loaders, not SDK link libraries.
    // Android profiles include GLES even when only Vulkan/OpenCL was requested.
    if cfg!(feature = "opengl") || prebuilt.is_some_and(|p| p.has_backend("opengl")) {
        match os {
            "android" => {
                println!("cargo:rustc-link-lib=GLESv3");
                println!("cargo:rustc-link-lib=EGL");
            }
            "linux" => println!("cargo:rustc-link-lib=GL"),
            _ => {}
        }
    }
    if cfg!(feature = "openmp") {
        let compiler = cc::Build::new().cpp(true).get_compiler();
        let library = env::var("MNN_OPENMP_LIB").unwrap_or_else(|_| {
            if compiler.is_like_clang() {
                "omp"
            } else if compiler.is_like_msvc() {
                "vcomp"
            } else {
                "gomp"
            }
            .to_owned()
        });
        println!("cargo:rustc-link-lib={library}");
    }
}

fn cuda_root() -> Option<PathBuf> {
    ["CUDA_TOOLKIT_ROOT_DIR", "CUDA_PATH", "CUDA_HOME"]
        .iter()
        .find_map(|name| env::var_os(name).map(PathBuf::from))
        .or_else(|| {
            Path::new("/usr/local/cuda")
                .is_dir()
                .then(|| PathBuf::from("/usr/local/cuda"))
        })
}

fn link_toolchain_static_library(library: &str) {
    let archive_name = format!("lib{library}.a");
    let output = cc::Build::new()
        .cpp(true)
        .get_compiler()
        .to_command()
        .arg(format!("-print-file-name={archive_name}"))
        .output()
        .expect("failed to query target C++ compiler");
    let archive = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    assert!(
        output.status.success() && archive.is_absolute() && archive.is_file(),
        "{archive_name} is missing from the target C++ toolchain; check CXX and the SDK installation"
    );
    // The NDK runtime directory also contains libc.a. Adding that directory to
    // the search path would shadow the API-specific libc.so stubs, so isolate
    // only the requested C++ runtime archives inside OUT_DIR.
    let runtime_dir = PathBuf::from(required_env("OUT_DIR")).join("cxx-runtime");
    fs::create_dir_all(&runtime_dir).expect("failed to create C++ runtime directory");
    fs::copy(&archive, runtime_dir.join(&archive_name))
        .expect("failed to copy target C++ runtime archive");
    println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    println!("cargo:rustc-link-lib=static={library}");
}

fn validate_cuda_prebuilt() {
    let root = cuda_root().expect(
        "CUDA 12 prebuilts require CUDA Toolkit >=12.8,<13; set CUDA_PATH, CUDA_HOME or CUDA_TOOLKIT_ROOT_DIR",
    );
    let header = fs::read_to_string(root.join("include/cuda.h"))
        .expect("CUDA toolkit is missing include/cuda.h");
    let version = header.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next() == Some("#define") && fields.next() == Some("CUDA_VERSION"))
            .then(|| fields.next()?.parse::<u32>().ok())
            .flatten()
    });
    assert!(
        version.is_some_and(|version| (12080..13000).contains(&version)),
        "CUDA 12 prebuilts require Toolkit >=12.8,<13; use build-from-source for a different toolkit"
    );
}

fn ensure_download(url: &str, destination: &Path, expected_sha256: &str) {
    if destination.is_file() && checksum(destination) == expected_sha256 {
        return;
    }
    if destination.exists() {
        fs::remove_file(destination).expect("failed to remove invalid cached download");
    }
    println!("cargo:warning=Downloading {url}");
    let temporary = destination.with_extension(format!("{}.partial", std::process::id()));
    download_file(url, &temporary);
    let actual = checksum(&temporary);
    if actual != expected_sha256 {
        fs::remove_file(&temporary).ok();
        panic!(
            "SHA-256 mismatch for `{url}`: expected {expected_sha256}, got {actual}; the release asset may have changed"
        );
    }
    if let Err(error) = fs::rename(&temporary, destination) {
        if destination.is_file() && checksum(destination) == expected_sha256 {
            fs::remove_file(temporary).ok();
        } else {
            panic!("failed to install verified download: {error}");
        }
    }
}

fn download_file(url: &str, destination: &Path) {
    let curl = Command::new("curl")
        .args([
            "--http1.1",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "--retry",
            "2",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--output",
        ])
        .arg(destination)
        .arg(url)
        .status();
    if curl.is_ok_and(|status| status.success()) {
        return;
    }

    if cfg!(windows) {
        let command = format!(
            "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
            url.replace('\'', "''"),
            destination.display().to_string().replace('\'', "''")
        );
        let powershell = Command::new("powershell")
            .args(["-NoProfile", "-Command", &command])
            .status();
        if powershell.is_ok_and(|status| status.success()) {
            return;
        }
    }

    panic!("failed to download `{url}`; install curl or place it in MNN_PREBUILT_CACHE_DIR");
}

fn checksum(path: &Path) -> String {
    let mut file = BufReader::new(File::open(path).expect("failed to open download for hashing"));
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).expect("failed to hash download");
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    format!("{:x}", digest.finalize())
}

fn extract_tar_gz(archive: &Path, destination: &Path) {
    let file = File::open(archive).expect("failed to open tar.gz archive");
    let decoder = GzDecoder::new(BufReader::new(file));
    tar::Archive::new(decoder)
        .unpack(destination)
        .expect("failed to extract tar.gz archive");
}

fn extract_zip(archive: &Path, destination: &Path) {
    let file = File::open(archive).expect("failed to open zip archive");
    let mut archive = zip::ZipArchive::new(BufReader::new(file)).expect("invalid zip archive");
    archive
        .extract(destination)
        .expect("failed to extract zip archive");
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("Cargo did not set {name}"))
}

fn validate_directory(path: &Path, variable: &str) {
    assert!(
        path.is_dir(),
        "{variable} is not a directory: {}",
        path.display()
    );
}

fn target_crt_static() -> bool {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .any(|feature| feature == "crt-static")
}
