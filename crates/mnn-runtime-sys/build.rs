use std::{
    env,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
};

mod build_support;
use build_support::{ios_sdk, target_suffix};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

const PREBUILT_REPOSITORY: &str = "zibo-chen/MNN-Prebuilds";
const PREBUILT_TAG: &str = "dev";
const MNN_SOURCE_VERSION: &str = "3.6.0";
const MNN_SOURCE_SHA256: &str = "4ddbe825a22ee06e8c237bf3382231d5b6130878c850291d015808946cb87690";

#[derive(Clone, Copy, PartialEq, Eq)]
enum LinkMode {
    Static,
    Dynamic,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=cpp/mnn_runtime_bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/mnn_runtime_bridge.h");
    for variable in [
        "MNN_INCLUDE_DIR",
        "MNN_LIB_DIR",
        "MNN_PREBUILT_CACHE_DIR",
        "MNN_PREBUILT_BASE_URL",
        "MNN_PREBUILT_TAG",
        "MNN_PREBUILT_SHA256",
        "MNN_SOURCE_DIR",
        "MNN_OPENMP_LIB",
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
    let has_prebuilt = prebuilt_compatible(&target_os)
        && target_suffix(&target_os, &target_arch, &target_env, &target).is_some();
    let (include_dir, lib_dir, prebuilt) = if let Some(external) = external_installation() {
        (external.0, external.1, false)
    } else if cfg!(feature = "build-from-source") || !has_prebuilt {
        let installation = build_from_source(&target_os, &target_arch, &target_env, &target);
        (installation.0, installation.1, false)
    } else if cfg!(feature = "prebuilt") {
        let installation = acquire_prebuilt(&target_os, &target_arch, &target_env, &target);
        (installation.0, installation.1, true)
    } else {
        let installation = build_from_source(&target_os, &target_arch, &target_env, &target);
        (installation.0, installation.1, false)
    };

    build_bridge(&include_dir, &target_env, prebuilt, link_mode);
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

fn prebuilt_compatible(target_os: &str) -> bool {
    let metal_ok = !cfg!(feature = "metal") || matches!(target_os, "macos" | "ios");
    let unsupported_backend = cfg!(feature = "coreml")
        || cfg!(feature = "opencl")
        || cfg!(feature = "opengl")
        || cfg!(feature = "vulkan")
        || cfg!(feature = "openmp");
    metal_ok
        && !unsupported_backend
        && cfg!(feature = "mnn-threadpool")
        && (target_os != "windows" || target_crt_static())
}

fn acquire_prebuilt(os: &str, arch: &str, target_env: &str, target: &str) -> (PathBuf, PathBuf) {
    let suffix = target_suffix(os, arch, target_env, target).unwrap_or_else(|| {
        panic!(
            "MNN-Prebuilds has no artifact for target `{target}`; enable `build-from-source` or provide MNN_INCLUDE_DIR and MNN_LIB_DIR"
        )
    });
    let tag = env::var("MNN_PREBUILT_TAG").unwrap_or_else(|_| PREBUILT_TAG.to_owned());
    let asset = format!("mnn-{tag}-{suffix}");
    let extension = if os == "windows" { "zip" } else { "tar.gz" };
    let checksum = env::var("MNN_PREBUILT_SHA256")
        .ok()
        .or_else(|| (tag == PREBUILT_TAG).then(|| prebuilt_checksum(suffix).to_owned()))
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

fn prebuilt_checksum(suffix: &str) -> &'static str {
    match suffix {
        "android-arm64-v8a" => "5d12bb49d7a020c8fdb9f01a40f208fdc08da51d0aadb016d4609aee7f63572c",
        "android-armeabi-v7a" => "35af840ad4d2aa2bf2078c2cc904629ea4556616c2729c8e91ce8ccb15a072ec",
        "ios-arm64-sim" => "db140b3cca7d03348fd230fff73f9ec38d6421d5a6bad86d5ad8af423d924206",
        "ios-arm64" => "46b5352801ee341e1fecd02622ea43b5633c2128e46dcec84e1761a5f458ad3e",
        "linux-aarch64" => "1ce0b2ed372fbb1db49273d8b835ae5338a0696002f3a6632ec8a14ff52bd50e",
        "linux-x86_64" => "0692b88f2a4caa4c1a3793bf93c84317e1f999e515102c91ddc278aa18b2a4df",
        "macos-universal" => "61e0f340b062cae44d0995610c90ad46b9609839f02854b61f4164ea91698bbd",
        "windows-aarch64" => "f46a233cc4ccbfb02d2edd3864891c7589ad9ed9fbaacd8c534d3d02cc183912",
        "windows-i686" => "9444706efa25add47732e6b88fe46ecf9921ce6dd28f8e8274932fcf9a42c38",
        "windows-x86_64" => "24166165d7451423aef3ebe6694651bad117bcdfc32261652b8f8275961ab91a",
        _ => panic!("no checksum is recorded for MNN prebuilt suffix `{suffix}`"),
    }
}

fn build_from_source(os: &str, arch: &str, target_env: &str, target: &str) -> (PathBuf, PathBuf) {
    let source = acquire_source();
    let mut config = cmake::Config::new(&source);
    config
        .profile("Release")
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
        .join(format!("MNN-{MNN_SOURCE_VERSION}"));
    if root.join("CMakeLists.txt").is_file() {
        return root;
    }
    let archive = out_dir.join(format!("MNN-{MNN_SOURCE_VERSION}.tar.gz"));
    let url =
        format!("https://codeload.github.com/alibaba/MNN/tar.gz/refs/tags/{MNN_SOURCE_VERSION}");
    ensure_download(&url, &archive, MNN_SOURCE_SHA256);
    let destination = out_dir.join("source");
    fs::create_dir_all(&destination).expect("failed to create MNN source directory");
    extract_tar_gz(&archive, &destination);
    root
}

fn build_bridge(include: &Path, target_env: &str, _prebuilt: bool, _link_mode: LinkMode) {
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
    } else {
        build
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-fvisibility=hidden");
    }
    build.compile("mnn_runtime_bridge");
}

fn link_mnn(lib: &Path, os: &str, target_env: &str, mode: LinkMode, prebuilt: bool) {
    println!("cargo:rustc-link-search=native={}", lib.display());
    match (os, mode, prebuilt) {
        ("windows", LinkMode::Static, true) => println!("cargo:rustc-link-lib=static=MNN_static"),
        (_, LinkMode::Static, _) => println!("cargo:rustc-link-lib=static=MNN"),
        (_, LinkMode::Dynamic, _) => println!("cargo:rustc-link-lib=dylib=MNN"),
    }

    match (os, target_env) {
        ("macos" | "ios", _) => println!("cargo:rustc-link-lib=dylib=c++"),
        ("linux", _) => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        }
        ("android", _) => {
            println!("cargo:rustc-link-lib=static=c++_static");
            println!("cargo:rustc-link-lib=log");
        }
        _ => {}
    }
    if matches!(os, "macos" | "ios") && (prebuilt || cfg!(feature = "metal")) {
        for framework in [
            "Foundation",
            "CoreFoundation",
            "Metal",
            "MetalPerformanceShaders",
        ] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=objc");
    }
    if cfg!(feature = "coreml") && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
    }
    if cfg!(feature = "opencl") {
        if os == "macos" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }
    if cfg!(feature = "opengl") {
        match os {
            "android" => {
                println!("cargo:rustc-link-lib=GLESv3");
                println!("cargo:rustc-link-lib=EGL");
            }
            "linux" => println!("cargo:rustc-link-lib=GL"),
            _ => {}
        }
    }
    if cfg!(feature = "vulkan") {
        println!("cargo:rustc-link-lib=vulkan");
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
