use std::{
    collections::hash_map::DefaultHasher,
    ffi::OsString,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

/// Keep native source/build paths independent of deeply nested Cargo targets.
/// The Cargo output directory and source revision isolate concurrent builds.
pub fn source_build_dir(
    out_dir: &Path,
    base: Option<&Path>,
    temporary: &Path,
    windows_host: bool,
    revision: &str,
) -> PathBuf {
    let base = match base {
        Some(base) => base.to_owned(),
        None if windows_host => temporary.join("mnn"),
        None => return out_dir.to_owned(),
    };
    let mut hash = DefaultHasher::new();
    out_dir.hash(&mut hash);
    revision.hash(&mut hash);
    base.join(format!("{:016x}", hash.finish()))
}

/// FindCUDA embeds toolkit paths in generated CMake scripts, where backslashes
/// can become escape sequences. Preserve native paths for filesystem access.
pub fn cmake_path(path: &Path, windows_host: bool) -> OsString {
    if !windows_host {
        return path.as_os_str().to_owned();
    }
    let path = path.to_string_lossy().replace('\\', "/");
    if let Some(unc) = path.strip_prefix("//?/UNC/") {
        format!("//{unc}").into()
    } else {
        path.strip_prefix("//?/").unwrap_or(&path).into()
    }
}

pub fn target_suffix(os: &str, arch: &str, target_env: &str, target: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") if target_env == "gnu" => Some("linux-x86_64"),
        ("linux", "aarch64") if target_env == "gnu" => Some("linux-aarch64"),
        ("windows", "x86_64") if target_env == "msvc" => Some("windows-x86_64"),
        ("windows", "x86") if target_env == "msvc" => Some("windows-i686"),
        ("windows", "aarch64") if target_env == "msvc" => Some("windows-aarch64"),
        ("macos", "aarch64" | "x86_64") => Some("macos-universal"),
        ("ios", _) if target.ends_with("-macabi") => None,
        ("ios", "aarch64") if target.contains("-sim") => Some("ios-arm64-sim"),
        ("ios", "aarch64") => Some("ios-arm64"),
        ("android", "aarch64") => Some("android-arm64-v8a"),
        ("android", "arm") => Some("android-armeabi-v7a"),
        _ => None,
    }
}

/// Select the SDK using Rust target ABI rules, including Intel simulators.
pub fn ios_sdk(target: &str) -> Option<&'static str> {
    if target.ends_with("-macabi") {
        None
    } else if target.ends_with("-sim")
        || target.starts_with("x86_64-")
        || target.starts_with("i386-")
    {
        Some("iphonesimulator")
    } else {
        Some("iphoneos")
    }
}

/// CUDA toolkit supported target combinations for the pinned MNN baseline.
pub fn cuda_target_supported(os: &str, env: &str) -> bool {
    os == "linux" || (os == "windows" && env == "msvc")
}

/// Linux MNN keeps CUDA execution in a shared side library even with static MNN.
pub fn cuda_side_library(os: &str, enabled: bool) -> Option<&'static str> {
    (enabled && os == "linux").then_some("MNN_Cuda_Main")
}

/// MinGW runtime archives required for an executable without MinGW DLLs.
pub fn static_cpp_libraries(os: &str, env: &str, enabled: bool) -> &'static [&'static str] {
    if enabled && os == "windows" && env == "gnu" {
        &["stdc++", "gcc_eh", "gcc", "winpthread"]
    } else {
        &[]
    }
}
