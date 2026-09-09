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
