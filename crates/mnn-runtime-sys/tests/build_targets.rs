#[path = "../build_support.rs"]
mod build_support;
#[path = "../prebuilt.rs"]
mod prebuilt;
use build_support::{ios_sdk, target_suffix};

#[test]
fn selects_compatible_abis_only() {
    assert_eq!(
        target_suffix("linux", "x86_64", "musl", "x86_64-unknown-linux-musl"),
        None
    );
    assert_eq!(
        target_suffix("linux", "x86_64", "gnu", "x86_64-unknown-linux-gnu"),
        Some("linux-x86_64")
    );
    assert_eq!(
        target_suffix("ios", "aarch64", "macabi", "aarch64-apple-ios-macabi"),
        None
    );
    assert_eq!(
        target_suffix("ios", "aarch64", "", "aarch64-apple-ios-sim"),
        Some("ios-arm64-sim")
    );
    assert_eq!(
        target_suffix("ios", "aarch64", "", "aarch64-apple-ios"),
        Some("ios-arm64")
    );
}

#[test]
fn intel_ios_targets_are_simulators() {
    assert_eq!(ios_sdk("x86_64-apple-ios"), Some("iphonesimulator"));
    assert_eq!(ios_sdk("aarch64-apple-ios-sim"), Some("iphonesimulator"));
    assert_eq!(ios_sdk("aarch64-apple-ios"), Some("iphoneos"));
    assert_eq!(ios_sdk("x86_64-apple-ios-macabi"), None);
}

#[test]
fn cuda_and_mingw_link_requirements_are_explicit() {
    use build_support::{cuda_side_library, cuda_target_supported, static_cpp_libraries};
    assert!(cuda_target_supported("linux", "gnu"));
    assert!(cuda_target_supported("windows", "msvc"));
    assert!(!cuda_target_supported("macos", ""));
    assert!(!cuda_target_supported("windows", "gnu"));
    assert_eq!(cuda_side_library("linux", true), Some("MNN_Cuda_Main"));
    assert_eq!(cuda_side_library("windows", true), None);
    assert_eq!(cuda_side_library("linux", false), None);
    assert_eq!(
        static_cpp_libraries("windows", "gnu", true),
        &["stdc++", "gcc_eh", "gcc", "winpthread"]
    );
    assert!(static_cpp_libraries("windows", "msvc", true).is_empty());
    assert!(static_cpp_libraries("linux", "gnu", true).is_empty());
}

#[test]
fn every_published_profile_has_an_exact_feature_selection() {
    use prebuilt::{select_prebuilt, PrebuiltFeatures, ARTIFACTS};
    assert_eq!(ARTIFACTS.len(), 22);
    assert_eq!(
        prebuilt::PREBUILT_REPOSITORY,
        "zibo-chen/mnn-native-prebuilds"
    );
    assert!(prebuilt::PREBUILT_TAG.starts_with(&format!("mnn-{}-", prebuilt::MNN_SOURCE_VERSION)));
    assert_eq!(prebuilt::MNN_SOURCE_REVISION.len(), 40);
    assert_eq!(prebuilt::MNN_SOURCE_SHA256.len(), 64);
    for artifact in ARTIFACTS {
        let (os, arch, env, target) = match artifact.base {
            "linux-x86_64" => ("linux", "x86_64", "gnu", "x86_64-unknown-linux-gnu"),
            "linux-aarch64" => ("linux", "aarch64", "gnu", "aarch64-unknown-linux-gnu"),
            "windows-x86_64" => ("windows", "x86_64", "msvc", "x86_64-pc-windows-msvc"),
            "windows-i686" => ("windows", "x86", "msvc", "i686-pc-windows-msvc"),
            "windows-aarch64" => ("windows", "aarch64", "msvc", "aarch64-pc-windows-msvc"),
            "macos-universal" => ("macos", "aarch64", "", "aarch64-apple-darwin"),
            "ios-arm64" => ("ios", "aarch64", "", "aarch64-apple-ios"),
            "ios-arm64-sim" => ("ios", "aarch64", "", "aarch64-apple-ios-sim"),
            "android-arm64-v8a" => ("android", "aarch64", "", "aarch64-linux-android"),
            "android-armeabi-v7a" => ("android", "arm", "", "armv7-linux-androideabi"),
            other => panic!("missing target coverage: {other}"),
        };
        let features = PrebuiltFeatures {
            metal: artifact.has_backend("metal"),
            coreml: artifact.has_backend("coreml"),
            opencl: artifact.has_backend("opencl"),
            opengl: artifact.has_backend("opengl"),
            vulkan: artifact.has_backend("vulkan"),
            cuda: artifact.has_backend("cuda"),
            threadpool: os != "ios",
            crt_static: true,
            ..PrebuiltFeatures::default()
        };
        let selected = select_prebuilt(os, arch, env, target, features).unwrap();
        assert_eq!(selected.suffix, artifact.suffix);
        assert_eq!(artifact.sha256.len(), 64);
        assert!(artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
}

#[test]
fn backend_subsets_choose_complete_profiles_and_mixed_profiles_fall_back() {
    use prebuilt::{select_prebuilt, PrebuiltFeatures};
    let mut features = PrebuiltFeatures {
        threadpool: true,
        ..PrebuiltFeatures::default()
    };
    let linux = |features| {
        select_prebuilt(
            "linux",
            "x86_64",
            "gnu",
            "x86_64-unknown-linux-gnu",
            features,
        )
    };
    assert_eq!(linux(features).unwrap().suffix, "linux-x86_64");
    features.vulkan = true;
    assert_eq!(
        linux(features).unwrap().suffix,
        "linux-x86_64-vulkan-opencl"
    );
    features.opencl = true;
    assert_eq!(
        linux(features).unwrap().suffix,
        "linux-x86_64-vulkan-opencl"
    );
    features.cuda = true;
    assert!(linux(features).is_none());
    features.vulkan = false;
    features.opencl = false;
    assert_eq!(linux(features).unwrap().suffix, "linux-x86_64-cuda12");
    features.cuda = false;
    features.opengl = true;
    assert!(linux(features).is_none());
    let android =
        select_prebuilt("android", "aarch64", "", "aarch64-linux-android", features).unwrap();
    assert!(android.has_backend("opengl"));
    assert!(android.has_backend("vulkan"));
    assert!(android.has_backend("opencl"));
}

#[test]
fn incompatible_threading_crt_abi_and_linkage_never_use_a_prebuilt() {
    use prebuilt::{select_prebuilt, PrebuiltFeatures};
    let features = PrebuiltFeatures {
        threadpool: true,
        ..PrebuiltFeatures::default()
    };
    assert!(select_prebuilt(
        "linux",
        "x86_64",
        "musl",
        "x86_64-unknown-linux-musl",
        features
    )
    .is_none());
    assert!(select_prebuilt(
        "windows",
        "x86_64",
        "msvc",
        "x86_64-pc-windows-msvc",
        features
    )
    .is_none());
    assert!(select_prebuilt(
        "windows",
        "x86_64",
        "gnu",
        "x86_64-pc-windows-gnu",
        features
    )
    .is_none());
    assert!(select_prebuilt(
        "linux",
        "x86_64",
        "gnu",
        "x86_64-unknown-linux-gnu",
        PrebuiltFeatures {
            openmp: true,
            ..features
        }
    )
    .is_none());
    assert!(select_prebuilt(
        "linux",
        "x86_64",
        "gnu",
        "x86_64-unknown-linux-gnu",
        PrebuiltFeatures::default()
    )
    .is_none());
    let ios = |features| select_prebuilt("ios", "aarch64", "", "aarch64-apple-ios", features);
    assert!(ios(features).is_none());
    assert!(ios(PrebuiltFeatures::default()).is_some());
    assert!(ios(PrebuiltFeatures {
        dynamic: true,
        ..PrebuiltFeatures::default()
    })
    .is_none());
    assert!(select_prebuilt(
        "ios",
        "aarch64",
        "macabi",
        "aarch64-apple-ios-macabi",
        PrebuiltFeatures::default()
    )
    .is_none());
}
