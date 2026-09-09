#[path = "../build_support.rs"]
mod build_support;
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
