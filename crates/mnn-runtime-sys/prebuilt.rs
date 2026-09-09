//! Pinned native release and feature-aware archive selection.
// Checksums and backend sets come from the release's SHA-256-verified index.json.

use super::build_support::target_suffix;

pub const PREBUILT_REPOSITORY: &str = "zibo-chen/mnn-native-prebuilds";
pub const PREBUILT_TAG: &str = "mnn-3.6.1-d407447e-r1";
pub const MNN_SOURCE_VERSION: &str = "3.6.1";
pub const MNN_SOURCE_REVISION: &str = "d407447ed56c4121a11ccbd266dc184ca1ead0c2";
pub const MNN_SOURCE_SHA256: &str =
    "13dca9547df7dac40ab40c7318136406f4a33dfe99cd40bfaa4dbb2270cb8795";

#[derive(Clone, Copy, Debug)]
pub struct PrebuiltArtifact {
    pub base: &'static str,
    pub suffix: &'static str,
    pub sha256: &'static str,
    pub backends: &'static [&'static str],
}

impl PrebuiltArtifact {
    pub fn has_backend(self, backend: &str) -> bool {
        self.backends.contains(&backend)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrebuiltFeatures {
    pub metal: bool,
    pub coreml: bool,
    pub opencl: bool,
    pub opengl: bool,
    pub vulkan: bool,
    pub cuda: bool,
    pub threadpool: bool,
    pub openmp: bool,
    pub crt_static: bool,
    pub dynamic: bool,
}

/// Select the smallest profile containing every requested backend.
/// Never substitute a different ABI, CRT, threading policy or linkage.
pub fn select_prebuilt(
    os: &str,
    arch: &str,
    target_env: &str,
    target: &str,
    features: PrebuiltFeatures,
) -> Option<&'static PrebuiltArtifact> {
    let base = target_suffix(os, arch, target_env, target)?;
    if features.openmp
        || features.threadpool != (os != "ios")
        || (os == "windows" && !features.crt_static)
        || (os == "ios" && features.dynamic)
    {
        return None;
    }
    let requested = [
        (features.metal, "metal"),
        (features.coreml, "coreml"),
        (features.opencl, "opencl"),
        (features.opengl, "opengl"),
        (features.vulkan, "vulkan"),
        (features.cuda, "cuda"),
    ];
    ARTIFACTS
        .iter()
        .filter(|artifact| {
            artifact.base == base
                && requested
                    .iter()
                    .all(|(enabled, backend)| !enabled || artifact.has_backend(backend))
        })
        .min_by_key(|artifact| artifact.backends.len())
}

pub const ARTIFACTS: &[PrebuiltArtifact] = &[
    PrebuiltArtifact {
        base: "android-arm64-v8a",
        suffix: "android-arm64-v8a-vulkan-opencl-opengl",
        sha256: "250a10cfadbac5b3a9c6e327b39fccb5a1781d9d154b3e437ce294e32b7493a5",
        backends: &["cpu", "vulkan", "opencl", "opengl"],
    },
    PrebuiltArtifact {
        base: "android-arm64-v8a",
        suffix: "android-arm64-v8a",
        sha256: "82988d6ad8feae7868c43c139bab4e968920c57505b963b47a7f96d6ffa50dcc",
        backends: &["cpu"],
    },
    PrebuiltArtifact {
        base: "android-armeabi-v7a",
        suffix: "android-armeabi-v7a-vulkan-opencl-opengl",
        sha256: "590ec0bbd1ddd82a2ace8aa922f8591e4262757f61637f710a7c1fd867ebe336",
        backends: &["cpu", "vulkan", "opencl", "opengl"],
    },
    PrebuiltArtifact {
        base: "android-armeabi-v7a",
        suffix: "android-armeabi-v7a",
        sha256: "f83486ab31cb45c9d47945019aacbc9dd9e55709c0ad3d0447216ee79577f030",
        backends: &["cpu"],
    },
    PrebuiltArtifact {
        base: "ios-arm64",
        suffix: "ios-arm64-metal-coreml",
        sha256: "694fd894217eaddd5db0ed58746e25875c0c48f92083ac786ebc3af0812130d3",
        backends: &["cpu", "metal", "coreml"],
    },
    PrebuiltArtifact {
        base: "ios-arm64-sim",
        suffix: "ios-arm64-sim-metal-coreml",
        sha256: "e2b4a03a8b81655d7690873e860a54608d0575e0592034366216853d2008b94b",
        backends: &["cpu", "metal", "coreml"],
    },
    PrebuiltArtifact {
        base: "ios-arm64-sim",
        suffix: "ios-arm64-sim",
        sha256: "c439bed33ffa6f85b748d81f6593142dff10a2a44b10899c57a4123a6d2c0f13",
        backends: &["cpu", "metal"],
    },
    PrebuiltArtifact {
        base: "ios-arm64",
        suffix: "ios-arm64",
        sha256: "7c4704add39db7f20943fbe83286dfab39e91ebb6bbeeab4175c24324c6bd755",
        backends: &["cpu", "metal"],
    },
    PrebuiltArtifact {
        base: "linux-aarch64",
        suffix: "linux-aarch64-vulkan-opencl",
        sha256: "d3e66242aaa822e4c35e86a6879c9b1dcedecdad67326b4f44c3adc8dae62cec",
        backends: &["cpu", "vulkan", "opencl"],
    },
    PrebuiltArtifact {
        base: "linux-aarch64",
        suffix: "linux-aarch64",
        sha256: "069967b5d83d90339c6a553ed2820051e53f5a7d45b6c741221e2e988f0f474d",
        backends: &["cpu"],
    },
    PrebuiltArtifact {
        base: "linux-x86_64",
        suffix: "linux-x86_64-cuda12",
        sha256: "5ad4ce3db6615abdb85ec9108b8ea4141dc5a978bfa55fd3dde6dac352b49810",
        backends: &["cpu", "cuda"],
    },
    PrebuiltArtifact {
        base: "linux-x86_64",
        suffix: "linux-x86_64-vulkan-opencl",
        sha256: "34e30209aa7d9988ee6ee6a27050294622a46f0d5cf20759c83c77891e21edeb",
        backends: &["cpu", "vulkan", "opencl"],
    },
    PrebuiltArtifact {
        base: "linux-x86_64",
        suffix: "linux-x86_64",
        sha256: "f1cf3e509deb82769db038f55af7eb63aeab1c5ffd8d4ba80efbf9af38b17474",
        backends: &["cpu"],
    },
    PrebuiltArtifact {
        base: "macos-universal",
        suffix: "macos-universal-metal-coreml",
        sha256: "db5a1d0756e5500951a7a71b4d5a535e87ba3281e9c794fac2e2a54786234213",
        backends: &["cpu", "metal", "coreml"],
    },
    PrebuiltArtifact {
        base: "macos-universal",
        suffix: "macos-universal",
        sha256: "1bd39996fe07940efb36918b78f349f8e317899a41071f4dcaffb7ce0e124bf3",
        backends: &["cpu", "metal"],
    },
    PrebuiltArtifact {
        base: "windows-aarch64",
        suffix: "windows-aarch64-vulkan-opencl",
        sha256: "c166f3ba6483135a7686e596edb9068fb4680f376c78d78b626f6b1c5c2d40ea",
        backends: &["cpu", "vulkan", "opencl"],
    },
    PrebuiltArtifact {
        base: "windows-aarch64",
        suffix: "windows-aarch64",
        sha256: "88ce9d5f3cedefd2d08d6c8b0b939fce37587d0231b55977265195c969709edc",
        backends: &["cpu"],
    },
    PrebuiltArtifact {
        base: "windows-i686",
        suffix: "windows-i686-vulkan-opencl",
        sha256: "6f2ba1ba1a7f66504c98eb1e9b343d6643fbe2732f2482257524170e1937b456",
        backends: &["cpu", "vulkan", "opencl"],
    },
    PrebuiltArtifact {
        base: "windows-i686",
        suffix: "windows-i686",
        sha256: "332c673473a29fa09364b4ccbbc2daf792399744a8872937918e4ecfe98b45d9",
        backends: &["cpu"],
    },
    PrebuiltArtifact {
        base: "windows-x86_64",
        suffix: "windows-x86_64-cuda12",
        sha256: "8dc6bdb21a2062b7df4776f0b961a278b5e764bac44cbe102d0ba32beb96500d",
        backends: &["cpu", "cuda"],
    },
    PrebuiltArtifact {
        base: "windows-x86_64",
        suffix: "windows-x86_64-vulkan-opencl",
        sha256: "afd92573300cf6f4c33f69518e2aaddff752d686c778e97baccb9f61ef131c35",
        backends: &["cpu", "vulkan", "opencl"],
    },
    PrebuiltArtifact {
        base: "windows-x86_64",
        suffix: "windows-x86_64",
        sha256: "13f70a97a99108b206f99b08f185c6b0c189f145256889f5c54b61fccc79f949",
        backends: &["cpu"],
    },
];
