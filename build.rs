// ONNX Runtime's prebuilt CoreML objects call `__isPlatformVersionAtLeast`,
// which lives in clang's runtime library; rustc does not link it by default.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = std::process::Command::new("xcrun")
        .args(["clang", "--print-resource-dir"])
        .output();
    if let Ok(o) = out {
        let dir = String::from_utf8_lossy(&o.stdout).trim().to_string();
        println!("cargo:rustc-link-search=native={dir}/lib/darwin");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
