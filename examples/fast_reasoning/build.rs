fn main() {
    println!("cargo:rerun-if-changed=src/ops/metal/mps_ndarray_bench.mm");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/ops/metal/mps_ndarray_bench.mm")
            .flag("-fobjc-arc")
            .flag("-std=c++17")
            .compile("mps_ndarray_bench");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }
}
