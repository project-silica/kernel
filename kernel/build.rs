//! Passes the architecture's linker script to the kernel link.
fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let script = format!("{dir}/arch/{arch}/link.ld");
    println!("cargo:rerun-if-changed={script}");
    if std::path::Path::new(&script).exists() {
        println!("cargo:rustc-link-arg-bin=kernel=-T{script}");
    }
}
