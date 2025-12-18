# extxyz-ng

- `extxyz` format reader and writer **in Rust**
- full compatibility with the C implementation ([`libAtom/extxyz`](github.com/libAtoms/extxyz)) via its Rust wrapper ([`extxyz-sys`](https://crates.io/crates/extxyz-sys))
- guarantees memory safety with **no segmentation faults**
- provides clear and user-friendly error messages during parsing

## dev

Clone `libAtoms/extxyz` source code (and its submodule `libcleri` for language parsing) as submodules

```console
git clone --recurse-submodules https://github.com/ccmat-lab/extxyz-rs.git
cd extxyz-rs
```
