# extxyz

Unsafe automatically-generated Rust bindings for libAtoms/extxyz parser

## vendor dependencies

We vendor dependency `libcleri` as parser on a sepecific version of it.
We don't vendor `pcre2` which is the dependency of `libcleri`, because it is a dependency of many major tools such as `git` or `Safari` or `Excel` so it is already in the most OS.
