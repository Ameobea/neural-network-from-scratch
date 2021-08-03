opt:
  wasm-opt src/wasm_interface_bg.wasm -g -O4 --enable-simd --enable-nontrapping-float-to-int --precompute-propagate --fast-math --detect-features --strip-dwarf -c -o src/wasm_interface_bg.wasm

build-engine:
  cd engine/wasm_interface \
    && RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
  wasm-bindgen ./engine/target/wasm32-unknown-unknown/release/wasm_interface.wasm --browser --remove-producers-section --out-dir ./src

run:
  just build-engine

  yarn start

build:
  just build-engine
  just opt

  rm -rf dist/*
  yarn build

deploy:
  phost update nn-viz patch dist
