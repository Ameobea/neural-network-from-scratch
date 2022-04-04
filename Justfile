opt:
  wasm-opt src/wasm_interface_bg.wasm -g -O4 --enable-simd --enable-nontrapping-float-to-int --precompute-propagate --fast-math --detect-features --strip-dwarf -c -o src/wasm_interface_bg.wasm
  wasm-opt src/wasm_interface_no_simd/wasm_interface_bg.wasm -g -O4 --enable-nontrapping-float-to-int --precompute-propagate --fast-math --detect-features --strip-dwarf -c -o src/wasm_interface_no_simd/wasm_interface_bg.wasm

build-engine:
  cd engine/wasm_interface \
    && RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
  wasm-bindgen ./engine/target/wasm32-unknown-unknown/release/wasm_interface.wasm --browser --remove-producers-section --out-dir ./src

build-engine-no-simd:
  cd engine/wasm_interface \
    && RUSTFLAGS="" cargo build --release --no-default-features --target wasm32-unknown-unknown
  mkdir -p /tmp/wasm_interface_no_simd && rm -rf /tmp/wasm_interface_no_simd/*
  wasm-bindgen ./engine/target/wasm32-unknown-unknown/release/wasm_interface.wasm --browser --remove-producers-section --out-dir /tmp/wasm_interface_no_simd
  cp /tmp/wasm_interface_no_simd/* ./src/wasm_interface_no_simd/

debug-engine:
  cd engine/wasm_interface \
    && RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
  wasm-bindgen ./engine/target/wasm32-unknown-unknown/debug/wasm_interface.wasm --browser --remove-producers-section --out-dir ./src

run:
  just build-engine

  yarn start

build:
  just build-engine
  just build-engine-no-simd
  just opt

  rm -rf dist/*
  yarn build
  cp -r public/* ./dist

deploy:
  phost update nn-viz patch dist
