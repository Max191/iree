# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards RISCV using CMake.

FROM gcr.io/iree-oss/base@sha256:24fb5467da30c7b4c0f4c191cdf6124bda63b172d3ae98906e53b3d55ed6ddcb AS install-riscv
WORKDIR /install-riscv
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/toolchain_iree_20220918.tar.gz"
RUN tar -xf "toolchain_iree_20220918.tar.gz" -C /usr/src/
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/toolchain_iree_rv32_20220918.tar.gz"
RUN tar -xf "toolchain_iree_rv32_20220918.tar.gz" -C /usr/src/
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/qemu-riscv.tar.gz"
RUN tar -xf "qemu-riscv.tar.gz" -C /usr/src/

FROM gcr.io/iree-oss/base@sha256:24fb5467da30c7b4c0f4c191cdf6124bda63b172d3ae98906e53b3d55ed6ddcb AS final
COPY --from=install-riscv "/usr/src/toolchain_iree" "/usr/src/toolchain_iree"
COPY --from=install-riscv "/usr/src/toolchain_iree_rv32imf" "/usr/src/toolchain_iree_rv32imf"
COPY --from=install-riscv "/usr/src/qemu-riscv" "/usr/src/qemu-riscv"
ENV RISCV_RV64_LINUX_TOOLCHAIN_ROOT="/usr/src/toolchain_iree"
ENV RISCV_RV32_NEWLIB_TOOLCHAIN_ROOT="/usr/src/toolchain_iree_rv32imf"
ENV QEMU_RV64_BIN="/usr/src/qemu-riscv/qemu-riscv64"
ENV QEMU_RV32_BIN="/usr/src/qemu-riscv/qemu-riscv32"