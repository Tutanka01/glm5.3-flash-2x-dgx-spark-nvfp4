FROM ghcr.io/0xsero/glm-5.3-flash-sglang-sm121@sha256:f9ac60ba4071f8acd64f0f3c074aca308f6d659405fee46fc8031489a1e8b19b

ARG SGLANG_DFLASH_REVISION=2d4b6acea720f5eb5a2b2c269f1cc9dcbc8b10c0
ARG SM121_PATCH_REVISION=dfb1bb7e45c20058d37df7a39cceda45a9d216a8

LABEL org.opencontainers.image.title="GLM-5.3-Flash DFlash2 SM121 experiment" \
      org.opencontainers.image.source="https://github.com/sgl-project/sglang/pull/36708" \
      org.opencontainers.image.revision="2d4b6acea720f5eb5a2b2c269f1cc9dcbc8b10c0" \
      org.opencontainers.image.base.digest="sha256:f9ac60ba4071f8acd64f0f3c074aca308f6d659405fee46fc8031489a1e8b19b"

# Bring in the exact official DFlash2/GLM branch without allowing pip to alter
# the accepted CUDA, FlashInfer, NCCL or CUTLASS dependency pins.
RUN git init /tmp/sglang-dflash \
 && git -C /tmp/sglang-dflash remote add origin https://github.com/sgl-project/sglang.git \
 && git -C /tmp/sglang-dflash fetch --depth 1 origin "${SGLANG_DFLASH_REVISION}" \
 && git -C /tmp/sglang-dflash checkout --detach FETCH_HEAD \
 && rm -rf /sgl-workspace/sglang/python/sglang \
 && cp -a /tmp/sglang-dflash/python/sglang /sgl-workspace/sglang/python/sglang

# Reapply the six exact SM121/NVFP4 files from the accepted 0xSero image source.
# The GLM file is patched once more below to restore the DFlash mHC adapter.
RUN git init /tmp/sm121-patches \
 && git -C /tmp/sm121-patches remote add origin https://github.com/0xSero/glm-5.3-flash-sglang-sm121.git \
 && git -C /tmp/sm121-patches fetch --depth 1 origin "${SM121_PATCH_REVISION}" \
 && git -C /tmp/sm121-patches checkout --detach FETCH_HEAD \
 && cp /tmp/sm121-patches/patches/sglang-glm5_next-debug.py /sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py \
 && cp /tmp/sm121-patches/patches/sglang-deepseek_nextn-glm53.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py \
 && cp /tmp/sm121-patches/patches/sglang-quant-utils-sm120.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/utils.py \
 && cp /tmp/sm121-patches/patches/sglang-modelopt-quant-sm120.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py \
 && cp /tmp/sm121-patches/patches/sglang-flash_mla_sm120-glm53.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py \
 && cp /tmp/sm121-patches/patches/sglang-dsa_backend-glm53.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py

COPY patch_glm_dflash.py /tmp/patch_glm_dflash.py
RUN python3 /tmp/patch_glm_dflash.py \
 && python3 -m compileall -q /sgl-workspace/sglang/python/sglang \
 && rm -rf /tmp/sglang-dflash /tmp/sm121-patches /tmp/patch_glm_dflash.py
