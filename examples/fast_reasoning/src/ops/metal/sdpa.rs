use candle_core::{DType, Layout, MetalStorage, Shape, Storage, Tensor};

pub fn tiled_sdpa(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> candle_core::Result<Tensor> {
    query.apply_op3_no_bwd(
        key,
        value,
        &TiledSdpa {
            mask: mask.cloned(),
            scale,
        },
    )
}

pub fn supports_tiled_sdpa(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> bool {
    let Ok((batch, query_heads, query_len, head_dim)) = query.dims4() else {
        return false;
    };
    let Ok((key_batch, kv_heads, key_len, key_dim)) = key.dims4() else {
        return false;
    };
    let Ok((value_batch, value_heads, value_len, value_dim)) = value.dims4() else {
        return false;
    };
    let valid_mask = mask.is_none_or(|mask| {
        mask.device().is_metal()
            && mask.dtype() == DType::F32
            && mask.layout().is_contiguous()
            && mask.layout().start_offset() == 0
            && mask.dims4().is_ok_and(|(mb, mh, mq, mk)| {
                (mb == 1 || mb == batch)
                    && (mh == 1 || mh == query_heads)
                    && mq == query_len
                    && mk == key_len
            })
    });
    query.device().is_metal()
        && key.device().is_metal()
        && value.device().is_metal()
        && query.dtype() == DType::F32
        && query_len > 32
        && [32, 64, 72, 80, 96, 128, 256].contains(&head_dim)
        && batch == key_batch
        && batch == value_batch
        && query_heads % kv_heads == 0
        && kv_heads == value_heads
        && query_len == key_len
        && key_len == value_len
        && key_dim == head_dim
        && value_dim == head_dim
        && query.layout().is_contiguous()
        && key.layout().is_contiguous()
        && value.layout().is_contiguous()
        && valid_mask
}

#[derive(Clone, Debug)]
struct TiledSdpa {
    mask: Option<Tensor>,
    scale: f32,
}

impl candle_core::CustomOp3 for TiledSdpa {
    fn name(&self) -> &'static str {
        "tiled-masked-sdpa"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("tiled Metal SDPA was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        query: &MetalStorage,
        query_layout: &Layout,
        key: &MetalStorage,
        key_layout: &Layout,
        value: &MetalStorage,
        value_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = query.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("tiled-masked-sdpa");
        let output_elements = query_layout.shape().elem_count();
        let output = device.new_buffer(output_elements, DType::F32, "tiled-masked-sdpa-output")?;
        let output_strides = [
            query_layout.dims()[1] * query_layout.dims()[2] * query_layout.dims()[3],
            query_layout.dims()[2] * query_layout.dims()[3],
            query_layout.dims()[3],
            1,
        ];

        if let Some(mask) = &self.mask {
            let (mask_storage, mask_layout) = mask.storage_and_layout();
            let Storage::Metal(mask_storage) = &*mask_storage else {
                candle_core::bail!("tiled Metal SDPA mask is not on Metal")
            };
            let [_, _, mask_query_stride, _] = mask_layout.stride() else {
                candle_core::bail!("tiled Metal SDPA mask must have rank four")
            };
            candle_metal_kernels::call_sdpa_full(
                device.metal_device(),
                &encoder,
                device.kernels(),
                query_layout.start_offset() * DType::F32.size_in_bytes(),
                query_layout.dims(),
                query_layout.stride(),
                query.buffer(),
                key_layout.start_offset() * DType::F32.size_in_bytes(),
                key_layout.dims(),
                key_layout.stride(),
                key.buffer(),
                value_layout.start_offset() * DType::F32.size_in_bytes(),
                value.buffer(),
                value_layout.stride(),
                Some(candle_metal_kernels::SdpaDType::F32),
                Some(mask_storage.buffer()),
                Some(&[
                    mask_layout.stride()[0],
                    mask_layout.stride()[1],
                    *mask_query_stride,
                ]),
                &output,
                &output_strides,
                self.scale,
                false,
                candle_metal_kernels::SdpaDType::F32,
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            candle_metal_kernels::call_sdpa_full(
                device.metal_device(),
                &encoder,
                device.kernels(),
                query_layout.start_offset() * DType::F32.size_in_bytes(),
                query_layout.dims(),
                query_layout.stride(),
                query.buffer(),
                key_layout.start_offset() * DType::F32.size_in_bytes(),
                key_layout.dims(),
                key_layout.stride(),
                key.buffer(),
                value_layout.start_offset() * DType::F32.size_in_bytes(),
                value.buffer(),
                value_layout.stride(),
                None,
                None,
                None,
                &output,
                &output_strides,
                self.scale,
                false,
                candle_metal_kernels::SdpaDType::F32,
            )
            .map_err(candle_core::Error::wrap)?;
        }

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            query_layout.shape().clone(),
        ))
    }
}

/// Fused Metal SDPA for causal GQA attention. Single-token decode uses the
/// vector kernel below 1024 KV tokens and the two-pass vector kernel at/above
/// 1024. Multi-token prefill uses the tiled causal SDPA kernel. This operator
/// does not accept an arbitrary attention mask and must not be used for
/// bidirectional or sliding-window attention.
pub fn fused_sdpa(query: &Tensor, key: &Tensor, value: &Tensor) -> candle_core::Result<Tensor> {
    query.apply_op3_no_bwd(key, value, &FusedSdpa)
}

/// Reports whether Candle's bundled Metal SDPA kernels cover these tensors.
/// This is an operator/backend capability check, independent of model config.
pub fn supports_fused_sdpa(query: &Tensor, key: &Tensor, value: &Tensor) -> bool {
    let Ok((batch, query_heads, query_len, head_dim)) = query.dims4() else {
        return false;
    };
    let Ok((key_batch, kv_heads, context, key_dim)) = key.dims4() else {
        return false;
    };
    let Ok((value_batch, value_heads, value_context, value_dim)) = value.dims4() else {
        return false;
    };
    let head_dim_supported = if query_len == 1 {
        [32, 64, 96, 128, 256].contains(&head_dim)
    } else {
        [32, 64, 72, 80, 96, 128, 256].contains(&head_dim)
    };
    let vector_batch_supported = query_len != 1 || batch == 1;
    head_dim_supported
        && vector_batch_supported
        && context > 0
        && key_batch == batch
        && value_batch == batch
        && key_dim == head_dim
        && value_dim == head_dim
        && value_heads == kv_heads
        && value_context == context
        && kv_heads > 0
        && query_heads % kv_heads == 0
        && query.layout().is_contiguous()
        && key.layout().is_contiguous()
        && value.layout().is_contiguous()
        && query.dtype() == key.dtype()
        && query.dtype() == value.dtype()
        && matches!(query.dtype(), DType::F32 | DType::F16 | DType::BF16)
}

#[derive(Clone, Debug)]
struct FusedSdpa;

impl candle_core::CustomOp3 for FusedSdpa {
    fn name(&self) -> &'static str {
        "fused-sdpa"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("fused Metal SDPA was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        query: &MetalStorage,
        query_layout: &Layout,
        key: &MetalStorage,
        key_layout: &Layout,
        value: &MetalStorage,
        value_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let (batch, query_heads, query_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, context, key_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_heads, value_context, value_dim) = value_layout.shape().dims4()?;
        let supported_head_dims = if query_len == 1 {
            &[32, 64, 96, 128, 256][..]
        } else {
            &[32, 64, 72, 80, 96, 128, 256][..]
        };
        if !supported_head_dims.contains(&head_dim) {
            candle_core::bail!(
                "fused decoder SDPA does not support head_dim={head_dim} for query_len={query_len}"
            )
        }
        if (query_len == 1 && batch != 1)
            || key_batch != batch
            || value_batch != batch
            || key_dim != head_dim
            || value_dim != head_dim
            || value_heads != kv_heads
            || value_context != context
            || query_heads % kv_heads != 0
        {
            candle_core::bail!("incompatible Q/K/V shapes for fused GQA decode")
        }
        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle_core::bail!("fused SDPA decode requires contiguous Q/K/V")
        }
        if query.dtype() != key.dtype() || query.dtype() != value.dtype() {
            candle_core::bail!("fused SDPA decode requires matching Q/K/V dtypes")
        }

        let dtype = match query.dtype() {
            DType::F32 => candle_metal_kernels::SdpaDType::F32,
            DType::F16 => candle_metal_kernels::SdpaDType::F16,
            DType::BF16 => candle_metal_kernels::SdpaDType::BF16,
            dtype => candle_core::bail!("fused SDPA decode does not support {dtype:?}"),
        };
        let device = query.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("fused-sdpa");

        let output_elements = batch * query_heads * query_len * head_dim;
        let output = device.new_buffer(output_elements, query.dtype(), "fused-sdpa-output")?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_offset = query_layout.start_offset() * query.dtype().size_in_bytes();
        let k_offset = key_layout.start_offset() * key.dtype().size_in_bytes();
        let v_offset = value_layout.start_offset() * value.dtype().size_in_bytes();

        if query_len == 1 && context >= 1024 {
            let block_count = candle_metal_kernels::SDPA_2PASS_BLOCKS;
            let partial_elements = output_elements * block_count;
            let stats_elements = batch * query_heads * query_len * block_count;
            let intermediate =
                device.new_buffer(partial_elements, query.dtype(), "fused-sdpa-partials")?;
            let sums = device.new_buffer(stats_elements, DType::F32, "fused-sdpa-sums")?;
            let maxs = device.new_buffer(stats_elements, DType::F32, "fused-sdpa-maxs")?;
            candle_metal_kernels::call_sdpa_vector_2pass(
                device.metal_device(),
                &encoder,
                device.kernels(),
                q_offset,
                query_layout.dims(),
                query.buffer(),
                k_offset,
                key_layout.dims(),
                key_layout.stride(),
                key.buffer(),
                v_offset,
                value_layout.stride(),
                value.buffer(),
                &output,
                &intermediate,
                &sums,
                &maxs,
                scale,
                1.0,
                dtype,
            )
            .map_err(candle_core::Error::wrap)?;
        } else if query_len == 1 {
            candle_metal_kernels::call_sdpa_vector(
                device.metal_device(),
                &encoder,
                device.kernels(),
                q_offset,
                query_layout.dims(),
                query.buffer(),
                k_offset,
                key_layout.dims(),
                key_layout.stride(),
                key.buffer(),
                v_offset,
                value_layout.stride(),
                value.buffer(),
                &output,
                scale,
                1.0,
                dtype,
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            candle_metal_kernels::call_sdpa_full(
                device.metal_device(),
                &encoder,
                device.kernels(),
                q_offset,
                query_layout.dims(),
                query_layout.stride(),
                query.buffer(),
                k_offset,
                key_layout.dims(),
                key_layout.stride(),
                key.buffer(),
                v_offset,
                value.buffer(),
                value_layout.stride(),
                None,
                None,
                None,
                &output,
                query_layout.stride(),
                scale,
                true,
                dtype,
            )
            .map_err(candle_core::Error::wrap)?;
        }

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, query.dtype()),
            query_layout.shape().clone(),
        ))
    }
}
