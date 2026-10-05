use super::VisionError;
use candle_core::Tensor;

pub struct VisionLayerWeights {
    pub norm1_weight: Tensor,
    pub norm1_bias: Tensor,
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
    pub norm2_weight: Tensor,
    pub norm2_bias: Tensor,
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

pub struct VisionTransformerLayer {
    weights: VisionLayerWeights,
    heads: usize,
    eps: f32,
}

impl VisionTransformerLayer {
    pub fn new(weights: VisionLayerWeights, heads: usize, eps: f32) -> Self {
        Self {
            weights,
            heads,
            eps,
        }
    }

    /// Pre-norm ViT layer on `[sequence, hidden]` patch embeddings.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, VisionError> {
        let sequence = input.dims()[0];
        let hidden = input.dims()[1];
        let head_dim = hidden / self.heads;
        if hidden % self.heads != 0 {
            return Err(VisionError::Image(
                "vision hidden size must divide by heads".into(),
            ));
        }
        let norm = layer_norm(
            input,
            &self.weights.norm1_weight,
            &self.weights.norm1_bias,
            self.eps,
        )?;
        let project = |weight: &Tensor, bias: &Tensor| -> Result<Tensor, VisionError> {
            norm.matmul(&weight.t().map_err(compute)?)
                .map_err(compute)?
                .broadcast_add(bias)
                .map_err(compute)?
                .reshape((sequence, self.heads, head_dim))
                .map_err(compute)?
                .transpose(0, 1)
                .map_err(compute)?
                // The transposed view interleaves heads with a batch stride of
                // `head_dim`, which Metal's strided batched GEMM rejects.
                .contiguous()
                .map_err(compute)
        };
        let q = project(&self.weights.q_weight, &self.weights.q_bias)?;
        let k = project(&self.weights.k_weight, &self.weights.k_bias)?;
        let v = project(&self.weights.v_weight, &self.weights.v_bias)?;
        let scores = q
            .matmul(&k.transpose(1, 2).map_err(compute)?)
            .map_err(compute)?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
            .map_err(compute)?;
        let attention = softmax_last_dim(&scores)?
            .matmul(&v)
            .map_err(compute)?
            .transpose(0, 1)
            .map_err(compute)?
            .contiguous()
            .map_err(compute)?
            .reshape((sequence, hidden))
            .map_err(compute)?;
        let attention = attention
            .matmul(&self.weights.out_weight.t().map_err(compute)?)
            .map_err(compute)?
            .broadcast_add(&self.weights.out_bias)
            .map_err(compute)?;
        let residual = input.add(&attention).map_err(compute)?;
        let norm = layer_norm(
            &residual,
            &self.weights.norm2_weight,
            &self.weights.norm2_bias,
            self.eps,
        )?;
        let mlp = norm
            .matmul(&self.weights.fc1_weight.t().map_err(compute)?)
            .map_err(compute)?
            .broadcast_add(&self.weights.fc1_bias)
            .map_err(compute)?
            .gelu()
            .map_err(compute)?;
        let mlp = mlp
            .matmul(&self.weights.fc2_weight.t().map_err(compute)?)
            .map_err(compute)?
            .broadcast_add(&self.weights.fc2_bias)
            .map_err(compute)?;
        residual.add(&mlp).map_err(compute)
    }
}

fn compute(error: candle_core::Error) -> VisionError {
    VisionError::Image(error.to_string())
}

pub fn layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor, VisionError> {
    if input.device().is_metal() {
        input
            .apply_op3_no_bwd(weight, bias, &MetalLayerNorm { eps })
            .map_err(compute)
    } else {
        candle_nn::ops::layer_norm(input, weight, bias, eps).map_err(compute)
    }
}

fn softmax_last_dim(input: &Tensor) -> Result<Tensor, VisionError> {
    if input.device().is_metal() {
        input
            .apply_op1_no_bwd(&MetalSoftmaxLastDim)
            .map_err(compute)
    } else {
        candle_nn::ops::softmax_last_dim(input).map_err(compute)
    }
}

#[derive(Clone, Debug)]
struct MetalLayerNorm {
    eps: f32,
}

impl candle_core::CustomOp3 for MetalLayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Metal LayerNorm hook was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &candle_core::Layout,
        weight: &candle_core::MetalStorage,
        weight_layout: &candle_core::Layout,
        bias: &candle_core::MetalStorage,
        bias_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if !(input_layout.is_contiguous()
            && weight_layout.is_contiguous()
            && bias_layout.is_contiguous())
        {
            candle_core::bail!("non-contiguous LayerNorm is not implemented on Metal")
        }
        let kernel = match (input.dtype(), weight.dtype(), bias.dtype()) {
            (candle_core::DType::F32, candle_core::DType::F32, candle_core::DType::F32) => {
                "layernorm_f32"
            }
            (candle_core::DType::F16, candle_core::DType::F16, candle_core::DType::F16) => {
                "layernorm_f16"
            }
            (candle_core::DType::BF16, candle_core::DType::BF16, candle_core::DType::BF16) => {
                "layernorm_bf16"
            }
            (input_dtype, weight_dtype, bias_dtype) => candle_core::bail!(
                "Metal LayerNorm does not support {input_dtype:?}, {weight_dtype:?}, {bias_dtype:?}"
            ),
        };
        let device = input.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("layernorm");
        let elements = input_layout.shape().elem_count();
        let output = device.new_buffer(elements, input.dtype(), "layernorm")?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            elements,
            input_layout.dims()[input_layout.shape().rank() - 1],
            self.eps,
            input.buffer(),
            input_layout.start_offset() * input.dtype().size_in_bytes(),
            weight.buffer(),
            weight_layout.start_offset() * weight.dtype().size_in_bytes(),
            bias.buffer(),
            bias_layout.start_offset() * bias.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, input.dtype()),
            input_layout.shape().clone(),
        ))
    }
}

#[derive(Clone, Debug)]
struct MetalSoftmaxLastDim;

impl candle_core::CustomOp1 for MetalSoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Metal softmax hook was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if !layout.is_contiguous() || layout.stride().last() != Some(&1) {
            candle_core::bail!("non-contiguous softmax is not implemented on Metal")
        }
        let kernel = match input.dtype() {
            candle_core::DType::F32 => "softmax_f32",
            candle_core::DType::F16 => "softmax_f16",
            candle_core::DType::BF16 => "softmax_bf16",
            dtype => candle_core::bail!("Metal softmax does not support {dtype:?}"),
        };
        let device = input.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("softmax");
        let elements = layout.shape().elem_count();
        let output = device.new_buffer(elements, input.dtype(), "softmax")?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            elements,
            layout.dims()[layout.shape().rank() - 1],
            input.buffer(),
            layout.start_offset() * input.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, input.dtype()),
            layout.shape().clone(),
        ))
    }
}
