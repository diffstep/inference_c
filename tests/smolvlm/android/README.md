# SmolVLM Android / QNN

This Android example connects camera capture to SmolVLM caption generation on
Qualcomm QNN. FP16 model execution uses the QNN GPU backend. At startup the app
checks the HTP v73 runtime with a ReLU smoke graph, then checks QNN GPU Linear,
RMSNorm, embedding, RoPE, gated MLP, and GQA attention against known answers.

The connected test phone reports that its HTP SoC model does not support FP16.
This SDK does not yet have a quantized QNN HTP model path, so the FP16 SmolVLM
example selects `INFERENCE_SDK_QNN_BACKEND=gpu`.

## Build and run

Install Android SDK command-line/build tools, Android NDK, Gradle, and QAIRT.
The default local paths are `~/Library/Android/sdk` and
`~/qairt/2.50.0.260828`; override them as needed:

```bash
ANDROID_SDK_ROOT="$HOME/Library/Android/sdk" \
QAIRT_SDK_ROOT="$HOME/qairt/2.50.0.260828" \
SMOLVLM_MODEL_PATH="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67edbbed1bf9888184d9df282b700a323964/model.safetensors" \
./tests/smolvlm/android/build.sh build

./tests/smolvlm/android/build.sh install
./tests/smolvlm/android/build.sh run
```

`ANDROID_SERIAL` selects a device when multiple devices are attached. The APK
contains QAIRT Android HTP and GPU runtime libraries, the HTP v73 DSP skeleton,
the matching model config, generation config, tokenizer, and official 489.3 MiB
SmolVLM-256M-Instruct Safetensors checkpoint. Weights and metadata are copied
from the same Hugging Face model directory at build time. Set
`SMOLVLM_MODEL_PATH=/path/to/model.safetensors` to bundle another checkpoint;
its `config.json`, `generation_config.json`, and `tokenizer.json` must be beside
it. The first launch copies the checkpoint into app-private storage, requiring
about 490 MiB of additional free device storage; later launches reuse that copy.

Take a photo, then tap “运行 SmolVLM 图片识别”. No model picker is needed. The
file descriptor remains open while the native runner
loads the vision and text models, preprocesses image tiles, encodes them,
formats the image prompt, and generates up to 64 tokens. The app displays the
caption or the first native error.

The example internally selects `INFERENCE_SDK_QNN_BACKEND=gpu`,
`TENSOR_BACKEND=qnn`, `TENSOR_ATTENTION_BACKEND=qnn`, `TENSOR_DTYPE=fp16`, and
`TENSOR_WEIGHT_QUANT=fp16`. Model operators use QNN callbacks and propagate
backend failures. Image decoding/preprocessing, tokenizer/prompt handling,
sampling, and pixel-shuffle/layout rearrangement use the CPU. The checkpoint is
opened directly from app-private storage.

On the connected phone, the HTP ReLU smoke and all QNN GPU primitive known-answer
checks passed. Full model inference and end-to-end performance should be verified
after installing the bundled-checkpoint APK on the phone.
