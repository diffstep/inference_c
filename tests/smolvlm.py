import time, torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("mps")

image = Image.open("/Users/dengtao2nd/Desktop/model_dev/inference_sdk/tests/smolvlm/test-data/statue-of-liberty.jpg")
messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": "Describe the image briefly."}
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to("mps")

torch.mps.synchronize()
t0 = time.time()
out = model.generate(**inputs, max_new_tokens=1024)
torch.mps.synchronize()
elapsed = time.time() - t0
new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
print(f"{new_tokens} tokens in {elapsed*1000:.1f}ms -> {new_tokens/elapsed:.1f} tok/s")