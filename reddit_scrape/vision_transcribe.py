import torch, os, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")

if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is NOT available.")
    device = torch.device("cpu")

import torch.cuda.amp as amp
scaler = amp.GradScaler()

from transformers import AutoModelForCausalLM
from PIL import Image

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# Use the small model
model_path = "deepseek-ai/deepseek-vl2-small"

import warnings, sys, io
from contextlib import redirect_stdout

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def get_images(folder_id="1mvldbb"):
    folder_path = os.path.join("images", folder_id)

    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    try:
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        )
    except FileNotFoundError:
        print(f"No image folder found for {folder_id}")
        return []

    images = []
    for fname in image_files:
        try:
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    print(f"Loaded {len(images)} images from {folder_path}")
    return images


def _is_oom_error(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda error" in msg and "memory" in msg)


def transcribe(id="1mvldbb"):
    images = get_images(folder_id=id)
    if len(images) == 0 or len(images) > 9:
        # no images or too many images -> treat as empty transcription
        return ""

    try:
        # Load processor/tokenizer quietly
        f = io.StringIO()
        with redirect_stdout(f):
            vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
            tokenizer = vl_chat_processor.tokenizer

        torch.cuda.empty_cache()

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        vl_gpt.eval()

        # One <image> token per screenshot
        image_tokens = "\n".join(["<image>"] * len(images))

        SYSTEM_PROMPT = ""

        # Prompt to transcribe images
        USER_PROMPT = (
            "You are given one or more screenshots of a two-person chat conversation.\n"
            "Read all of the chat bubbles and output exactly one JSON object:\n"
            "{ \"messages\": [\n"
            "  { \"speaker\": \"Speaker A\", \"text\": \"...\" },\n"
            "  { \"speaker\": \"Speaker B\", \"text\": \"...\" }\n"
            "]}\n"
            "\n"
            "How to decide who is speaking:\n"
            "1. Look at each bubble's visual SIDE and STYLE (left vs right, bubble color, or name label).\n"
            "2. The first side/style you see = Speaker A. The other side/style = Speaker B.\n"
            "3. For EACH bubble, choose the speaker ONLY from its side/style, NOT from the order of messages.\n"
            "4. If several bubbles in a row are on the same side/style, keep the SAME speaker. Do NOT alternate speakers just because it is a new message.\n"
            "\n"
            "Formatting rules:\n"
            "- Output one JSON object with a single key \"messages\".\n"
            "- \"messages\" is a list of objects with exactly two keys: \"speaker\" and \"text\".\n"
            "- \"speaker\" must be \"Speaker A\" or \"Speaker B\".\n"
            "- \"text\" must contain the bubble text with no line breaks (replace line breaks with spaces).\n"
            "- Do not include timestamps, IDs, or any other keys.\n"
            "- If the screenshots are NOT a two-person chat, output exactly: { \"messages\": [] }.\n"
            "- The response must be pure JSON and nothing else.\n"
        )


        conversation = [
            {
                "role": "<|User|>",
                "content": f"{image_tokens}\n{USER_PROMPT}",
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        model_inputs = vl_chat_processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=SYSTEM_PROMPT
        ).to(vl_gpt.device)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**model_inputs)

        with torch.no_grad():
            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=model_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=2048,
                repetition_penalty=1.05,
                do_sample=False,
                use_cache=True
            )

        raw_text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True).strip()

    except Exception as e:
        if _is_oom_error(e):
            print(f"Out of memory while transcribing {id}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        raise

    print("\nGot Raw Text\n")

    # Try to isolate a JSON object
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_text = raw_text[start:end+1]

    data = json.loads(raw_text)

    # If it's the special non-chat case, pass it through as-is.
    msgs_in = data.get("messages", [])
    if isinstance(msgs_in, list) and len(msgs_in) == 0:
        print("\nDetected non-chat screenshots (messages = []).\n")
        return json.dumps({"messages": []}, ensure_ascii=False)

    # SIMPLE POST-PROCESSING:
    # - Normalize whitespace in text
    # - Respect Speaker A/B if the model already used them
    # - Otherwise, map whatever labels it used to A/B
    speaker_map = {}
    out_msgs = []
    last_speaker_label = None

    for m in msgs_in:
        raw_speaker = str(m.get("speaker", "")).strip()
        text = (m.get("text") or "").strip()

        if not text:
            continue

        # Normalize text: collapse internal whitespace
        text = " ".join(text.split())

        # If the model already chose Speaker A / B, just keep that
        if raw_speaker in ("Speaker A", "Speaker B"):
            normalized_speaker = raw_speaker
        else:
            # If no label at all, default to last speaker instead of creating a new one
            if not raw_speaker:
                raw_speaker = last_speaker_label or "Speaker A"

            if raw_speaker not in speaker_map:
                if len(speaker_map) == 0:
                    speaker_map[raw_speaker] = "Speaker A"
                elif len(speaker_map) == 1:
                    speaker_map[raw_speaker] = "Speaker B"
                else:
                    # More than two different labels: reuse last speaker to avoid crazy alternation
                    speaker_map[raw_speaker] = speaker_map.get(last_speaker_label, "Speaker B")

            normalized_speaker = speaker_map[raw_speaker]

        last_speaker_label = raw_speaker

        out_msgs.append({
            "speaker": normalized_speaker,
            "text": text
        })


    data_out = {"messages": out_msgs}
    print("\nDid post processing\n")

    return json.dumps(data_out, ensure_ascii=False)
