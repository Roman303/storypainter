#!/usr/bin/env python3
"""
scene_extractor_local.py – Offline GPU scene segmenter (XML workflow, visual dictionary)
---------------------------------------------------------------------------------------
All comments and console logs are in English (plain text with ANSI colors).

Key features:
- Fully local (no HTTP). Uses HuggingFace transformers to run the LLM.
- GPU-aware: prints GPU name + VRAM, clears CUDA cache, sets seeds for repeatability.
- Precision selectable via --precision [8bit|16bit|32bit]. Defaults to 16bit.
- Stateless per block: the full instruction + context is sent for each block.
- Robust: per-block logical retries if XML is invalid (up to 5 attempts).
- XML repair attempt with explicit notice in console.
- Sentence-boundary correction (never end a scene at a comma).
- Visual Dictionary: forces replacing bare names with full visual descriptions.
- Prompt normalization (<= 300 chars).
- Clean JSON output with absolute offsets and original text slices.

CLI:
  python3 scene_extractor_local.py \
    --book mein_buch.txt \
    --context mein_context.json \
    --output book_scenes.json \
    --block-size 8000 \
    --scene-size 700 \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --precision 16bit
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# ------------------------------
# ANSI colors (auto-disabled if not a TTY; env SCENE_EXTRACTOR_COLOR overrides)
# ------------------------------
class Ansi:
    def __init__(self) -> None:
        enable = sys.stdout.isatty()
        if os.getenv("SCENE_EXTRACTOR_COLOR") == "0":
            enable = False
        elif os.getenv("SCENE_EXTRACTOR_COLOR") == "1":
            enable = True
        if enable:
            self.RESET = "\033[0m"
            self.CYAN  = "\033[36m"
            self.GREEN = "\033[32m"
            self.YELLOW= "\033[33m"
            self.RED   = "\033[31m"
        else:
            self.RESET = self.CYAN = self.GREEN = self.YELLOW = self.RED = ""

ANSI = Ansi()
def log_info(msg: str) -> None:  print(f"{ANSI.CYAN}INFO{ANSI.RESET}  {msg}")
def log_ok(msg: str) -> None:    print(f"{ANSI.GREEN}OK{ANSI.RESET}    {msg}")
def log_warn(msg: str) -> None:  print(f"{ANSI.YELLOW}WARN{ANSI.RESET}  {msg}")
def log_error(msg: str) -> None: print(f"{ANSI.RED}ERROR{ANSI.RESET} {msg}")

# ------------------------------
# HF transformers
# ------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig  # optional (8bit)
    HAS_BNB = True
except Exception:
    HAS_BNB = False

# ------------------------------
# Data models
# ------------------------------
@dataclass
class Block:
    id: int
    start: int
    end: int
    text: str

@dataclass
class Scene:
    start_pos: int
    end_pos: int
    text: str
    image_prompt: str

# ------------------------------
# SceneExtractor (local model)
# ------------------------------
class SceneExtractor:
    def __init__(
        self,
        path: str = ".",
        book: str = "book.txt",
        context: str = "book_context.json",
        output: str = "book_scenes.json",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        precision: str = "16bit",  # 8bit | 16bit | 32bit
        scene_size: int = 700,
        block_size: int = 8000,
        tolerance: float = 0.35,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 1500,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.precision = precision
        self.scene_size = scene_size
        self.block_size = block_size
        self.tolerance = tolerance
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.path = path
        self.book = os.path.join(path, book)
        self.context = os.path.join(path, context)
        self.output = os.path.join(path, output)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log_ok(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            torch.cuda.empty_cache()
        else:
            log_warn("CUDA not available, running on CPU (will be slow)")

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.context: Dict = {}
        self.failed_blocks: List[int] = []
        self.repaired_blocks: List[int] = []

        self._load_model()

    # ------------------------------
    # Model loading
    # ------------------------------
    def _load_model(self) -> None:
        log_info(f"Loading model: {self.model_name} [precision={self.precision}]")
        dtype = torch.float16 if self.precision == "16bit" else (torch.float32 if self.precision == "32bit" else torch.float16)
        quant_cfg = None
        if self.precision == "8bit":
            if not HAS_BNB:
                log_warn("bitsandbytes not available; falling back to 16-bit")
            else:
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if quant_cfg is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=quant_cfg, device_map="auto", trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
            )
        log_ok("Model loaded")

    # ------------------------------
    # I/O
    # ------------------------------
    def load_context(self, context_path: str) -> None:
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                self.context = json.load(f)
            log_ok(
                f"Context loaded: {len(self.context.get('characters', []))} characters, "
                f"{len(self.context.get('locations', []))} locations, style='{self.context.get('style', 'n/a')}'"
            )
        except Exception as e:
            log_warn(f"Context could not be loaded: {e}")
            self.context = {}

    def read_book(self, book_path: str) -> str:
        with open(book_path, "r", encoding="utf-8") as f:
            text = f.read()
        log_ok(f"Book loaded: {len(text)} chars")
        return text

    # ------------------------------
    # Block splitting
    # ------------------------------
    def split_into_blocks(self, text: str) -> List[Block]:
        blocks: List[Block] = []
        n = len(text)
        i = 0
        block_id = 1
        while i < n:
            j = min(i + self.block_size, n)
            j = self._find_natural_block_end(text, i, j)
            blocks.append(Block(id=block_id, start=i, end=j, text=text[i:j]))
            block_id += 1
            i = j
        return blocks

    def _find_natural_block_end(self, text: str, start: int, end: int) -> int:
        if end >= len(text):
            return len(text)
        s = max(start, end - 300)
        e = min(len(text), end + 150)
        # Prefer paragraph break
        p = text.find("\n\n", s, e)
        if p != -1:
            return p + 2
        # Then try sentence end going forward
        for k in range(end, e):
            ch = text[k]
            if ch in ".!?" and (k + 1 >= len(text) or text[k + 1] in " \n\t\"«»'“”»«"):
                return k + 1
        # Or backward
        for k in range(end, s, -1):
            ch = text[k - 1]
            if ch in ".!?":
                return k
        # Fallback: hard cut
        return end

    # ------------------------------
    # Visual Dictionary helper
    # ------------------------------
    def _format_context_visual_dict(self, entries: List[Dict]) -> str:
        if not entries:
            return "(none)"
        # Name => full visual description (for inline replacement)
        return "\n".join([f"{e.get('name','?')} => {e.get('description','')}" for e in entries])

    def _format_context_entries(self, entries: List[Dict]) -> str:
        if not entries:
            return "(none)"
        return "\n".join([f"- {e.get('name', '?')}: {e.get('description', '')}" for e in entries])

    # ------------------------------
    # Prompt building
    # ------------------------------
    # ------------------------------
    # Prompt building – SEGMENTATION (1st pass)
    # ------------------------------
    def make_segmentation_prompt(self, block: Block) -> str:
        min_len = math.floor(self.scene_size * (1 - self.tolerance))
        max_len = math.ceil(self.scene_size * (1 + self.tolerance))
        instruction = f"""
You are a veteran film editor and scene segmenter. Your task is to segment the following text BLOCK into VISUAL scenes and return STRICT XML.

SEGMENTATION RULES (CRITICAL):
1) Prefer splitting when LOCATION, TIME or MAIN CHARACTER changes.
2) Target scene length: ~{self.scene_size} characters (acceptable range: {min_len}-{max_len}).
3) If no natural change occurs and the text exceeds {max_len} characters, you MUST split anyway near the next sentence boundary.
4) NEVER end a scene at a comma. Scenes must end at a sentence boundary ('.', '!', '?' or paragraph break).
5) Each scene must be a contiguous substring of the BLOCK.
6) start and end are 0-based offsets RELATIVE TO THIS BLOCK.

Return ONLY well-formed XML:
<scenes>
  <scene start="START" end="END"/>
  ...
</scenes>

BLOCK (length: {len(block.text)}):
<BLOCK>
{block.text}
</BLOCK>
""".strip()
        return instruction

    # ------------------------------
    # Prompt building – IMAGE PROMPT (2nd pass)
    # ------------------------------
    def make_image_prompt(self, scene_text: str) -> str:
        characters_dict = self._format_context_visual_dict(self.context.get("characters", []))
        locations_dict  = self._format_context_visual_dict(self.context.get("locations", []))
        instruction = f"""
You are a visual prompt generator for a text-to-image model.
Use the SCENE TEXT plus the VISUAL DICTIONARIES below to write ONE IMAGE PROMPT (CRITICAL):

IMAGE PROMPTS (CRITICAL):
- Max 300 characters, commas only (no full sentences).
- Visual and observable only (no dialogue, thoughts, or actions like “turns”, “looks”, “reacts”).
- Do NOT use bare names. Inline the FULL visual description from the VISUAL DICTIONARY.
- Structure hint: "subject + 2–3 traits, location + 1–2 details".
- Each prompt must be fully self-contained and restate the full visual context.

VISUAL DICTIONARY — CHARACTERS:
{characters_dict}

VISUAL DICTIONARY — LOCATIONS:
{locations_dict}

SCENE TEXT:
<SCENE>
{scene_text}
</SCENE>

Output ONLY the final prompt, nothing else.
""".strip()
        return instruction

    # ------------------------------
    # Generate image prompt for one scene
    # ------------------------------
    def generate_image_prompt(self, scene_text: str) -> str:
        """Generate one short image prompt for the given scene text using the LLM."""
       
        prompt = self.make_image_prompt(scene_text)

        # 🧩 we allow longer input (scene_text is short anyway)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.model.device)

        # --- inference ---
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # --- handle empty or weird output ---
        if not raw or len(raw) < 10:
            log_warn("Empty image prompt, retrying with higher temperature")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=min(self.temperature * 1.3, 0.9),
                    do_sample=True,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # --- clean up output ---
        raw = raw.strip('"').strip("'").strip()
        return self._normalize_prompt(raw)

    # ------------------------------
    # Main pipeline (two-stage)
    # ------------------------------
    def analyze_book(self, text: str) -> List[Scene]:
        all_scenes: List[Scene] = []
        blocks = self.split_into_blocks(text)
        log_info(f"Starting 2-pass analysis – {len(blocks)} blocks")

        # --- PASS 1: segmentation ---
        raw_segments: List[Scene] = []
        for b in blocks:
            log_info(f"Segmenting block {b.id} ({len(b.text)} chars)")
            prompt = self.make_segmentation_prompt(b)
            xml_out = self.llm_generate(prompt)
            rel_scenes, repaired = self.parse_llm_response(xml_out, b)
            if repaired:
                log_warn(f"XML repaired in block {b.id}")
            if not rel_scenes:
                log_error(f"No scenes detected in block {b.id}")
                continue
            fixed = self.correct_scene_boundaries(b, rel_scenes)
            raw_segments.extend(fixed)
            log_ok(f"Block {b.id}: {len(fixed)} segments found")

        # --- PASS 2: image prompt generation ---
        log_info(f"Generating image prompts for {len(raw_segments)} scenes...")
        final_scenes: List[Scene] = []
        for s in raw_segments:
            try:
                img_prompt = self.generate_image_prompt(s.text)
                s.image_prompt = img_prompt
                final_scenes.append(s)
                log_ok(f"Prompt ok ({len(img_prompt)} chars)")
            except Exception as e:
                log_warn(f"Prompt generation failed: {e}")
                s.image_prompt = ""
                final_scenes.append(s)
                log_info(f"Scene text sample (first 120 chars): {s.text[:120]!r}")

        return final_scenes


    # ------------------------------
    # Local LLM generate
    # ------------------------------
    def llm_generate(self, prompt: str) -> str:
        assert self.model is not None and self.tokenizer is not None
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    # ------------------------------
    # XML parsing + repair
    # ------------------------------
    def parse_llm_response(self, xml_str: str, block: Block) -> Tuple[List[Tuple[int, int, str]], bool]:
        results: List[Tuple[int, int, str]] = []
        repaired = False
       
        if not xml_str.strip():
            return results, repaired
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            xml_str2 = self._extract_first_scenes_xml(xml_str)
            if not xml_str2:
                frags = re.findall(r"<scene[\s\S]*?>[\s\S]*?</scene>", xml_str)
                if frags:
                    xml_str2 = "<scenes>" + "".join(frags) + "</scenes>"
            if not xml_str2:
                return results, repaired
            try:
                root = ET.fromstring(xml_str2)
                repaired = True
            except ET.ParseError:
                return results, repaired
        for node in root.findall("scene"):
            try:
                s = int((node.get("start", "0") or "0").strip())
                e = int((node.get("end", "0") or "0").strip())
                p = (node.findtext("prompt", default="") or "").strip()
            except Exception:
                continue
            if e > s:
                results.append((s, e, p))
        results.sort(key=lambda t: t[0])
        # Deduplicate/clip
        dedup: List[Tuple[int, int, str]] = []
        last_end = -1
        for s, e, p in results:
            if s < last_end:
                s = last_end
            e = min(e, len(block.text))
            if e > s:
                dedup.append((s, e, p))
                last_end = e
        return dedup, repaired

    def _extract_first_scenes_xml(self, s: str) -> str:
        m = re.search(r"<scenes>[\s\S]*?</scenes>", s)
        return m.group(0) if m else ""

    # ------------------------------
    # Boundary correction / slicing
    # ------------------------------
    def correct_scene_boundaries(self, block: Block, rel_scenes: List[Tuple[int, int, str]]) -> List[Scene]:
        min_len = math.floor(self.scene_size * (1 - self.tolerance))
        max_len = math.ceil(self.scene_size * (1 + self.tolerance))
        fixed: List[Scene] = []
        cursor = 0
        for (rs, re_, prompt) in rel_scenes:
            rs = max(rs, cursor)
            re_ = max(re_, rs + 1)
            rs = self._adjust_to_sentence_start(block.text, rs)
            re_ = self._adjust_to_sentence_end(block.text, re_)
            while rs < re_:
                target_end = min(rs + max_len, re_)
                slice_end = self._adjust_to_sentence_end(block.text, target_end)
                if slice_end <= rs:
                    slice_end = self._find_next_sentence_end(block.text, rs)
                    if slice_end == rs:
                        slice_end = self._find_next_paragraph_or_end(block.text, rs)
                slice_end = self._avoid_comma_end(block.text, slice_end)
                abs_s = block.start + rs
                abs_e = block.start + slice_end
                # --- Text für diese Szene extrahieren ---
                seg_text = block.text[rs:slice_end]

                # 🔧 Entferne alles am Anfang, bis der erste echte Buchstabe/Zahl auftaucht
                # (einschließlich Guillemets, Quotes, Newlines, Leerzeichen usw.)
                seg_text = re.sub(r'^[\s«»"„“‚‘\'\n\r\t]+', '', seg_text)
                
                # 🔧 Nur noch überzählige Newlines am Ende entfernen
                seg_text = re.sub(r'\n+\s*$', '', seg_text)
                
                seg_text = seg_text.strip()
                
                # enforce visual descriptions on the prompt
                img_prompt = self._post_enforce_visual_descriptions(self._normalize_prompt(prompt))
                
                # 🔧 Falls das LLM den Prompt in Anführungszeichen gepackt hat → abstreifen
                #    (z.B.  "foo, bar, baz"  ->  foo, bar, baz)
                if img_prompt.startswith('"') and img_prompt.endswith('"') and len(img_prompt) >= 2:
                    img_prompt = img_prompt[1:-1].strip()
                
                if seg_text:
                    fixed.append(Scene(abs_s, abs_e, seg_text, img_prompt))

                rs = slice_end
            cursor = rs
        return fixed

    # Sentence helpers
    def _is_sentence_end(self, text: str, idx: int) -> bool:
        if idx <= 0 or idx > len(text):
            return False
        ch = text[idx - 1]
        if ch not in ".!?":
            return False
        if idx == len(text):
            return True
        nxt = text[idx] if idx < len(text) else ""
        return nxt in " \n\t\"'»«”“\u00ab\u00bb"

    def _adjust_to_sentence_end(self, text: str, pos: int) -> int:
        limit = min(len(text), pos + 300)
        k = pos
        while k < limit:
            if self._is_sentence_end(text, k + 1):
                return k + 1
            k += 1
        k = pos
        back_limit = max(0, pos - 200)
        while k > back_limit:
            if self._is_sentence_end(text, k):
                return k
            k -= 1
        return pos

    def _adjust_to_sentence_start(self, text: str, pos: int) -> int:
        k = pos
        back_limit = max(0, pos - 200)
        while k > back_limit:
            if self._is_sentence_end(text, k):
                break
            if text[k - 1:k + 1] == "\n\n":
                break
            k -= 1
        while k < len(text) and text[k] in " \n\t":
            k += 1
        return k

    def _find_next_sentence_end(self, text: str, pos: int) -> int:
        k = pos
        limit = min(len(text), pos + 1000)
        while k < limit:
            if self._is_sentence_end(text, k + 1):
                return k + 1
            k += 1
        return pos

    def _find_next_paragraph_or_end(self, text: str, pos: int) -> int:
        p = text.find("\n\n", pos)
        return p if p != -1 else len(text)

    def _avoid_comma_end(self, text: str, pos: int) -> int:
        if pos > 0 and text[pos - 1] == ',':
            next_end = self._find_next_sentence_end(text, pos)
            return max(pos, next_end)
        return pos

    # ------------------------------
    # Prompt normalization + post-enforcement of visual descriptions
    # ------------------------------
    def _normalize_prompt(self, prompt: str) -> str:
        p = (prompt or "").strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) > 300:
            p = p[:297].rstrip() + "..."
        return p

    def _post_enforce_visual_descriptions(self, prompt: str) -> str:
        """Replace any bare names with full visual descriptions from context."""
        p = prompt
        # Replace character names with their full visual descriptions
        for c in self.context.get("characters", []):
            name = c.get("name", "")
            desc = c.get("description", "")
            if name and desc:
                p = re.sub(rf"\b{re.escape(name)}\b", desc, p)
        # Replace location names similarly
        for l in self.context.get("locations", []):
            name = l.get("name", "")
            desc = l.get("description", "")
            if name and desc:
                p = re.sub(rf"\b{re.escape(name)}\b", desc, p)
        return self._normalize_prompt(p)

# ------------------------------
# Save JSON
# ------------------------------
def save_scenes_json(scenes: List[Scene], output_path: str, full_text: str, context: Dict) -> None:
    # 🔧 Buch-Metadaten exakt aus book_context übernehmen (so wie dort formatiert)
    #    und nur total_chars / total_scenes hinzufügen.
    book_info_src = context.get("book_info", {})
    book_info = dict(book_info_src)  # flache Kopie (keine Veränderung am Original)

    # Pflichtfelder ergänzen/überschreiben
    book_info["total_chars"] = len(full_text)
    book_info["total_scenes"] = len(scenes)

    data = {
        "book_info": book_info,
        "scenes": [
            {
                "start_pos": s.start_pos,
                "end_pos": s.end_pos,
                "text": full_text[s.start_pos:s.end_pos],
                "image_prompt": s.image_prompt,
            }
            for s in scenes
        ],
    }

    out = Path(output_path)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    log_ok(f"Saved JSON: {out} ({size_kb:.1f} KB), scenes={len(scenes)}")


# ------------------------------
# CLI
# ------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Offline scene extractor (XML workflow, local GPU, visual dictionary)")
    ap.add_argument("--path", required=True, help="Path to the input text file")
    ap.add_argument("--book", default="book.txt", help="input text file")
    ap.add_argument("--context", default="book_context.json", help="input .json")
    ap.add_argument("--output", default="book_scenes.json", help="Path to output JSON file")
    ap.add_argument("--block-size", type=int, default=4000, help="Characters per LLM block")
    ap.add_argument("--scene-size", type=int, default=1000, help="Target scene size (+/- 35%)")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF model id to load")
    ap.add_argument("--precision", choices=["8bit", "16bit", "32bit"], default="16bit", help="Model precision")
    ap.add_argument("--max-input-tokens", type=int, default=7000, help="Tokenizer truncate length")
    ap.add_argument("--max-new-tokens", type=int, default=350, help="Max new tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    args = ap.parse_args()

     # 📁 Alle Datei-Pfade mit dem übergebenen --path kombinieren
    args.book = os.path.join(args.path, args.book)
    args.context = os.path.join(args.path, args.context)
    args.output = os.path.join(args.path, args.output)

    extractor = SceneExtractor(
        model_name=args.model,
        precision=args.precision,
        scene_size=args.scene_size,
        block_size=args.block_size,
        tolerance=0.35,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    text = extractor.read_book(args.book)
    extractor.load_context(args.context)

    scenes = extractor.analyze_book(text)
    save_scenes_json(scenes, args.output, text, extractor.context)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_warn("Interrupted by user")
