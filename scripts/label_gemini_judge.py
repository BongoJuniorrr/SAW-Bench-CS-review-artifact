import argparse
import json
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import _bootstrap  # noqa: F401
from saw_bench_cs.io import load_warnings
from saw_bench_cs.schema import Warning

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"

def call_gemini(prompt: str) -> dict:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    data_bytes = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(ENDPOINT, data=data_bytes, headers={'Content-Type': 'application/json'})
    
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_text = response.read().decode('utf-8')
                data = json.loads(resp_text)
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                # Clean markdown code blocks if any
                if text.startswith("```json"):
                    text = text.strip()[7:-3].strip()
                elif text.startswith("```"):
                    text = text.strip()[3:-3].strip()
                return json.loads(text)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Other Error: {e}")
            time.sleep(2 ** attempt)
    print("API call failed after 5 attempts.")
    return {"labels": {}, "rationales": {}}

def label_warning(w: Warning) -> dict:
    prompt = f"""You are an expert Java developer reviewing a static analysis warning from SpotBugs.
Warning Rule: {w.rule_id}
Message: {w.warning_message}

You have {len(w.candidate_snippets)} candidate snippets of code to review. 
Your task is to assign a relevance label to each snippet:
- "essential": strictly necessary to understand the cause or determine if the warning is a false positive. (At most 3 snippets may be essential).
- "helpful": provides useful background context but not strictly necessary.
- "irrelevant": redundant, distracting, or completely unrelated.

For any snippet labeled "essential", provide a 1-sentence rationale explaining why.

Candidate Snippets:
"""
    for s in w.candidate_snippets:
        prompt += f"\n--- Snippet {s.snippet_id} ({s.type}) ---\n"
        prompt += s.text[:1500] + "\n" # truncate to avoid massive prompts

    prompt += """
Return ONLY a valid JSON object in this exact format, with no other text:
{
  "labels": {
    "s01": "essential",
    "s02": "helpful",
    "s03": "irrelevant"
  },
  "rationales": {
    "s01": "Brief rationale here."
  }
}
"""
    result = call_gemini(prompt)
    
    # Enforce constraints
    labels = result.get("labels", {})
    rationales = result.get("rationales", {})
    
    # Ensure all snippets have a label, default irrelevant
    for s in w.candidate_snippets:
        if s.snippet_id not in labels:
            labels[s.snippet_id] = "irrelevant"
            
    # Cap essentials at 3
    essential_count = 0
    for sid in list(labels.keys()):
        if labels[sid] == "essential":
            if essential_count >= 3:
                labels[sid] = "helpful"
            else:
                essential_count += 1
                if sid not in rationales:
                    rationales[sid] = "The LLM identified this as essential evidence."
    
    return {
        "warning_id": w.warning_id,
        "annotator": "artifact_labeler_B",
        "labels": labels,
        "rationales": rationales
    }

def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    print("Loading unlabeled warnings...")
    warnings = load_warnings("data/public_java_100/saw_bench_cs_unlabeled.jsonl")
    
    # Check if we already have a partial file to resume
    out_file = "annotation/gemini_passes_b.jsonl"
    temp_out = out_file + ".tmp"
    
    successful_warnings = set()
    processed_warnings = set()
    all_gemini_passes = {}

    def load_progress(path):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            p = json.loads(line)
                            wid = p["warning_id"]
                            all_gemini_passes[wid] = p
                            processed_warnings.add(wid)
                            if any(l == "essential" for l in p["labels"].values()):
                                successful_warnings.add(wid)
                        except json.JSONDecodeError:
                            continue

    # Load from main file and then temp file (temp overrides main)
    load_progress(out_file)
    load_progress(temp_out)
                    
    print(f"Loaded {len(warnings)} warnings. {len(processed_warnings)} already processed. {len(successful_warnings)} successful.")
    
    # We want to process anything that isn't successful AND isn't already in the temp file
    warnings_to_process = [w for w in warnings if w.warning_id not in processed_warnings and w.warning_id not in successful_warnings]
    
    print(f"Warnings remaining to process: {len(warnings_to_process)}")

    # Open temp file in append mode if it exists, else write mode
    mode = "a" if os.path.exists(temp_out) else "w"
    with open(temp_out, mode) as f:
        # If starting fresh, write the already successful ones from the main file
        if mode == "w":
            for wid in successful_warnings:
                f.write(json.dumps(all_gemini_passes[wid]) + "\n")
            
        for i, w in enumerate(warnings_to_process):
            print(f"[{i+1}/{len(warnings_to_process)}] Processing {w.warning_id}...")
            try:
                res = label_warning(w)
                f.write(json.dumps(res) + "\n")
                f.flush()
                # Strict 5s sleep to stay under 15 RPM (60/15 = 4s, 5s is safer)
                time.sleep(5.0) 
            except Exception as e:
                print(f"Failed to process {w.warning_id}: {e}")
                
    os.replace(temp_out, out_file)
    print(f"Finished generating all {len(warnings)} passes to {out_file}.")
    
    # Now merge back into annotator_passes.jsonl
    existing_passes = []
    if os.path.exists("annotation/annotator_passes.jsonl"):
        with open("annotation/annotator_passes.jsonl") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p["annotator"] != "artifact_labeler_B":
                        existing_passes.append(p)
                    
    with open(out_file) as f:
        for line in f:
            if line.strip():
                existing_passes.append(json.loads(line))
                
    with open("annotation/annotator_passes.jsonl", "w") as f:
        for p in existing_passes:
            f.write(json.dumps(p) + "\n")
            
    print("Successfully updated annotation/annotator_passes.jsonl")

if __name__ == "__main__":
    main()
