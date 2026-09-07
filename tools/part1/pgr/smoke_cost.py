"""Sum the proxy request log for the post-graph-rag smoke and project the cost.

Reads:
  env/pgr-smoke-phases.json   line ranges of requests.jsonl per smoke phase
  env/requests.jsonl          the litellm per-request log (shared by every stage)
  env/models.json             prices, USD per 1M tokens

The log is shared with whatever else was using the proxy at the time, so rows
are attributed to the smoke by model: chat rows on the smoke model, embedding
rows on the smoke embedding model with at least MIN_EMBED_TOKENS prompt tokens
(the foreign embedding probes in the same window were 5 to 12 tokens; the
smoke's smallest embedding, a bare query, was 25).
"""
import json
import pathlib
import sys

ENV = pathlib.Path(__file__).resolve().parent.parent / "env"
CHAT_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"
MIN_EMBED_TOKENS = 20
CHUNK_CHARS = [1238, 1126, 1095]          # the three smoke documents
TARGET_CHUNK_CHARS = 2000
CHARS_PER_TOKEN = 1238 / 277              # document 1: 1238 chars embedded as 277 tokens

phases = json.loads((ENV / "pgr-smoke-phases.json").read_text())
prices = json.loads((ENV / "models.json").read_text())["prices"]
rows = [json.loads(l) for l in (ENV / "requests.jsonl").open() if l.strip()]


def mine(r):
    m = r["model"].split("/")[-1]
    if m == CHAT_MODEL and r["call_type"] == "acompletion":
        return "chat"
    if m == EMBED_MODEL and r["call_type"] == "aembedding" and (r["prompt_tokens"] or 0) >= MIN_EMBED_TOKENS:
        return "embed"
    return None


def cost(model, tin, tout):
    p = prices[model]
    return tin * p["in"] / 1e6 + tout * p["out"] / 1e6


grand = 0.0
per_phase = {}
for name, (start, end) in phases.items():
    sel = rows[start:end]                  # rows are 0-based; the phase file stores line counts
    chat = [r for r in sel if mine(r) == "chat"]
    emb = [r for r in sel if mine(r) == "embed"]
    foreign = [r for r in sel if mine(r) is None]
    c_in = sum(r["prompt_tokens"] for r in chat)
    c_out = sum(r["completion_tokens"] for r in chat)
    e_in = sum(r["prompt_tokens"] for r in emb)
    usd = cost(CHAT_MODEL, c_in, c_out) + cost(EMBED_MODEL, e_in, 0)
    litellm_usd = sum(r.get("response_cost") or 0 for r in chat + emb)
    grand += usd
    per_phase[name] = dict(chat_calls=len(chat), chat_in=c_in, chat_out=c_out,
                           embed_calls=len(emb), embed_in=e_in,
                           foreign_rows_in_window=len(foreign),
                           usd=round(usd, 5), litellm_usd=round(litellm_usd, 5))
    print(f"{name:<16} chat {len(chat)} calls in={c_in} out={c_out} | embed {len(emb)} calls in={e_in} "
          f"| foreign rows {len(foreign)} | USD {usd:.5f} (litellm {litellm_usd:.5f})")
print(f"{'TOTAL':<16} USD {grand:.5f}")

# ---- projection to 1,000 chunks of 2,000 characters -------------------------
idx = per_phase["index"]
n = len(CHUNK_CHARS)
mean_chars = sum(CHUNK_CHARS) / n
scale = TARGET_CHUNK_CHARS / mean_chars
text_tokens_now = mean_chars / CHARS_PER_TOKEN
text_tokens_target = TARGET_CHUNK_CHARS / CHARS_PER_TOKEN
# input: fixed prompt overhead per call plus the chunk text; both chat calls carry the text
chat_in_per_chunk = idx["chat_in"] / n
overhead_in = chat_in_per_chunk - 2 * text_tokens_now
chat_in_target = overhead_in + 2 * text_tokens_target
chat_out_per_chunk = idx["chat_out"] / n
embed_per_chunk = idx["embed_in"] / n
# two embedding rows (the chunk embeddings of documents 2 and 3) are missing from the log;
# add them back from their character counts so the projection is not short
embed_missing = sum(c / CHARS_PER_TOKEN for c in CHUNK_CHARS[1:])
embed_per_chunk_full = (idx["embed_in"] + embed_missing) / n

def per_chunk(out_tokens):
    return cost(CHAT_MODEL, chat_in_target, out_tokens) + cost(EMBED_MODEL, embed_per_chunk_full * scale, 0)

flat = per_chunk(chat_out_per_chunk)             # output tokens unchanged by chunk length
linear = per_chunk(chat_out_per_chunk * scale)   # output tokens grow with chunk length
print()
print(f"measured per chunk ({mean_chars:.0f} chars): chat in {chat_in_per_chunk:.0f}, chat out {chat_out_per_chunk:.0f}, "
      f"embed {embed_per_chunk_full:.0f} tokens, USD {cost(CHAT_MODEL, chat_in_per_chunk, chat_out_per_chunk) + cost(EMBED_MODEL, embed_per_chunk_full, 0):.5f}")
print(f"prompt overhead per chunk (two chat calls, text removed): {overhead_in:.0f} tokens")
print(f"projected per 2000-char chunk: chat in {chat_in_target:.0f} tokens; "
      f"USD {flat:.5f} if output stays {chat_out_per_chunk:.0f} tokens, USD {linear:.5f} if output scales x{scale:.2f}")
print(f"projected per 1,000 chunks of 2,000 chars: USD {1000*flat:.2f} (flat output) to USD {1000*linear:.2f} (linear output)")
print(f"output-token share of the linear figure: {cost(CHAT_MODEL, 0, chat_out_per_chunk*scale)/linear:.0%}")

json.dump({"phases": per_phase, "total_usd": round(grand, 5),
           "projection_usd_per_1000_chunks": {"flat_output": round(1000 * flat, 2), "linear_output": round(1000 * linear, 2)}},
          open(ENV / "pgr-smoke-cost.json", "w"), indent=2)
