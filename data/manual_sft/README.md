# Manual SFT Pairs

Create your 25-50 handcrafted SFT (Supervised Fine-Tuning) pairs here.

## Format

Each line should be a JSON object with these fields:

```json
{
  "context": [
    {"role": "user", "content": "user message here"},
    {"role": "assistant", "content": "assistant response here"},
    {"role": "user", "content": "follow-up message"}
  ],
  "scratchpad": "Any scratchpad notes or context",
  "instruction": "Provide a software design suggestion based on the given context.",
  "output": "Your concrete, actionable design suggestion"
}
```

### Context Format

The `context` field is an array of message objects, each with:
- `role`: Either `"user"` or `"assistant"` (required)
- `content`: The message text (required)

## Tips

- Keep `context` realistic - short chat exchanges (2-4 messages)
- `scratchpad` can be empty string `""` if not needed
- `instruction` can be the same for all pairs, or vary it slightly
- `output` should be specific and actionable (1-3 sentences)

## Example

```json
{
  "context": [
    {"role": "user", "content": "I need to add real-time updates to my app."},
    {"role": "assistant", "content": "What kind of updates?"},
    {"role": "user", "content": "Like notifications when new data arrives."}
  ],
  "scratchpad": "Need low latency, handle 10k concurrent users",
  "instruction": "Provide a software design suggestion based on the given context.",
  "output": "Use WebSocket connections with a message queue (Redis) for pub/sub distribution. Implement connection pooling to handle scale."
}
```

## Files

Create as many `.jsonl` files as you want (e.g., `batch1.jsonl`, `batch2.jsonl`). The `prepare_dataset.py` script will merge them all.
