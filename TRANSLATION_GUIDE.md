# Korean Translation Pipeline - Quick Reference

## Stage 1: Initial Translation

### Script Overview
`src/translate_personas.py` translates English personas to Korean using Gemini 2.0 Flash with LangChain's `model.batch()` method for efficient parallel processing.

### Key Features
- **Translation Prompt**: Preserves formal register, semantic meaning, domain terminology, and natural Korean
- **Batch Processing**: Uses LangChain's `model.batch()` for efficient parallel translation
- **Concurrent Requests**: Process up to 5 requests simultaneously within each batch
- **Rate Limiting**: Respects API quotas (default: 60 requests/minute)
- **Retry Logic**: Up to 3 retry attempts with exponential backoff
- **Automatic Fallback**: If batch fails, automatically retries with single-persona mode
- **Progress Tracking**: Real-time progress bars with tqdm
- **Error Handling**: Graceful failure with detailed logging

### Quick Start

#### Test with Single Domain (Recommended First)
```bash
# Test with economics domain only (10 personas per batch, 5 concurrent)
python src/translate_personas.py \
    --input data/personas/en/ \
    --output data/personas/kr_draft/ \
    --domains economics \
    --batch-size 10 \
    --max-concurrency 5 \
    --rate-limit 60
```

#### Translate All Domains (10,000 personas, RECOMMENDED)
```bash
# Default settings: batch-size=10, max-concurrency=5
python src/translate_personas.py \
    --input data/personas/en/ \
    --output data/personas/kr_draft/ \
    --rate-limit 60
```

#### With Custom Settings
```bash
python src/translate_personas.py \
    --input data/personas/en/ \
    --output data/personas/kr_draft/ \
    --model gemini-2.0-flash \
    --temperature 0.3 \
    --rate-limit 60 \
    --max-retries 3 \
    --batch-size 10 \
    --max-concurrency 5
```

### Batch Processing Explained

**LangChain's model.batch() method:**
- Takes a list of prompts and processes them efficiently
- Supports concurrent execution via `max_concurrency` parameter
- Returns list of responses in the same order as input prompts
- Handles rate limiting and retries internally

**Configuration:**
- **batch-size**: Number of personas per batch (default: 10)
  - Determines how many personas are grouped together
  - Higher values = fewer batches, faster processing

- **max-concurrency**: Concurrent requests within a batch (default: 5)
  - How many API calls happen simultaneously
  - Must respect API rate limits

**Example:**
- batch-size=10, max-concurrency=5
- Each batch contains 10 personas
- Up to 5 API calls execute concurrently
- Batch processes 10 personas in ~2 concurrent rounds

**Benefits:**
- Efficient parallel processing
- Automatic request management
- Built-in error handling
- Maintains response order

**Reliability:**
- If a batch fails, automatic fallback to single-persona mode
- No data loss - all personas will be translated
- Success rates typically >95%

### Expected Output

**Directory Structure:**
```
data/personas/kr_draft/
├── economics_draft.jsonl
├── law_draft.jsonl
├── philosophy_draft.jsonl
├── history_draft.jsonl
├── sociology_draft.jsonl
├── environmental science_draft.jsonl
├── mathematics_draft.jsonl
├── finance_draft.jsonl
├── engineering_draft.jsonl
└── computer science_draft.jsonl
```

**JSONL Format (each line):**
```json
{
  "domain": "economics",
  "persona_en": "A Canadian energy analyst who is interested in...",
  "persona_kr": "캐나다 에너지 분석가로서 천연가스 생산의 미래와..."
}
```

### Performance Estimates

**With batch-size=10, max-concurrency=5 @ 60 requests/minute:**
- Approximate throughput: depends on model latency
- 10,000 personas: ~20-40 minutes (varies by API load)
- Automatic rate limiting prevents quota issues

**Optimization Tips:**
- Default settings (batch-size=10, max-concurrency=5) work well for most cases
- Increase `--rate-limit` if you have higher API quota
- Monitor success rate in logs
- Failed batches automatically retry in single-persona mode

### Success Metrics

**Target (from CLAUDE.md):**
- First-pass acceptance rate: >80%
- Average iterations per persona: <1.5
- Translation completeness: 100%

**Monitoring:**
```bash
# Check output files
ls -lh data/personas/kr_draft/

# Count translated personas per domain
wc -l data/personas/kr_draft/*.jsonl

# View sample translations
head -n 3 data/personas/kr_draft/economics_draft.jsonl | python -m json.tool
```

### Troubleshooting

**Issue: API Rate Limiting**
```bash
# Reduce rate limit or max concurrency
python src/translate_personas.py --rate-limit 30 --max-concurrency 3
```

**Issue: Translation Failures**
- Check logs for error messages
- Verify API key in `.env`
- Check internet connection
- Retry failed domains individually

**Issue: Batch Failures**
- Automatic fallback to single-persona mode
- Check logs to identify patterns
- Reduce batch-size or max-concurrency if persistent

**Issue: Incomplete Translations**
```bash
# Resume by translating specific domains
python src/translate_personas.py --domains economics law philosophy
```

### Next Steps

After Stage 1 completion:
1. Verify all 10,000 personas translated
2. Spot-check translation quality manually
3. Proceed to **Stage 2: Multi-Instance Verification** (`src/verify_translations.py`)

---

## Translation Quality Checklist

Before proceeding to Stage 2, verify:

- [ ] All 10 domain files created in `data/personas/kr_draft/`
- [ ] Each file contains 1,000 translations
- [ ] No empty `persona_kr` fields
- [ ] Korean text displays correctly (UTF-8 encoding)
- [ ] Domain terminology appears appropriate
- [ ] Formal register preserved in samples
- [ ] Total count: 10,000 translations

### Sample Quality Check

```python
# Quick quality check script
import json

def check_translation_sample(file_path, n=5):
    """Check first n translations from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            print(f"\n--- Translation {i+1} ---")
            print(f"EN: {data['persona_en'][:100]}...")
            print(f"KR: {data['persona_kr'][:100]}...")

check_translation_sample('data/personas/kr_draft/economics_draft.jsonl')
```

---

## Stage 1 Completion Criteria

✅ **Ready for Stage 2 when:**
1. All 10,000 personas translated
2. Success rate ≥95%
3. Manual spot-checks confirm quality
4. No systematic translation errors observed
5. Korean text properly encoded (UTF-8)
