# Storage Requirements for Research Buddy

## Overview
This document outlines storage requirements for running the Research Buddy application with WizardLM model and document storage.

## Storage Breakdown

### 1. AI Models
| Component | Size | Location | Purpose |
|-----------|------|----------|---------|
| **WizardLM-13B-Uncensored** | ~26 GB | `/tmp/hf_cache` or `~/.cache/huggingface` | Main LLM for text generation |
| **sentence-transformers/all-MiniLM-L6-v2** | ~90 MB | `/tmp/hf_cache` | Embedding model for RAG |
| **PyTorch & Dependencies** | ~5 GB | `.venv/lib` | ML framework |
| **TOTAL MODELS** | **~31 GB** | | |

### 2. Application & Dependencies
| Component | Size | Location |
|-----------|------|----------|
| Python Virtual Environment | ~2-3 GB | `/researchsite/.venv` |
| Application Code | ~50 MB | `/researchsite` |
| Logs | ~100-500 MB | `/var/log/researchsite` |
| **TOTAL APP** | **~3.5 GB** | |

### 3. User Data (Variable)
| Component | Est. Size per File | Location | Notes |
|-----------|-------------------|----------|-------|
| **PDF Documents** | 1-10 MB avg | `/researchsite/data/uploads` | Depends on page count |
| **Images** | 0.5-5 MB avg | `/researchsite/data/uploads` | Depends on resolution |
| **Vector Database (ChromaDB)** | 10-50% of PDF size | `/researchsite/data/vector_store` | Embeddings + metadata |
| **Temporary Files** | Variable | `/tmp` | Cleaned periodically |

### 4. Storage Recommendations

#### Minimum Configuration
```
Total: 50 GB
- 31 GB: Models
- 4 GB: Application
- 15 GB: User data + buffer
```
**Use case**: Testing, small document library (< 100 PDFs)

#### Recommended Configuration
```
Total: 100 GB
- 31 GB: Models
- 4 GB: Application  
- 65 GB: User data (500-1000 PDFs)
```
**Use case**: Production, moderate document library

#### Heavy Usage Configuration
```
Total: 200+ GB
- 31 GB: Models
- 4 GB: Application
- 165+ GB: Large document library (2000+ PDFs)
```
**Use case**: Enterprise, large document repositories

## Storage Optimization Tips

### 1. Model Caching
The app is configured to cache models in `/tmp/hf_cache` which is typically cleared on reboot. For persistent storage:

```bash
# Set permanent cache location
export HF_HOME=/researchsite/models_cache
export TRANSFORMERS_CACHE=/researchsite/models_cache
```

Add to your `.env` file or systemd service.

### 2. PDF Storage Optimization
- **Compress PDFs**: Use tools like `ghostscript` to compress before upload
- **OCR on demand**: Only OCR pages when queried (current implementation)
- **Archive old documents**: Move unused PDFs to cheaper storage

### 3. Vector Database Management
```python
# Periodic cleanup of old embeddings
cd /researchsite
python -c "from rag_service import get_rag_service; get_rag_service().cleanup_old_embeddings(days=90)"
```

### 4. Log Rotation
Add to `/etc/logrotate.d/researchsite`:
```
/var/log/researchsite/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 root root
}
```

## Storage by Use Case

### Small Research Team (5-10 users)
- **Storage**: 100 GB
- **Documents**: ~500 PDFs (5 GB)
- **Monthly Growth**: ~5-10 GB
- **Duration**: 6-12 months before expansion

### Medium Organization (20-50 users)
- **Storage**: 250 GB
- **Documents**: ~2000 PDFs (20 GB)
- **Monthly Growth**: ~20-30 GB
- **Duration**: 8-12 months before expansion

### Large Enterprise (100+ users)
- **Storage**: 500 GB - 1 TB
- **Documents**: 5000+ PDFs (50+ GB)
- **Monthly Growth**: 50-100 GB
- **Consider**: Object storage (S3, MinIO) for PDFs

## Cloud Storage Recommendations

### RunPod / GPU Cloud
- **Pod Storage**: 50-100 GB (sufficient for models + cache)
- **Network Volume**: 200-500 GB (persistent user data)
- **Snapshot regularly**: Models can be re-downloaded but user data is irreplaceable

### AWS / Azure / GCP
- **Compute Instance**: 100 GB SSD (app + models)
- **Object Storage**: S3/Blob/GCS for PDFs (unlimited, pay per use)
- **Database**: Managed vector DB (Pinecone, Weaviate) for scale

## Monitoring Storage Usage

### Check Current Usage
```bash
# Overall disk usage
df -h

# Application data
du -sh /researchsite/data/*

# Model cache
du -sh /tmp/hf_cache ~/.cache/huggingface

# Vector database
du -sh /researchsite/data/vector_store
```

### Set Up Alerts
Add to your monitoring:
```bash
# Alert when data directory exceeds 80%
if [ $(du -s /researchsite/data | awk '{print $1}') -gt 85899345920 ]; then
    echo "WARNING: Data directory > 80GB"
fi
```

## Cost Estimates (Monthly)

### Cloud Storage Costs
| Provider | Storage Type | 100 GB/month | 500 GB/month |
|----------|-------------|--------------|--------------|
| AWS | EBS SSD | ~$10 | ~$50 |
| AWS | S3 Standard | ~$2.30 | ~$11.50 |
| Azure | Managed Disk | ~$10 | ~$50 |
| Azure | Blob Storage | ~$2 | ~$10 |
| GCP | Persistent SSD | ~$17 | ~$85 |
| GCP | Cloud Storage | ~$2 | ~$10 |

### Recommendations by Budget
- **Budget (<$20/month)**: 100 GB local storage + periodic backups
- **Standard ($20-50/month)**: 200 GB local + S3 for archives
- **Enterprise ($50+/month)**: Fast SSD for models + Object storage for data

## FAQ

### Q: Can I use slower storage for models?
**A**: Not recommended. Model loading requires fast I/O. Use SSD for models, HDD acceptable for archived PDFs.

### Q: How much RAM do I need?
**A**: 
- WizardLM-13B (quantized): 16-24 GB RAM
- WizardLM-13B (full): 32-64 GB RAM
- Embedding model: 2-4 GB RAM
- **Recommended minimum**: 32 GB RAM

### Q: Can I reduce model storage?
**A**: Yes, use quantized models:
- 4-bit quantization: ~7 GB (recommended)
- 8-bit quantization: ~13 GB
- Full model: ~26 GB

To use quantized model, update `.env`:
```bash
WIZARDLM_QUANTIZATION=4bit
```

### Q: What happens if I run out of space?
**A**: 
1. Model downloads will fail (use `/tmp` to avoid)
2. PDF uploads will be rejected
3. Vector DB writes will fail
4. Application may crash

**Prevention**: Monitor storage and set alerts at 80% capacity.

## Summary

| Configuration | Storage | RAM | Use Case |
|---------------|---------|-----|----------|
| **Minimal** | 50 GB | 16 GB | Testing, dev |
| **Recommended** | 100 GB | 32 GB | Small teams |
| **Production** | 200+ GB | 64 GB | Organizations |
| **Enterprise** | 500+ GB | 128 GB | Large scale |

**Our current setup**: Models in `/tmp` (saved on disk space) + persistent data in `/researchsite/data`

---

*Last updated: Based on WizardLM-13B-Uncensored and current architecture*