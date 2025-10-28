# Qwen 2.5-VL Setup & Troubleshooting Guide

## ✅ Current Status

Your research site is now running with Qwen2.5-VL-32B-Instruct working correctly!

- **Server Status**: Running on port 80
- **Model**: Qwen/Qwen2.5-VL-32B-Instruct (32B parameters)
- **GPU Memory**: ~62GB allocated on NVIDIA A100 80GB
- **Workers**: 1 worker process (optimized for GPU memory)
- **Access**: http://localhost:80 or your RunPod proxy URL

## 🔧 What Was Fixed

### Issue
The Flask/Gunicorn application wasn't running, so no responses were being generated.

### Solution
1. Started the Flask application using `start.sh`
2. Optimized Gunicorn configuration to use only 1 worker (instead of 4)
   - Reason: The 32B qwen model uses ~62GB GPU memory
   - Multiple workers would try to load the model and exceed 80GB GPU limit
3. Increased thread count from 2 to 4 to handle concurrent requests

## 🚀 Starting the Server

To start the server manually:

```bash
cd /researchsite
bash start.sh
```

The server will:
- Create necessary directories
- Set up the virtual environment (if needed)
- Start Gunicorn with the optimized configuration
- Listen on port 80

## 🔄 Checking Server Status

Check if the server is running:
```bash
ps aux | grep gunicorn | grep -v grep
```

Check health endpoint:
```bash
curl http://localhost:80/health
```

Check logs:
```bash
tail -f /var/log/researchsite/access.log
tail -f /var/log/researchsite/error.log
```

## 🛑 Stopping the Server

```bash
pkill -f "gunicorn.*app:app"
```

Or more forcefully:
```bash
pkill -9 -f gunicorn
```

## 🎯 Using Qwen on the Website

1. Open the website at your RunPod proxy URL
2. In the model selector dropdown (top right), select **"Qwen2.5-VL-32B"**
3. Type your question and send
4. The first request after starting the server will take ~25 seconds (model loading)
5. Subsequent requests are much faster (~1-2 seconds for simple queries)

## 📊 Performance Notes

- **First Request**: 20-30 seconds (model loads into GPU memory)
- **Subsequent Requests**: 1-5 seconds depending on complexity
- **GPU Memory**: 62.4 GB allocated, 62.7 GB reserved
- **Max Tokens**: Currently set to 256 for faster responses
- **Context Window**: Can handle long context (32K tokens)

## ⚙️ Configuration Files

- **gunicorn.conf.py**: Server configuration (workers, threads, timeouts)
- **research_llm.py**: Qwen model implementation
- **langchain_service.py**: Model integration with RAG system
- **.env**: Environment variables (HF_TOKEN, model names)

## 🐛 Troubleshooting

### Server Not Responding
```bash
# Check if server is running
ps aux | grep gunicorn

# Check logs
tail -50 /var/log/researchsite/error.log

# Restart server
pkill -f gunicorn && cd /researchsite && bash start.sh
```

### "CUDA out of memory" Error
- Make sure only 1 Gunicorn worker is configured (check gunicorn.conf.py)
- Restart the server to clear GPU memory
- Check GPU usage: `nvidia-smi`

### Model Not Loading
```bash
# Test model loading directly
python3 -c "from langchain_service import get_qwenvl; llm = get_qwenvl(); print('OK')"
```

### Slow Responses
- First request is always slow (model loading)
- Check GPU memory usage: `nvidia-smi`
- Increase max_tokens in research_llm.py if needed
- Make sure you're using GPU (check health endpoint)

## 🔐 Environment Variables

Required in `.env`:
```bash
HF_TOKEN=your_huggingface_token_here
QWENVL_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
ANTHROPIC_API_KEY=your_anthropic_key  # For Claude fallback
```

## 📝 Model Comparison

Your site supports two models:

| Feature | Anthropic Claude | Qwen2.5-VL-32B |
|---------|-----------------|----------------|
| Speed | Fast (API) | Slower (local) |
| Cost | Per-token cost | Free (self-hosted) |
| Context | 200K tokens | 32K tokens |
| Vision | Limited | Strong |
| Code | Excellent | Good |
| Running | Cloud API | Local GPU |

## 💡 Tips

1. **Keep server running**: Use a process manager like systemd or supervisor for production
2. **Monitor GPU**: Use `watch -n 1 nvidia-smi` to monitor GPU usage
3. **Logs**: Check `/var/log/researchsite/` for debugging
4. **Test endpoint**: `curl http://localhost:80/health` for quick status check
5. **Switch models**: Use the dropdown in the UI to switch between Claude and Qwen

## 🎉 Success!

You should now be able to:
- ✅ Access the website
- ✅ Select Qwen2.5-VL-32B from the dropdown
- ✅ Get responses from the local qwen model
- ✅ Switch between Claude and Qwen as needed

Enjoy your self-hosted AI research assistant! 🚀
