# 🚀 Quick Start Guide

## Get Running in 5 Minutes

### Step 1: Install Dependencies (1 min)

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Set Up API Key (1 min)

Create `.env` file:
```bash
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
```

### Step 3: Test the System (1 min)

```bash
python test_system.py
```

You should see:
- ✅ Guardrails tests passing
- ✅ Classifier routing queries correctly
- ✅ Agents responding with sample data

### Step 4: Run the Server (1 min)

```bash
python main.py
```

Visit: **http://localhost:8000/docs**

### Step 5: Try It Out (1 min)

In the interactive docs, try the `/chat` endpoint:

```json
{
  "message": "What's your work experience?"
}
```

## Next Steps

### Customize Your Data

Replace sample data with YOUR information:

1. **data/resume.json** - Your work history, education, skills
2. **data/projects.json** - Your GitHub projects and applications  
3. **data/publications.json** - Your research papers
4. **data/hobbies.json** - Your personal interests

### Test Different Queries

Try these to test each agent:

- **General**: "What companies have you worked for?"
- **Projects**: "What projects have you built?"
- **Publications**: "Tell me about your research"
- **Calendar**: "Can we schedule a meeting?"
- **Hobbies**: "What do you do for fun?"
- **Else**: "What's the weather?" (should politely decline)

### Deploy to Production

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin your-repo-url
   git push -u origin main
   ```

2. **Deploy to Render**
   - Go to [render.com](https://render.com)
   - New Web Service → Connect your repo
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Add env var: `OPENAI_API_KEY`
   - Deploy!

3. **Build React Frontend**
   - Share your UI mockup
   - I'll help build the React components
   - Deploy to Vercel
   - Connect to Render backend

## Troubleshooting

**Server won't start?**
- Check `.env` file exists and has valid API key
- Ensure all dependencies installed: `pip list`

**Agents give generic responses?**
- Update `data/*.json` files with your real information
- Sample data is just placeholders

**API errors?**
- Check OpenAI API key is valid
- Ensure you have API credits

**Need help?**
- Check `PROJECT_OVERVIEW.md` for detailed docs
- Review `README.md` for deployment guide
- Run `test_system.py` to diagnose issues

## What's Included

✅ Complete FastAPI backend  
✅ 5 specialized agents  
✅ Guardrails (moderation, PII, jailbreak)  
✅ Classification routing  
✅ Conversation memory  
✅ Session management  
✅ Sample data files  
✅ Test suite  
✅ Full documentation  

Ready to customize and deploy! 🎉
