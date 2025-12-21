# Digital Avatar Frontend

React-based chat interface for Sean's Digital Avatar.

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure API URL

Create a `.env` file:

```bash
cp .env.example .env
```

For local development, it's already set to `http://localhost:8000`.

### 3. Start Development Server

```bash
npm start
```

The app will open at `http://localhost:3000`

**Make sure your backend is running first:**
```bash
# In a separate terminal
cd ../backend
conda activate digital_avatar
python main.py
```

## Features

- ✅ Clean, minimal chat interface
- ✅ Real-time messaging
- ✅ Session persistence
- ✅ Typing indicators
- ✅ Category-based routing (transparent to user)
- ✅ Responsive design
- ✅ Error handling

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.jsx          # Main chat component
│   ├── App.css          # Styles
│   ├── index.js         # Entry point
│   └── index.css        # Global styles
├── package.json
├── .env.example
└── README.md
```

## Deployment to Vercel

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial frontend commit"
git remote add origin your-repo-url
git push -u origin main
```

### 2. Deploy on Vercel

1. Go to [vercel.com](https://vercel.com)
2. Import your repository
3. **Framework Preset:** Create React App
4. **Root Directory:** `frontend`
5. **Environment Variables:**
   - Key: `REACT_APP_API_URL`
   - Value: `https://your-backend.onrender.com` (your Render backend URL)
6. Deploy!

### 3. Update Backend CORS

After deploying, update the CORS settings in your backend's `main.py`:

```python
allow_origins=["https://your-app.vercel.app"]
```

## Available Scripts

- `npm start` - Run development server
- `npm build` - Build for production
- `npm test` - Run tests

## Customization

### Change Colors

Edit `App.css`:
- Background: `.app { background-color: ... }`
- User messages: `.message.user .message-content { background-color: ... }`
- Send button: `.send-button { background-color: ... }`

### Modify Welcome Message

Edit `App.jsx`:
```jsx
<h1>Hi, I am Sean's Digital Avatar, how can I help you today?</h1>
```

### Add Features

The component is well-structured for adding:
- File uploads
- Voice input
- Message reactions
- Export conversations
- Dark mode

## Troubleshooting

**CORS errors?**
- Make sure backend allows your frontend origin
- Check `main.py` CORS settings

**Can't connect to backend?**
- Verify backend is running
- Check `.env` file has correct API_URL
- Try `http://localhost:8000/health` in browser

**Build fails?**
- Delete `node_modules` and run `npm install` again
- Clear cache: `npm cache clean --force`

## License

MIT
