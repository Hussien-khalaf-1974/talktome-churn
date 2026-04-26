# 📡 Talk to Me — Churn Prediction App

## Platform Recommendation: Render.com ✅

**Why Render over Vercel:**

| Factor | Render | Vercel |
|---|---|---|
| Python/Flask support | ✅ Full native support | ❌ Only via serverless functions (very limited) |
| .pkl model files | ✅ Persistent disk, loads normally | ⚠️ Serverless size limits cause issues |
| Free tier | ✅ Free web service (sleeps after 15min idle) | ⚠️ Free tier doesn't support Python servers well |
| ML libraries | ✅ No restrictions | ❌ 50MB bundle size limit breaks sklearn/xgboost |
| Setup difficulty | ⭐ Very beginner-friendly | Requires extra configuration for Python |

**Verdict: Use Render.** Vercel is excellent for JavaScript/Next.js apps, but for a Python ML app with large model files, Render is the right tool.

---

## Project Structure

```
talktome/
├── app.py                    ← Flask backend (main file)
├── requirements.txt          ← Python dependencies
├── render.yaml               ← Render deployment config
├── templates/
│   └── index.html            ← Full UI (dashboard + form + batch)
└── models/                   ← PUT YOUR MODEL FILES HERE
    ├── tuned_churn_model.pkl
    ├── churn_scaler.pkl
    ├── selected_features.pkl
    └── model_config.json
```

---

## Step-by-Step Deployment on Render

### STEP 1 — Prepare your files locally

1. Download this entire `talktome/` folder to your computer.
2. Create a folder called `models/` inside it.
3. Copy your four model files into `models/`:
   - `tuned_churn_model.pkl`
   - `churn_scaler.pkl`
   - `selected_features.pkl`
   - `model_config.json`

Your folder should now look like the structure above.

---

### STEP 2 — Create a GitHub repository

Render deploys directly from GitHub, so you need to push your code there.

1. Go to https://github.com and sign in (or create a free account).
2. Click the **+** button (top right) → **New repository**.
3. Name it: `talktome-churn`
4. Set it to **Private** (your model files will be inside).
5. Click **Create repository**.
6. Follow the instructions GitHub shows to push your local folder:

```bash
# Open your terminal / command prompt in the talktome/ folder

git init
git add .
git commit -m "Initial commit — Talk to Me churn app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/talktome-churn.git
git push -u origin main
```

> If you don't have Git installed: download it from https://git-scm.com

---

### STEP 3 — Create a Render account

1. Go to https://render.com
2. Click **Get Started** (top right).
3. Click **Continue with GitHub** — this links Render to your GitHub account.
4. Authorise Render to access your repositories.

---

### STEP 4 — Deploy on Render

1. From your Render dashboard, click **New +** → **Web Service**.
2. Click **Connect a repository**.
3. Find `talktome-churn` in the list and click **Connect**.
4. Fill in the deployment form:

| Field | Value |
|---|---|
| **Name** | `talktome-churn` (or any name you like) |
| **Region** | Choose the closest to you |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120` |
| **Instance Type** | `Free` (to start) |

5. Click **Create Web Service**.
6. Render will now install your dependencies and start the app.
   This takes about **3–5 minutes** on first deploy.

---

### STEP 5 — View your live app

1. Once the deploy shows **Live** (green), click the URL at the top of the page.
   It will look like: `https://talktome-churn.onrender.com`
2. Your app is now live! Share this URL with anyone.

---

### STEP 6 — Updating the app

Whenever you change any file:

```bash
git add .
git commit -m "Update: describe what you changed"
git push
```

Render automatically detects the push and redeploys — no manual action needed.

---

## Important notes

### Free tier behaviour
On Render's free tier, the app **goes to sleep after 15 minutes of inactivity**.
The first visit after sleep takes ~30 seconds to wake up. This is normal.
To keep it always awake, upgrade to the **Starter plan** ($7/month).

### Model file size
If your `.pkl` files are very large (>500MB total), you may need to use
Render's **Persistent Disk** feature or store models in a cloud bucket (S3/GCS).
For typical sklearn/xgboost models under 100MB, the approach above works fine.

### Environment variables (optional)
If you want to change the threshold without redeploying, add an environment
variable in Render dashboard → Environment → Add variable:
- Key: `THRESHOLD`
- Value: `0.40`
Then update `app.py` to read: `THRESHOLD = float(os.environ.get("THRESHOLD", config.get("optimal_threshold", 0.40)))`

---

## Local development

To run the app on your own computer before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open http://localhost:5000 in your browser.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "ModuleNotFoundError" on deploy | Check `requirements.txt` has all packages listed |
| Model not loading | Make sure `models/` folder and all 4 files are committed to GitHub |
| App shows error on prediction | Check Render logs: Dashboard → your service → Logs tab |
| CSV upload fails | Make sure CSV has the same column names as training data |
| App is slow on first load | Normal on free tier — it was sleeping. Wait 30 seconds. |
