# ⚡ Quick Deploy Guide

Choose your deployment method below.

## 🏆 Recommended: Vercel + Render (30 minutes)

**Perfect for:** Most users, easiest setup, production-ready

### Cost: ~$21-35/month
### Difficulty: ⭐ Easy

```bash
# 1. Create accounts (free tier available)
https://vercel.com/signup
https://render.com/register

# 2. Deploy Database on Render
- Go to https://dashboard.render.com
- Click "New +" → PostgreSQL
- Name: fake-news-db
- Plan: Free or Starter ($7/mo)
- Click Create

# 3. Deploy Redis on Render
- Click "New +" → Redis
- Name: fake-news-redis
- Plan: Free or Starter ($7/mo)
- Click Create

# 4. Deploy Backend on Render
- Click "New +" → Web Service
- Connect GitHub repo
- Root Directory: backend
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
- Add environment variables (see below)

# 5. Deploy Frontend on Vercel
cd frontend
vercel --prod
# Or use Vercel dashboard to import from GitHub

# Done! ✅
```

### Environment Variables for Render Backend:

```bash
APP_NAME=Fake News Detection
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=<run: openssl rand -hex 32>
DATABASE_URL=<from Render PostgreSQL Internal URL>
REDIS_URL=<from Render Redis URL>
CORS_ORIGINS=["https://your-app.vercel.app"]
```

### Environment Variables for Vercel Frontend:

```bash
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
NEXT_PUBLIC_ENVIRONMENT=production
```

---

## 🐳 Docker on DigitalOcean (1 hour)

**Perfect for:** Full control, cost-effective at scale

### Cost: ~$12/month (Basic Droplet)
### Difficulty: ⭐⭐ Moderate

```bash
# 1. Create DigitalOcean Droplet
- OS: Ubuntu 22.04 LTS
- Plan: Basic ($12/mo - 2GB RAM)
- Add SSH key

# 2. SSH into droplet
ssh root@your-droplet-ip

# 3. Install Docker
curl -fsSL https://get.docker.com | sh
apt install docker-compose -y

# 4. Clone repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# 5. Configure environment
cp backend/.env.example backend/.env
nano backend/.env  # Add your credentials

# 6. Deploy
docker-compose -f docker-compose.prod.yml up -d

# 7. Set up domain & SSL (optional)
apt install nginx certbot python3-certbot-nginx
certbot --nginx -d yourdomain.com

# Done! ✅
```

---

## ☁️ AWS (Advanced)

**Perfect for:** Enterprise, high traffic, custom requirements

### Cost: ~$50-200/month
### Difficulty: ⭐⭐⭐ Advanced

1. **Frontend:** Deploy to AWS Amplify or S3 + CloudFront
2. **Backend:** ECS with Fargate or EC2 with Auto Scaling
3. **Database:** RDS PostgreSQL
4. **Cache:** ElastiCache Redis

See [AWS Deployment Guide](./AWS_DEPLOYMENT.md) for details.

---

## 🎯 Which Should You Choose?

| Option | Best For | Cost/mo | Setup Time | Scaling |
|--------|----------|---------|------------|---------|
| **Vercel + Render** | Most users, MVP | $21-35 | 30 min | Auto |
| **DigitalOcean** | Cost-conscious, learning | $12-50 | 1 hour | Manual |
| **AWS** | Enterprise, high traffic | $50-200+ | 4+ hours | Auto |

---

## 📋 Pre-Deployment Checklist

Before deploying, make sure you have:

- [ ] Changed database password from default
- [ ] Generated new SECRET_KEY (`openssl rand -hex 32`)
- [ ] Set DEBUG=false in production
- [ ] Configured CORS with your actual domain
- [ ] Uploaded ML models to backend/models/
- [ ] Set up domain name (optional but recommended)
- [ ] Created backup strategy
- [ ] Set up monitoring (Sentry, UptimeRobot)

---

## 🚨 Important Security Notes

```bash
# ALWAYS change these before deploying:

1. Database password (not default!)
2. SECRET_KEY (generate new: openssl rand -hex 32)
3. CORS_ORIGINS (your actual domain, not *)
4. DEBUG=false (never true in production)
5. HTTPS enabled (Let's Encrypt is free)
```

---

## 🔍 Test Your Deployment

After deployment, test these:

```bash
# 1. Backend health check
curl https://your-backend.com/health

# 2. Frontend loads
curl https://your-frontend.com

# 3. API endpoint
curl https://your-backend.com/api/v1/health

# 4. Database connection
# Check backend logs for successful DB connection

# 5. Run a classification
# Use the UI to classify some text
```

---

## 📚 More Resources

- **Full Guide:** [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Security:** [SECURITY.md](./SECURITY.md)
- **Monitoring:** [MONITORING.md](./MONITORING.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

## 💬 Need Help?

1. Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed steps
2. See [Troubleshooting section](#troubleshooting)
3. Open an issue on GitHub
4. Contact the team

---

**Last Updated:** October 4, 2025
