# 🚀 How to Deploy - Quick Reference

## Choose Your Path

### 🏆 Easiest: Vercel + Render (30 min, $21/mo)
```bash
# 1. Create accounts
https://vercel.com & https://render.com

# 2. Deploy services on Render
- PostgreSQL database
- Redis cache
- FastAPI backend

# 3. Deploy frontend on Vercel
vercel --prod

# See: docs/QUICK_DEPLOY.md
```

### 🐳 Full Control: Docker on VPS (1 hour, $12/mo)
```bash
# 1. Get VPS (DigitalOcean, Linode, etc.)
# 2. Install Docker
curl -fsSL https://get.docker.com | sh

# 3. Deploy
git clone <repo>
docker-compose -f docker-compose.prod.yml up -d

# See: docs/DEPLOYMENT_GUIDE.md
```

### ☁️ Enterprise: AWS/GCP (4+ hours, $50+/mo)
```bash
# Complex setup, see full guide
# docs/DEPLOYMENT_GUIDE.md
```

## 📚 Full Documentation

- **Quick Start:** [docs/QUICK_DEPLOY.md](docs/QUICK_DEPLOY.md)
- **Complete Guide:** [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **Docker Compose:** [docker-compose.prod.yml](docker-compose.prod.yml)
- **Helper Script:** [scripts/deploy-render.sh](scripts/deploy-render.sh)

## 🔐 Security Reminder

Before deploying:
- [ ] Change database password
- [ ] Generate new SECRET_KEY
- [ ] Set DEBUG=false
- [ ] Configure proper CORS

## 💰 Cost Estimates

| Platform | Small | Medium | Large |
|----------|-------|--------|-------|
| Vercel+Render | $21/mo | $96/mo | - |
| DigitalOcean | $12/mo | $50/mo | $100+/mo |
| AWS | $50/mo | $100/mo | $400+/mo |

---
**Recommendation:** Start with Vercel + Render 🏆
