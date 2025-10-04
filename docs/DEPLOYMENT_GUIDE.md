# 🚀 Production Deployment Guide

Complete guide for deploying the Fake News Detection Platform to production.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Recommended Stack](#recommended-stack)
3. [Step-by-Step Deployment](#step-by-step-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Security Checklist](#security-checklist)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Deployment Options

### Option 1: Cloud Platform (Recommended) ⭐

**Best for:** Production-ready, scalable deployments

| Platform | Best For | Cost | Difficulty |
|----------|----------|------|------------|
| **Vercel + Railway** | Quick deployment, auto-scaling | $20-50/mo | ⭐ Easy |
| **AWS (ECS/EC2)** | Full control, enterprise | $50-200/mo | ⭐⭐⭐ Advanced |
| **DigitalOcean** | Balance of simplicity & control | $30-100/mo | ⭐⭐ Moderate |
| **Google Cloud Run** | Serverless, pay-per-use | $20-80/mo | ⭐⭐ Moderate |
| **Render** | Simple, unified platform | $25-75/mo | ⭐ Easy |

### Option 2: Self-Hosted (VPS)

**Best for:** Cost-effective, full control

- DigitalOcean Droplet
- Linode
- Vultr
- Hetzner

### Option 3: Hybrid Approach

- **Frontend:** Vercel (CDN, auto-scaling)
- **Backend:** Railway/Render (managed services)
- **Database:** Managed PostgreSQL (AWS RDS, DigitalOcean Managed DB)

---

## Recommended Stack (Production)

### 🏆 **Easiest & Fastest: Vercel + Render**

**Total Setup Time:** ~30 minutes  
**Monthly Cost:** ~$25-50  
**Scalability:** Excellent

```
Frontend (Next.js)  →  Vercel
Backend (FastAPI)   →  Render
Database            →  Render PostgreSQL
Redis               →  Render Redis
```

#### Why This Stack?

✅ Zero DevOps required  
✅ Automatic HTTPS/SSL  
✅ Auto-scaling built-in  
✅ GitHub integration  
✅ Easy rollbacks  
✅ Free SSL certificates  

---

## Step-by-Step Deployment

### 🚀 Quick Deploy (Vercel + Render)

#### Prerequisites

```bash
# 1. Create accounts (free tier available)
- Vercel account: https://vercel.com
- Render account: https://render.com

# 2. Install CLIs (optional)
npm i -g vercel
```

#### Step 1: Deploy Database (Render)

1. **Go to Render Dashboard** → https://dashboard.render.com
2. **Click "New +"** → Select "PostgreSQL"
3. **Configure:**
   ```
   Name: fake-news-db
   Database: gtds_db
   User: postgres
   Region: Oregon (or closest to you)
   Plan: Starter ($7/mo) or Free
   ```
4. **Click "Create Database"**
5. **Copy connection details:**
   - Internal Database URL
   - External Database URL
   - PSQL Command

#### Step 2: Deploy Redis (Render)

1. **Click "New +"** → Select "Redis"
2. **Configure:**
   ```
   Name: fake-news-redis
   Region: Oregon (same as DB)
   Plan: Starter ($7/mo) or Free
   ```
3. **Copy Redis URL**

#### Step 3: Deploy Backend (Render)

1. **Click "New +"** → Select "Web Service"
2. **Connect GitHub repository**
3. **Configure:**
   ```yaml
   Name: fake-news-backend
   Region: Oregon
   Branch: main
   Root Directory: backend
   Runtime: Python 3.11
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   Plan: Starter ($7/mo) or Free
   ```

4. **Add Environment Variables:**
   ```bash
   # Application
   APP_NAME=GTDS Fake News API
   ENVIRONMENT=production
   DEBUG=false
   
   # Database (use Internal URL from Step 1)
   DATABASE_URL=postgresql://user:pass@host/db
   
   # Redis (from Step 2)
   REDIS_URL=redis://...
   
   # Security (GENERATE NEW KEYS!)
   SECRET_KEY=<generate-with-openssl-rand-hex-32>
   
   # CORS (add your frontend domain)
   CORS_ORIGINS=["https://your-app.vercel.app"]
   
   # ML Models
   ML_MODEL_PATH=./backend/models/
   ```

5. **Click "Create Web Service"**
6. **Wait for build (~5-10 minutes)**
7. **Copy backend URL:** `https://fake-news-backend.onrender.com`

#### Step 4: Deploy Frontend (Vercel)

```bash
# Option A: Using Vercel CLI
cd frontend
vercel

# Follow prompts:
# - Link to existing project? No
# - Project name? fake-news-detection
# - Directory? ./
# - Override settings? No

# Option B: Using Vercel Dashboard
```

1. **Go to Vercel Dashboard** → https://vercel.com/dashboard
2. **Click "Add New"** → "Project"
3. **Import Git Repository**
4. **Configure:**
   ```
   Framework Preset: Next.js
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: .next
   ```

5. **Environment Variables:**
   ```bash
   NEXT_PUBLIC_API_URL=https://fake-news-backend.onrender.com
   NEXT_PUBLIC_APP_NAME=Fake News Detection
   NEXT_PUBLIC_ENVIRONMENT=production
   ```

6. **Click "Deploy"**
7. **Wait for deployment (~3-5 minutes)**
8. **Get your URL:** `https://fake-news-detection.vercel.app`

#### Step 5: Initialize Database

```bash
# Connect to your Render PostgreSQL
psql <EXTERNAL_DATABASE_URL_FROM_RENDER>

# Run migrations (if you have any)
# Or use the init script
\i backend/scripts/init-db.sql
```

#### Step 6: Upload ML Models

```bash
# Option 1: Include in repository (if < 100MB)
git lfs track "backend/models/*.joblib"
git add backend/models/
git commit -m "Add ML models"
git push

# Option 2: Upload to Render manually
# SSH into Render service or use their file upload

# Option 3: Use cloud storage (S3/GCS)
# Update ML_MODEL_PATH to point to bucket
```

---

## Alternative: Docker + DigitalOcean

### Step 1: Create DigitalOcean Droplet

```bash
# 1. Go to DigitalOcean
# 2. Create Droplet:
#    - Ubuntu 22.04 LTS
#    - Basic Plan: $12/mo (2GB RAM, 1 vCPU)
#    - Datacenter: Closest to your users
#    - Add SSH key

# 3. SSH into droplet
ssh root@your-droplet-ip
```

### Step 2: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify
docker --version
docker-compose --version
```

### Step 3: Clone & Configure

```bash
# Clone repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# Create production .env
cp backend/.env.example backend/.env
nano backend/.env  # Edit with production values

# Create frontend .env
echo "NEXT_PUBLIC_API_URL=https://api.yourdomain.com" > frontend/.env.production
```

### Step 4: Deploy with Docker Compose

```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose logs -f

# Check status
docker-compose ps
```

### Step 5: Configure Nginx (Reverse Proxy)

```bash
# Install Nginx
apt install nginx -y

# Create config
nano /etc/nginx/sites-available/fake-news
```

```nginx
# Frontend
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Backend API
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
ln -s /etc/nginx/sites-available/fake-news /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

### Step 6: Add SSL with Let's Encrypt

```bash
# Install Certbot
apt install certbot python3-certbot-nginx -y

# Get SSL certificates
certbot --nginx -d yourdomain.com -d www.yourdomain.com
certbot --nginx -d api.yourdomain.com

# Auto-renewal is configured automatically
certbot renew --dry-run
```

---

## Environment Configuration

### Production Environment Variables

#### Backend (.env)

```bash
# Application
APP_NAME=GTDS Fake News Detection API
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production

# Server
HOST=0.0.0.0
PORT=8000

# Database (use managed service)
DATABASE_URL=postgresql://user:pass@host:5432/gtds_db
DB_ECHO=false

# Redis (use managed service)
REDIS_URL=redis://host:6379/0

# Security - GENERATE NEW KEYS!
SECRET_KEY=<use: openssl rand -hex 32>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS (add your domains)
CORS_ORIGINS=["https://yourdomain.com","https://www.yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true

# ML Models
ML_MODEL_PATH=./backend/models/
CLASSIFICATION_THRESHOLD=0.523

# Logging
LOG_LEVEL=WARNING
LOG_FILE=/var/log/fake-news/app.log

# Monitoring (optional)
SENTRY_DSN=your-sentry-dsn
```

#### Frontend (.env.production)

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_APP_NAME=Fake News Detection
NEXT_PUBLIC_ENVIRONMENT=production
```

---

## Security Checklist

### Pre-Deployment

- [ ] **Change ALL default passwords**
- [ ] **Generate new SECRET_KEY** (`openssl rand -hex 32`)
- [ ] **Remove DEBUG mode** (`DEBUG=false`)
- [ ] **Set up CORS properly** (specific domains, not `*`)
- [ ] **Enable HTTPS/SSL** (Let's Encrypt or cloud provider)
- [ ] **Set secure cookie settings**
- [ ] **Rate limiting enabled**
- [ ] **SQL injection protection** (using ORMs)
- [ ] **Input validation** on all endpoints
- [ ] **Secrets in environment variables** (not in code)

### Post-Deployment

- [ ] **Set up monitoring** (Sentry, LogRocket)
- [ ] **Configure backups** (database snapshots)
- [ ] **Set up alerts** (uptime monitoring)
- [ ] **Enable firewall** (ufw or cloud security groups)
- [ ] **Regular security updates** (`apt update && apt upgrade`)
- [ ] **Implement logging** (access logs, error logs)
- [ ] **Set up CI/CD** (automated deployments)

---

## Monitoring & Maintenance

### Monitoring Tools

```bash
# Application Monitoring
Sentry: Error tracking
LogRocket: Session replay
DataDog: Full stack monitoring

# Uptime Monitoring
UptimeRobot: Free uptime checks
Pingdom: Advanced monitoring
StatusCake: Global monitoring

# Infrastructure Monitoring
Grafana + Prometheus: Metrics
ELK Stack: Logs aggregation
```

### Health Checks

Add health check endpoints:

```python
# backend/app/main.py
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

### Backup Strategy

```bash
# Database backups (cron job)
0 2 * * * pg_dump -U postgres gtds_db | gzip > /backups/db_$(date +\%Y\%m\%d).sql.gz

# Keep last 7 days
find /backups -name "db_*.sql.gz" -mtime +7 -delete
```

### Scaling Considerations

**When to scale:**
- Response time > 500ms
- CPU usage > 70%
- Memory usage > 80%
- Database connections > 80% of max

**How to scale:**

1. **Vertical Scaling:**
   - Upgrade server size
   - Add more RAM/CPU

2. **Horizontal Scaling:**
   - Add more backend instances
   - Use load balancer (Nginx, AWS ELB)
   - Database read replicas

3. **Caching:**
   - Redis for predictions cache
   - CDN for static assets (Cloudflare)

---

## Cost Estimates

### Small Scale (< 1000 users)

```
Vercel (Frontend):           $0-20/mo
Render (Backend):            $7/mo
Render (PostgreSQL):         $7/mo
Render (Redis):              $7/mo
Domain:                      $12/year
Total:                       ~$21-35/mo
```

### Medium Scale (1000-10,000 users)

```
Vercel Pro:                  $20/mo
Render Standard:             $25/mo
Managed PostgreSQL:          $15/mo
Managed Redis:               $10/mo
Monitoring (Sentry):         $26/mo
CDN (Cloudflare):            $0-20/mo
Total:                       ~$96-116/mo
```

### Large Scale (10,000+ users)

```
AWS/GCP Infrastructure:      $200-500/mo
Database (managed):          $100-200/mo
CDN & Assets:                $50-100/mo
Monitoring & Logging:        $50-100/mo
Total:                       ~$400-900/mo
```

---

## Quick Start Commands

### Deploy to Vercel + Render (Recommended)

```bash
# 1. Push to GitHub
git push origin main

# 2. Deploy frontend
cd frontend && vercel --prod

# 3. Deploy backend on Render
# Use dashboard: https://dashboard.render.com

# 4. Update environment variables
# Set NEXT_PUBLIC_API_URL in Vercel
# Set DATABASE_URL, REDIS_URL in Render

# Done! 🎉
```

### Deploy to DigitalOcean (Docker)

```bash
# 1. Create droplet & SSH
ssh root@your-droplet-ip

# 2. Install Docker
curl -fsSL https://get.docker.com | sh

# 3. Clone & configure
git clone <your-repo>
cd fake-news-detection
cp backend/.env.example backend/.env
nano backend/.env

# 4. Deploy
docker-compose -f docker-compose.prod.yml up -d

# 5. Set up Nginx + SSL
apt install nginx certbot python3-certbot-nginx
certbot --nginx -d yourdomain.com

# Done! 🎉
```

---

## Troubleshooting

### Common Issues

**Issue:** "Cannot connect to database"
```bash
# Check DATABASE_URL format
# Should be: postgresql://user:pass@host:port/db

# Test connection
psql $DATABASE_URL -c "SELECT 1;"
```

**Issue:** "CORS errors in browser"
```bash
# Check CORS_ORIGINS in backend .env
# Must include your frontend domain
CORS_ORIGINS=["https://yourdomain.com"]
```

**Issue:** "Module not found" errors
```bash
# Rebuild with fresh dependencies
docker-compose build --no-cache
```

**Issue:** "Out of memory"
```bash
# Increase server RAM or
# Optimize model loading (lazy loading)
```

---

## Next Steps

After deployment:

1. ✅ Set up monitoring (Sentry, UptimeRobot)
2. ✅ Configure automated backups
3. ✅ Add custom domain
4. ✅ Set up CI/CD pipeline
5. ✅ Load test application
6. ✅ Create runbooks for common issues
7. ✅ Document API for users

---

## Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Render Documentation](https://render.com/docs)
- [DigitalOcean Tutorials](https://www.digitalocean.com/community/tutorials)
- [Docker Documentation](https://docs.docker.com)
- [Let's Encrypt](https://letsencrypt.org)

---

**Last Updated:** October 4, 2025  
**Maintained By:** Project Team
