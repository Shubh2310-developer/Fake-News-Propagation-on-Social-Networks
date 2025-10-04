# Database Setup Guide

## ✅ Database Configuration Fixed

The PostgreSQL password has been updated in the backend configuration.

### Configuration Details

**Location:** `backend/.env`

```env
DATABASE_URL=postgresql://postgres:Meghal0987@23@localhost:5432/gtds_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gtds_db
DB_USER=postgres
DB_PASSWORD=Meghal0987@23
```

## Quick Setup

### Option 1: Automatic (Recommended)

```bash
./scripts/init-db.sh
```

This script will:
- ✅ Create the `gtds_db` database
- ✅ Run migrations (if available)
- ✅ Initialize tables
- ✅ Verify connection

### Option 2: Manual Setup

**1. Create Database:**
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -c "CREATE DATABASE gtds_db;"
```

**2. Verify Database:**
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -c "\l gtds_db"
```

**3. Initialize Tables:**
```bash
cd backend
conda activate fake_news
python -c "
import asyncio
from app.core.database import init_db
asyncio.run(init_db())
"
```

## Verification

### Check Database Exists
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -l | grep gtds_db
```

### Test Connection from Backend
```bash
cd backend
conda activate fake_news
python -c "
import asyncio
from app.core.database import engine
from sqlalchemy import text

async def test():
    async with engine.begin() as conn:
        result = await conn.execute(text('SELECT 1'))
        print('✓ Database connection successful!')

asyncio.run(test())
"
```

### List Tables
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -d gtds_db -c "\dt"
```

## Restart Backend

After database setup, restart the backend:

```bash
# Stop current backend (Ctrl+C if running)
pkill -f "uvicorn app.main:app"

# Start again
./start.sh
```

## Expected Startup Logs

**Before Fix (with error):**
```
⚠️  Database connection failed: password authentication failed for user "postgres"
⚠️  Running without database - some features may be limited
```

**After Fix (successful):**
```
[INFO] Starting application...
✅ Database connected successfully
✅ Redis connected successfully
[INFO] ✅ Application startup complete.
```

## Troubleshooting

### Issue: "password authentication failed"

**Check password:**
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -c "SELECT 1;"
```

If this fails, check PostgreSQL authentication settings:
```bash
sudo nano /etc/postgresql/*/main/pg_hba.conf
```

Ensure there's a line like:
```
local   all             postgres                                md5
host    all             postgres        127.0.0.1/32            md5
```

Then restart PostgreSQL:
```bash
sudo service postgresql restart
```

### Issue: "database does not exist"

**Create it:**
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -c "CREATE DATABASE gtds_db;"
```

### Issue: PostgreSQL not running

**Start PostgreSQL:**
```bash
sudo service postgresql start
```

**Check status:**
```bash
sudo service postgresql status
```

### Issue: Connection timeout

**Check PostgreSQL is listening:**
```bash
sudo netstat -tuln | grep 5432
```

Should show:
```
tcp        0      0 127.0.0.1:5432          0.0.0.0:*               LISTEN
```

## Database Schema

The application will automatically create these tables:

- `simulations` - Simulation run metadata and results
- `networks` - Network topology data
- `players` - Game theory players/agents
- `equilibria` - Nash equilibrium calculations
- `game_states` - Historical game states
- `analytics` - Analysis results

## Production Considerations

### Security

**For production, update `.env` with:**
```env
# Use a strong password
DB_PASSWORD=your-strong-production-password

# Use environment-specific database
DB_NAME=gtds_production_db

# Enable SSL
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### Performance

**Add connection pooling:**
```env
# In backend/.env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
```

### Backups

**Create backup:**
```bash
PGPASSWORD='Meghal0987@23' pg_dump -U postgres -h localhost gtds_db > backup.sql
```

**Restore backup:**
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost gtds_db < backup.sql
```

## Testing Database

### Run Test Suite
```bash
cd backend
conda activate fake_news
pytest tests/test_database.py -v
```

### Manual Test
```python
# test_db.py
import asyncio
from app.core.database import get_db_session
from sqlalchemy import text

async def test():
    async for session in get_db_session():
        result = await session.execute(text("SELECT current_database()"))
        db_name = result.scalar()
        print(f"✓ Connected to database: {db_name}")

        # Test table creation
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100)
            )
        """))
        print("✓ Table creation successful")

        # Test insert
        await session.execute(text("""
            INSERT INTO test_table (name) VALUES ('test')
        """))
        await session.commit()
        print("✓ Insert successful")

        # Test select
        result = await session.execute(text("""
            SELECT * FROM test_table
        """))
        rows = result.fetchall()
        print(f"✓ Query successful: {len(rows)} rows")

asyncio.run(test())
```

## Environment Variables Reference

```env
# Primary connection string (used by SQLAlchemy)
DATABASE_URL=postgresql://postgres:Meghal0987@23@localhost:5432/gtds_db

# Individual components (for manual connections)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gtds_db
DB_USER=postgres
DB_PASSWORD=Meghal0987@23

# Connection pool settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_PRE_PING=true
DB_POOL_RECYCLE=3600

# Debug options
DB_ECHO=false  # Set to true to log all SQL queries
```

## Next Steps

1. ✅ Database configured with correct password
2. ✅ Database `gtds_db` created
3. ⏳ Restart backend to connect
4. ⏳ Run a simulation to test database integration

**To complete setup:**
```bash
# 1. Restart the application
./start.sh

# 2. Check logs for successful connection
tail -f logs/backend.log

# 3. Test by running a simulation
# Open http://localhost:3000/simulation
```

---

**Status:** ✅ Database Ready

**Last Updated:** 2025-10-02
