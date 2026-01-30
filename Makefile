# WAF Project Makefile
# Docker commands for easy management

.PHONY: help build up down logs shell test clean

# Default target
help:
	@echo "WAF Project - Docker Commands"
	@echo "=============================="
	@echo ""
	@echo "Development:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs from all services"
	@echo "  make restart    - Restart all services"
	@echo ""
	@echo "Individual Services:"
	@echo "  make up-db      - Start only database services"
	@echo "  make up-backend - Start backend service"
	@echo "  make up-frontend- Start frontend service"
	@echo ""
	@echo "Production:"
	@echo "  make prod       - Start with nginx (production profile)"
	@echo "  make prod-down  - Stop production services"
	@echo ""
	@echo "Development Tools:"
	@echo "  make shell      - Open shell in backend container"
	@echo "  make test       - Run tests"
	@echo "  make train      - Start training service"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove containers and volumes"
	@echo "  make prune      - Remove all unused Docker resources"

# ============================================
# BUILD
# ============================================

build:
	docker compose build

build-backend:
	docker compose build backend

build-frontend:
	docker compose build frontend

build-no-cache:
	docker compose build --no-cache

# ============================================
# DEVELOPMENT
# ============================================

up:
	docker compose up -d postgres redis
	@echo "Waiting for database..."
	@sleep 5
	docker compose up -d backend frontend
	@echo ""
	@echo "Services started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:3001"
	@echo "  API Docs: http://localhost:3001/docs"

up-db:
	docker compose up -d postgres redis

up-backend:
	docker compose up -d backend

up-frontend:
	docker compose up -d frontend

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-frontend:
	docker compose logs -f frontend

# ============================================
# PRODUCTION
# ============================================

prod:
	docker compose --profile production up -d

prod-down:
	docker compose --profile production down

# ============================================
# DEVELOPMENT TOOLS
# ============================================

shell:
	docker compose --profile debug run --rm shell

shell-backend:
	docker compose exec backend bash

shell-frontend:
	docker compose exec frontend sh

shell-db:
	docker compose exec postgres psql -U waf_user -d waf_db

test:
	docker compose exec backend python -m pytest tests/ -v

train:
	docker compose --profile training run --rm train python scripts/finetune_waf_model.py

# ============================================
# STATUS
# ============================================

status:
	docker compose ps

health:
	@echo "Backend health:"
	@curl -s http://localhost:3001/health | python -m json.tool || echo "Backend not responding"
	@echo ""
	@echo "Frontend:"
	@curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:3000 || echo "Frontend not responding"

# ============================================
# MAINTENANCE
# ============================================

clean:
	docker compose down -v --remove-orphans
	docker compose rm -f

prune:
	docker system prune -af
	docker volume prune -f

# ============================================
# DATABASE
# ============================================

db-migrate:
	docker compose exec backend alembic upgrade head

db-backup:
	docker compose exec postgres pg_dump -U waf_user waf_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@echo "Usage: cat backup.sql | docker compose exec -T postgres psql -U waf_user -d waf_db"
