# MEGA PLAN 3: Production Deployment & Operational Readiness

**Focus:** Make this deployable and operable in real production environments
**Completion lift:** 97% -> 100%
**Key metric:** Deploy: Docker -> K8s
**Critical fix:** Continuous learning loop
**Dependencies:** Mega Plan 2 (CI/CD needed before K8s)
**Estimated scope:** ~50 files created

---

## Phase 3A — Kubernetes Deployment

| Step | Action | Files |
|------|--------|-------|
| 1 | Namespace & RBAC | `k8s/namespace.yaml`, `k8s/rbac.yaml` |
| 2 | Backend deployment | `k8s/backend/deployment.yaml` — 2 replicas, health probes, resource limits, configmap mount |
| 3 | Frontend deployment | `k8s/frontend/deployment.yaml` — 2 replicas, Nginx serving |
| 4 | Gateway deployment | `k8s/gateway/deployment.yaml` — 3+ replicas (high traffic), anti-affinity |
| 5 | Redis | `k8s/redis/statefulset.yaml` — persistent volume, sentinel for HA |
| 6 | MongoDB | `k8s/mongodb/statefulset.yaml` — persistent volume |
| 7 | PostgreSQL | `k8s/postgres/statefulset.yaml` — persistent volume, backups |
| 8 | Ingress | `k8s/ingress.yaml` — TLS termination, routing rules, rate limit annotations |
| 9 | HPA | `k8s/backend/hpa.yaml`, `k8s/gateway/hpa.yaml` — CPU/memory autoscaling |
| 10 | Secrets template | `k8s/secrets.yaml.template` — JWT keys, DB passwords, Redis auth |
| 11 | Helm chart (optional) | `helm/` — parameterized deployment for different environments |
| 12 | Validate on minikube | Deploy, verify all pods healthy, run smoke tests |

---

## Phase 3B — Continuous Learning Pipeline

| Step | Action | Details |
|------|--------|---------|
| 1 | Feedback API endpoint | `POST /api/feedback` — operators mark FP/FN with corrected labels |
| 2 | Labeled sample storage | Store in `feedback_samples` DB table with timestamp, request, label, source |
| 3 | Export pipeline | Script to export labeled samples to training format |
| 4 | Automated retraining trigger | When 1000+ new labeled samples accumulate, trigger retraining |
| 5 | A/B model validation | Run new model on shadow traffic, compare accuracy to current |
| 6 | Canary deployment | Gradually shift traffic to new model (10% -> 50% -> 100%) |
| 7 | Rollback on regression | Auto-rollback if accuracy drops >5% from baseline |

---

## Phase 3C — Monitoring & Alerting Polish

| Step | Action | Details |
|------|--------|---------|
| 1 | Grafana dashboards | Pre-built dashboards: WAF overview, attack trends, model performance, system health |
| 2 | Alert rules | Prometheus alerting rules: high attack rate, model latency spike, service down, disk full |
| 3 | PagerDuty/Slack integration | Notification channels for critical alerts |
| 4 | SLA dashboard | Uptime, latency percentiles, error rates |
| 5 | Log aggregation | Structured JSON logs -> ELK/Loki for centralized search |

---

## Phase 3D — Docker Compose Apps File

| Step | Action | Details |
|------|--------|---------|
| 1 | Populate `docker-compose.apps.yml` | Define Juice Shop, WebGoat, DVWA services (currently empty/0 bytes) |
| 2 | Network configuration | Connect apps to gateway network for live WAF testing |
| 3 | Data collection mode | Script to crawl apps and generate fresh benign training data |

---

## Phase 3E — Hardening & Polish

| Step | Action | Details |
|------|--------|---------|
| 1 | TLS everywhere | HTTPS for gateway, backend, frontend; mTLS between internal services |
| 2 | Secret management | No hardcoded secrets; use Docker secrets or Vault |
| 3 | Database migrations | Alembic migration scripts for schema changes |
| 4 | Backup/restore | Automated DB backups, model checkpoint backups |
| 5 | Jupyter notebooks | Add 3-5 notebooks to `/notebooks/`: model evaluation, data exploration, threshold analysis, attack pattern visualization |
| 6 | Security audit | OWASP self-check on the WAF's own endpoints (auth, injection, CSRF) |
| 7 | README polish | Badges (CI, coverage, license), architecture diagram update, GIF demo |

---

## Deliverables

- [ ] Full Kubernetes manifests (+ optional Helm chart)
- [ ] Continuous learning pipeline with feedback loop
- [ ] Grafana dashboards + Prometheus alert rules
- [ ] Populated `docker-compose.apps.yml`
- [ ] TLS, secret management, DB migrations
- [ ] Jupyter notebooks for analysis
- [ ] Production-hardened, deployable system
