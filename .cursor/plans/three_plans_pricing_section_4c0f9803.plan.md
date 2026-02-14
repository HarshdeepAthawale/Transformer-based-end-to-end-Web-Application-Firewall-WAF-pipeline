---
name: Three plans pricing section
overview: Replace the single "Pro" pricing card with three market-aligned tiers (Free, Pro, Business) in the landing pricing section, keeping the existing Positivus styling and copy tone.
todos: []
isProject: false
---

# Three Plans Pricing Section (Market-Aligned)

## Market context

- **WAF/security**: Cloudflare uses Free → Pro → Business → Enterprise; AWS/Azure use usage-based. Tiered plans with clear feature limits are the norm for product-led WAF/SaaS.
- **SaaS norms**: Free (entry, limited), Pro (full features, growth), Business/Enterprise (advanced security, support, “Contact sales”).
- **Your product**: AI threat detection, real-time analytics, bot protection, geo rules & IP management are the core differentiators and should scale across tiers (limited in Free, full in Pro/Business).

## Recommended 3-tier structure (market-aligned)


| Tier         | Price                          | Positioning                 | Differentiators                                                                                                                          |
| ------------ | ------------------------------ | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Free**     | $0 — “Get started free”        | Try the product, single app | 1 protected app, basic threat detection, 7-day analytics, community/email support                                                        |
| **Pro**      | “Free to start” or e.g. $29/mo | Growth / default choice     | AI threat detection, real-time analytics, bot protection, geo rules, unlimited requests, email support — **highlight as “Most popular”** |
| **Business** | “Contact sales”                | Teams & compliance          | Everything in Pro + SSO/SAML, priority support, SLA, custom rules / dedicated success                                                    |


Backend has no plan/subscription model yet; limits (apps, retention, etc.) are not enforced. So this is **UI/marketing only**: copy and feature lists per tier. Enforcement can be added later if you introduce plans in the backend.

## Implementation

**Single file to change:** [frontend/components/landing/pricing-section.tsx](frontend/components/landing/pricing-section.tsx)

1. **Data**
  - Define a `plans` array (e.g. `Free`, `Pro`, `Business`) with for each:
    - `name`, `price` (string: `"$0"`, `"Free to start"`, `"Contact sales"`), `description` (optional)
    - `features`: list of strings; support “included” vs “not included” (e.g. “1 protected app” for Free, “Unlimited requests” only for Pro/Business) so the matrix is accurate
    - `cta`: label + href (`/dashboard` for Free/Pro, `#contact` or `/login` for Business)
    - `highlight`: boolean for Pro (“Most popular”) for optional border/badge
2. **Layout**
  - Replace the single `max-w-lg mx-auto` card with a responsive grid: e.g. `grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8`, or 3 columns on large screens with the middle (Pro) card slightly elevated (e.g. scale/border) if desired.
3. **Styling**
  - Reuse existing Positivus theme: `bg-[var(--positivus-gray)]`, heading `var(--positivus-green)`, cards `var(--positivus-white)` with `border-2` `var(--positivus-black)`, Check icon `var(--positivus-green)`, primary CTA `var(--positivus-black)` background. Keep “Simple pricing” and “Start free. Scale as you grow. No surprises.” as-is.
4. **Copy**
  - Free: 1 app, basic threat detection, 7-day analytics, community/email support; CTA “Get started free” → `/dashboard`.
  - Pro: current 6 bullets (AI threat detection, real-time analytics, bot protection, geo rules & IP management, unlimited requests, email support); CTA “Get started free” or “Start free” → `/dashboard`.
  - Business: Pro features + “SSO / SAML”, “Priority support”, “SLA & compliance”; CTA “Contact sales” → `#contact` or contact section link.
5. **Optional**
  - “Most popular” badge on Pro (small label above or below the plan name).
  - Secondary style for Business CTA (outline instead of filled) to signal “Contact” vs “Start free”.

No new components or API calls; no backend changes. Keeps the section self-contained and easy to tweak (prices, feature list) later.

## Summary

- **Scope**: One file — `pricing-section.tsx`.
- **Content**: 3 plans (Free, Pro, Business) with market-aligned names, prices, and feature lists; Pro as the highlighted default.
- **Design**: Same Positivus look; 3-card grid; optional “Most popular” and outline CTA for Business.
- **Backend**: No changes; plan enforcement can be added in a future iteration.

