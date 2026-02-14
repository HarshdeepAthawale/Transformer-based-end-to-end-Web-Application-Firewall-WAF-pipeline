---
name: Dashboard UI Positivus Theme
overview: Apply the Positivus landing/login design system (acidic green, black, white, Space Grotesk) to the WAF dashboard so the UI is visually consistent across the app.
todos: []
isProject: false
---

# Dashboard UI/UX Alignment with Login and Landing Page

## Current State

- **Landing/Login**: Positivus theme (acidic green #C5E246, black #191A23, white, gray), Space Grotesk font, border-2 cards, sharp corners (rounded-none buttons)
- **Dashboard**: Generic design system (bg-card, bg-sidebar, muted grays), Inter font, rounded-lg, different visual language

## Design Tokens to Apply


| Element    | Current                | Target                                                |
| ---------- | ---------------------- | ----------------------------------------------------- |
| Sidebar BG | Dark (sidebar)         | Light: --positivus-white with --positivus-gray border |
| Active nav | bg-black               | --positivus-green background                          |
| Main BG    | bg-background          | --positivus-gray or --positivus-white                 |
| Cards      | bg-card, border-border | White with --positivus-gray border-2                  |
| Accents    | Various                | --positivus-green for highlights                      |
| Typography | Inter                  | Space Grotesk for headings                            |
| Buttons    | rounded-md             | rounded-none (sharp)                                  |


## Implementation Plan

### 1. Update global design tokens

In [frontend/app/globals.css](frontend/app/globals.css): Map the existing semantic variables to Positivus palette so dashboard components using `bg-card`, `bg-background`, etc. inherit the new look without touching every class:

- `--background` → `var(--positivus-gray)` (main content area)
- `--card` → `var(--positivus-white)`
- `--sidebar` → `var(--positivus-white)`
- `--border` → `var(--positivus-gray)` (lighter, softer)
- `--primary` → `var(--positivus-black)`
- Ensure Space Grotesk is applied to headings (already in layout)

### 2. Restyle Sidebar

[frontend/components/sidebar.tsx](frontend/components/sidebar.tsx):

- Replace `bg-sidebar` with `bg-[var(--positivus-white)]`, `border-[var(--positivus-gray)]`
- Logo area: shield on `--positivus-green-bg` with green icon (match landing)
- Active nav item: `bg-[var(--positivus-green)]` with black text instead of black bg/white text
- Hover: `bg-[var(--positivus-green-bg)]`
- Nav items: `rounded-none` or minimal rounding
- Mobile overlay and toggle: adjust for light theme
- Footer (Settings, Logout): same styling
- Add `font-[family-name:var(--font-space-grotesk)]` for nav labels

### 3. Restyle Header

[frontend/components/header.tsx](frontend/components/header.tsx):

- Background: `bg-[var(--positivus-white)]`, `border-[var(--positivus-gray)]`
- Title: Space Grotesk, `--positivus-black`
- Search input: `border-2` with `--positivus-gray`, `rounded-none`
- Time range select: same border treatment
- Bell/Settings icons: `--positivus-black` or gray-dark
- Overall layout and spacing preserved

### 4. Restyle Metrics Overview

[frontend/components/metrics-overview.tsx](frontend/components/metrics-overview.tsx):

- Card: `border-2 border-[var(--positivus-gray)]`, `bg-[var(--positivus-white)]`
- Icon containers: `bg-[var(--positivus-green-bg)]` with `--positivus-green` icon
- Labels: `--positivus-gray-dark`
- Values: `--positivus-black`
- Live indicator dot: `--positivus-green`
- Trend arrows: use `--positivus-green` for positive, keep red for critical
- `rounded-lg` → `rounded-md` or keep subtle rounding for metric cards

### 5. Restyle Charts Section

[frontend/components/charts-section.tsx](frontend/components/charts-section.tsx):

- Card containers: `border-2 border-[var(--positivus-gray)]`, `bg-[var(--positivus-white)]`
- Section titles: Space Grotesk, `--positivus-black`
- Empty state: dashed border `--positivus-gray`, muted text `--positivus-gray-dark`
- Chart colors: include `--positivus-green` in the color palette (e.g. chart-4 or custom)
- Tooltips: white bg, gray border (Positivus-style)

### 6. Restyle Alerts and Activity Feed

[frontend/components/alerts-section.tsx](frontend/components/alerts-section.tsx) and [frontend/components/activity-feed.tsx](frontend/components/activity-feed.tsx):

- Card: `border-2 border-[var(--positivus-gray)]`, `bg-[var(--positivus-white)]`
- Section headers: Space Grotesk, `--positivus-black`
- Alert/activity items: subtle dividers with `--positivus-gray`
- Icons: use `--positivus-green` for positive/safe, keep red for critical
- Loading/empty states: `--positivus-gray-dark` text

### 7. Dashboard layout background

[frontend/app/dashboard/page.tsx](frontend/app/dashboard/page.tsx):

- Change `bg-background` to `bg-[var(--positivus-gray)]` for main content area if not already applied via globals

### 8. Shared dashboard layout (other pages)

Pages like analytics, traffic, threats reuse Sidebar + Header. They use `bg-card`, `border-border` etc. Once globals are updated, these will pick up the new tokens. Spot-check [frontend/app/analytics/page.tsx](frontend/app/analytics/page.tsx) for any hardcoded colors.

## File Summary


| File                   | Changes                                        |
| ---------------------- | ---------------------------------------------- |
| `globals.css`          | Map semantic vars to Positivus palette         |
| `sidebar.tsx`          | Full restyle (light theme, green active, logo) |
| `header.tsx`           | Border, typography, colors                     |
| `metrics-overview.tsx` | Card styling, icon colors                      |
| `charts-section.tsx`   | Cards, titles, chart colors                    |
| `alerts-section.tsx`   | Card and text colors                           |
| `activity-feed.tsx`    | Card and text colors                           |
| `dashboard/page.tsx`   | Background if needed                           |


## Visual Consistency Checklist

- Sidebar: light background, green active state
- Cards: white, 2px gray border
- Headings: Space Grotesk
- Accents: acidic green for positive/safe/active
- Sharp or minimal rounding on buttons/inputs
- Consistent spacing and padding

